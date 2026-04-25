"""
Script 2: XSA Intervention — Apply XSA to Encoder Self-Attention + LoRA Retraining
=====================================================================================
Patches NLLB-200's encoder self-attention with the XSA projection step,
then retrains LoRA adapters and evaluates BLEU/chrF on all six translation
directions from your nios_test.csv.

Experiment design (mirrors the XSA paper's ablation approach):
  Baseline   : NLLB-200 + LoRA (standard SA)
  Intervention: NLLB-200 + XSA (encoder only) + LoRA

Usage:
    pip install transformers torch peft sacrebleu sentencepiece pandas tqdm
    python xsa_intervention.py \
        --data nios_test.csv \
        --model facebook/nllb-200-distilled-600M \
        --baseline_checkpoint ./lora_baseline \
        --output_dir ./xsa_intervention_output \
        --train_frac 0.8 \
        --epochs 3 \
        --lora_r 16 \
        --xsa_enc       # flag: apply XSA to encoder SA
        --xsa_dec     # optional: also patch decoder SA (for ablation in script 3)
"""

import argparse
import os
import json
import warnings
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from sacrebleu.metrics import BLEU, CHRF
from tqdm import tqdm

warnings.filterwarnings("ignore")

LANG_CODES = {
    "Hindi":    "hin_Deva",
    "Kannada":  "kan_Knda",
    "Sanskrit": "san_Deva",
}
LANG_NAMES_BY_CODE = {v: k for k, v in LANG_CODES.items()}

# Six translation directions
DIRECTIONS = [
    ("Sanskrit", "Hindi"),
    ("Sanskrit", "Kannada"),
    ("Hindi",    "Sanskrit"),
    ("Hindi",    "Kannada"),
    ("Kannada",  "Sanskrit"),
    ("Kannada",  "Hindi"),
]


def _normalise_lang_code(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    if value in LANG_CODES:
        return LANG_CODES[value]
    if value in LANG_NAMES_BY_CODE:
        return value
    return None


def _to_lang_code(series, field_name):
    mapped = series.map(_normalise_lang_code)
    if mapped.isna().any():
        bad_values = sorted(series[mapped.isna()].astype(str).unique().tolist())
        raise ValueError(
            f"Unknown values in {field_name}: {bad_values}. "
            f"Expected one of {sorted(LANG_CODES.keys())} or {sorted(LANG_NAMES_BY_CODE.keys())}."
        )
    return mapped


def _wide_to_direction_frame(df):
    required_cols = set(LANG_CODES.keys())
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(
            f"Input is missing required language columns: {missing}. "
            f"Expected columns: {sorted(required_cols)}"
        )

    parts = []
    for src_col, tgt_col in DIRECTIONS:
        rows = df[[src_col, tgt_col]].dropna().copy()
        rows = rows.rename(columns={src_col: "src_text", tgt_col: "tgt_text"})
        rows["src_lang"] = LANG_CODES[src_col]
        rows["tgt_lang"] = LANG_CODES[tgt_col]
        parts.append(rows[["src_lang", "tgt_lang", "src_text", "tgt_text"]])

    if not parts:
        return pd.DataFrame(columns=["src_lang", "tgt_lang", "src_text", "tgt_text"])
    return pd.concat(parts, ignore_index=True)


def _direction_frame_from_jsonl(df):
    directional_cols = {"src_lang", "tgt_lang", "src_text", "tgt_text"}
    if directional_cols.issubset(df.columns):
        out = df[["src_lang", "tgt_lang", "src_text", "tgt_text"]].dropna().copy()
        out["src_lang"] = _to_lang_code(out["src_lang"], "src_lang")
        out["tgt_lang"] = _to_lang_code(out["tgt_lang"], "tgt_lang")
        out["src_text"] = out["src_text"].astype(str)
        out["tgt_text"] = out["tgt_text"].astype(str)
        return out

    # Also accept JSONL where each row has Hindi/Kannada/Sanskrit columns (CSV-like shape).
    return _wide_to_direction_frame(df)


def load_directional_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        raw_df = pd.read_csv(path)
        direction_df = _wide_to_direction_frame(raw_df)
        data_format = "csv"
    elif ext == ".jsonl":
        raw_df = pd.read_json(path, lines=True)
        direction_df = _direction_frame_from_jsonl(raw_df)
        data_format = "jsonl"
    else:
        raise ValueError(
            f"Unsupported data file extension '{ext}'. Use .csv or .jsonl"
        )

    if direction_df.empty:
        raise ValueError("No usable training pairs were found in the input data.")

    direction_df = direction_df.reset_index(drop=True)
    return direction_df, data_format


def count_direction_samples(direction_df):
    counts = {}
    for src_name, tgt_name in DIRECTIONS:
        src_code = LANG_CODES[src_name]
        tgt_code = LANG_CODES[tgt_name]
        mask = (direction_df["src_lang"] == src_code) & (direction_df["tgt_lang"] == tgt_code)
        counts[(src_name, tgt_name)] = int(mask.sum())
    return counts


def print_direction_counts(title, counts):
    print(f"\n{title}")
    for src_name, tgt_name in DIRECTIONS:
        key = f"{src_name[:3]}→{tgt_name[:3]}"
        print(f"  {key}: {counts[(src_name, tgt_name)]}")


def ensure_all_directions_present(counts, split_name):
    missing = [
        f"{src}→{tgt}"
        for src, tgt in DIRECTIONS
        if counts[(src, tgt)] == 0
    ]
    if missing:
        raise ValueError(
            f"{split_name} is missing required translation directions: {missing}. "
            "Need non-zero samples for all 6 directions."
        )


def _split_cutoff(n_samples, train_frac):
    if n_samples <= 1:
        return n_samples
    raw_cut = int(n_samples * train_frac)
    return max(1, min(raw_cut, n_samples - 1))


# ─── XSA Patch ────────────────────────────────────────────────────────────

class XSAPatch(nn.Module):
    """
    Wraps an MBartAttention module and applies the XSA projection step
    to its self-attention output (before out_proj), removing the component
    along the self value vector.

        z_i = y_i - (y_i . v̂_i) * v̂_i

    where v̂_i = v_i / ||v_i||  (L2-normalised value vector)
    """

    def __init__(self, attn_module):
        super().__init__()
        self.attn = attn_module

    def forward(self, hidden_states, attention_mask=None, layer_head_mask=None,
                past_key_value=None, output_attentions=False, use_cache=True, **kwargs):

        # Run the original attention
        out = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        attn_output = out[0]  # (B, T, D)

        # Compute value vectors for self-position
        # v_proj: Linear(D -> D)
        B, T, D = hidden_states.shape
        v = self.attn.v_proj(hidden_states)  # (B, T, D)

        # XSA projection step (in full D space — equivalent to per-head)
        v_norm = F.normalize(v, dim=-1)           # (B, T, D)
        # scalar coefficient: (y_i . v̂_i)
        coeff = (attn_output * v_norm).sum(dim=-1, keepdim=True)  # (B, T, 1)
        # subtract the projection
        attn_output_xsa = attn_output - coeff * v_norm            # (B, T, D)

        return (attn_output_xsa,) + out[1:]


def apply_xsa(model, enc=True, dec=False):
    """Patch the model's attention modules in-place."""
    if enc:
        for layer in model.model.encoder.layers:
            layer.self_attn = XSAPatch(layer.self_attn)
        print("  XSA applied to encoder self-attention.")
    if dec:
        for layer in model.model.decoder.layers:
            layer.self_attn = XSAPatch(layer.self_attn)
        print("  XSA applied to decoder self-attention.")


# ─── Dataset ──────────────────────────────────────────────────────────────

class MTDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer, src_lang, tgt_lang,
                 max_src_len=128, max_tgt_len=128):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        self.tokenizer.src_lang = self.src_lang
        self.tokenizer.tgt_lang = self.tgt_lang
        enc = self.tokenizer(
            self.src_texts[idx],
            max_length=self.max_src_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        dec = self.tokenizer(
            text_target=self.tgt_texts[idx],
            max_length=self.max_tgt_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = dec["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         labels,
        }


def build_all_direction_dataset(direction_df, split="train", train_frac=0.8):
    """Pool all six directions from directional data for joint training/eval splits."""
    all_src, all_tgt, all_src_lang, all_tgt_lang = [], [], [], []
    split_counts = {}

    for src_name, tgt_name in DIRECTIONS:
        src_code = LANG_CODES[src_name]
        tgt_code = LANG_CODES[tgt_name]
        rows = direction_df[
            (direction_df["src_lang"] == src_code) &
            (direction_df["tgt_lang"] == tgt_code)
        ]
        cut = _split_cutoff(len(rows), train_frac)
        if split == "train":
            rows = rows.iloc[:cut]
        else:
            rows = rows.iloc[cut:]
        split_counts[(src_name, tgt_name)] = len(rows)

        all_src.extend(rows["src_text"].tolist())
        all_tgt.extend(rows["tgt_text"].tolist())
        all_src_lang.extend([src_code] * len(rows))
        all_tgt_lang.extend([tgt_code] * len(rows))

    return all_src, all_tgt, all_src_lang, all_tgt_lang, split_counts


class MultiDirectionDataset(Dataset):
    """Handles multiple (src_lang, tgt_lang) pairs in one dataset."""

    def __init__(self, src_texts, tgt_texts, src_langs, tgt_langs,
                 tokenizer, max_src_len=128, max_tgt_len=128):
        self.data = list(zip(src_texts, tgt_texts, src_langs, tgt_langs))
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text, tgt_text, src_lang, tgt_lang = self.data[idx]
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang
        enc = self.tokenizer(
            src_text,
            max_length=self.max_src_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        dec = self.tokenizer(
            text_target=tgt_text,
            max_length=self.max_tgt_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = dec["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        tgt_lang_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        return {
            "input_ids":       enc["input_ids"].squeeze(),
            "attention_mask":  enc["attention_mask"].squeeze(),
            "labels":          labels,
            "forced_bos_id":   torch.tensor(tgt_lang_id),
        }


# ─── Training ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="  train", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)


# ─── Evaluation ───────────────────────────────────────────────────────────

def evaluate_direction(model, tokenizer, src_texts, tgt_texts, src_lang, tgt_lang,
                       device, max_src_len=128, max_new_tokens=128, batch_size=8):
    bleu_metric = BLEU(effective_order=True)
    chrf_metric = CHRF()

    tokenizer.src_lang = src_lang
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    hypotheses, references = [], []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(src_texts), batch_size):
            batch_src = src_texts[i: i + batch_size]
            enc = tokenizer(
                batch_src,
                return_tensors="pt",
                max_length=max_src_len,
                truncation=True,
                padding=True,
            ).to(device)
            generated = model.generate(
                **enc,
                forced_bos_token_id=tgt_lang_id,
                max_new_tokens=max_new_tokens,
                num_beams=4,
            )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            hypotheses.extend(decoded)
            references.extend(tgt_texts[i: i + batch_size])

    bleu_score = bleu_metric.corpus_score(hypotheses, [references]).score
    chrf_score = chrf_metric.corpus_score(hypotheses, [references]).score
    return {"bleu": round(bleu_score, 2), "chrf": round(chrf_score, 2)}


def evaluate_all_directions(model, tokenizer, direction_df, device, split_start_frac=0.8,
                             max_src_len=128):
    results = {}
    for src_name, tgt_name in DIRECTIONS:
        src_code = LANG_CODES[src_name]
        tgt_code = LANG_CODES[tgt_name]
        rows = direction_df[
            (direction_df["src_lang"] == src_code) &
            (direction_df["tgt_lang"] == tgt_code)
        ]
        cut = _split_cutoff(len(rows), split_start_frac)
        rows = rows.iloc[cut:]
        src_texts = rows["src_text"].tolist()
        tgt_texts = rows["tgt_text"].tolist()
        key = f"{src_name[:3]}→{tgt_name[:3]}"

        if len(src_texts) == 0:
            print(f"  Evaluating {key}: skipped (0 test sentences after split).")
            results[key] = {"bleu": None, "chrf": None, "n_test": 0}
            continue

        print(f"  Evaluating {key} ({len(src_texts)} sentences)...")
        results[key] = evaluate_direction(
            model, tokenizer, src_texts, tgt_texts,
            src_code, tgt_code, device,
            max_src_len=max_src_len,
        )
        results[key]["n_test"] = len(src_texts)
        print(f"    BLEU: {results[key]['bleu']:.2f}  chrF: {results[key]['chrf']:.2f}")
    return results


# ─── LoRA Config ──────────────────────────────────────────────────────────

def build_lora_model(model, r=16, alpha=32, dropout=0.1):
    """Apply LoRA to query and value projections in all attention layers."""
    config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        # target q_proj and v_proj in all MBartAttention layers
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    return get_peft_model(model, config)


# ─── Main ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",                 default="nios_test.csv",
                   help="Path to input data (.csv or .jsonl)")
    p.add_argument("--csv",                  default=None,
                   help=argparse.SUPPRESS)  # backward-compatible alias
    p.add_argument("--model",                default="facebook/nllb-200-distilled-600M")
    p.add_argument("--baseline_checkpoint",  default=None,
                   help="Path to existing LoRA checkpoint (skips training if provided)")
    p.add_argument("--output_dir",           default="./xsa_intervention_output")
    p.add_argument("--train_frac",           type=float, default=0.8)
    p.add_argument("--epochs",               type=int,   default=3)
    p.add_argument("--batch_size",           type=int,   default=8)
    p.add_argument("--lr",                   type=float, default=5e-4)
    p.add_argument("--lora_r",               type=int,   default=16)
    p.add_argument("--max_src_len",          type=int,   default=128)
    p.add_argument("--max_tgt_len",          type=int,   default=128)
    p.add_argument("--xsa_enc",              action="store_true", default=False,
                   help="Apply XSA to encoder self-attention")
    p.add_argument("--xsa_dec",              action="store_true", default=False,
                   help="Apply XSA to decoder self-attention")
    p.add_argument("--device",               default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.csv:
        args.data = args.csv
    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_name = "xsa" if (args.xsa_enc or args.xsa_dec) else "baseline"
    print(f"\n=== Run: {run_name} ===")
    print(f"xsa_enc={args.xsa_enc}  xsa_dec={args.xsa_dec}  device={args.device}")

    direction_df, data_format = load_directional_data(args.data)
    print(f"\nLoaded {len(direction_df)} directional pairs from {args.data} ({data_format}).")
    all_counts = count_direction_samples(direction_df)
    print_direction_counts("Samples per direction in full dataset:", all_counts)
    ensure_all_directions_present(all_counts, "Input dataset")

    # ── load model & tokenizer
    print(f"\nLoading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, torch_dtype=torch.float32)

    # ── apply XSA patch before LoRA wrapping
    if args.xsa_enc or args.xsa_dec:
        print("\nApplying XSA patch...")
        apply_xsa(model, enc=args.xsa_enc, dec=args.xsa_dec)

    # ── load or train LoRA
    if args.baseline_checkpoint and os.path.exists(args.baseline_checkpoint):
        print(f"\nLoading LoRA from {args.baseline_checkpoint}...")
        model = PeftModel.from_pretrained(model, args.baseline_checkpoint)
    else:
        print("\nWrapping with LoRA...")
        model = build_lora_model(model, r=args.lora_r, alpha=args.lora_r * 2)
        model.print_trainable_parameters()

        # ── build dataset
        src_list, tgt_list, sl_list, tl_list, train_counts = build_all_direction_dataset(
            direction_df, split="train", train_frac=args.train_frac,
        )
        print_direction_counts(
            f"Training split counts (train_frac={args.train_frac}):", train_counts
        )
        ensure_all_directions_present(train_counts, "Training split")
        print(f"Training samples (all 6 directions): {len(src_list)}")
        train_ds = MultiDirectionDataset(
            src_list, tgt_list, sl_list, tl_list, tokenizer,
            args.max_src_len, args.max_tgt_len,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

        model.to(args.device)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr
        )
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
        )

        print(f"\nTraining for {args.epochs} epochs...")
        for epoch in range(args.epochs):
            loss = train_one_epoch(model, train_loader, optimizer, scheduler, args.device)
            print(f"  Epoch {epoch+1}/{args.epochs}  loss={loss:.4f}")

        ckpt_path = os.path.join(args.output_dir, f"lora_{run_name}")
        model.save_pretrained(ckpt_path)
        print(f"Saved LoRA checkpoint → {ckpt_path}")

    # ── evaluate
    model.to(args.device)
    print(f"\nEvaluating {run_name} on test split...")
    results = evaluate_all_directions(
        model, tokenizer, direction_df, args.device,
        split_start_frac=args.train_frac, max_src_len=args.max_src_len,
    )

    # ── save results
    out_file = os.path.join(args.output_dir, f"results_{run_name}.json")
    with open(out_file, "w") as f:
        json.dump({"run": run_name, "xsa_enc": args.xsa_enc, "xsa_dec": args.xsa_dec,
                   "scores": results}, f, indent=2)
    print(f"\nSaved results → {out_file}")
    print("\nSummary:")
    for direction, scores in results.items():
        if scores["bleu"] is None or scores["chrf"] is None:
            print(f"  {direction}: skipped (n_test={scores.get('n_test', 0)})")
        else:
            print(
                f"  {direction}: BLEU={scores['bleu']:.2f}  "
                f"chrF={scores['chrf']:.2f}  n_test={scores.get('n_test', 0)}"
            )


if __name__ == "__main__":
    main()
