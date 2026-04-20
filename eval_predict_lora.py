"""
Evaluate and predict using a saved XSA+LoRA checkpoint.

Usage:
    python eval_predict_lora.py \
        --lora_path ./xsa_intervention_output/lora_xsa \
        --data val_multilingual.jsonl \
        [--xsa_enc] [--xsa_dec] \
        [--output_dir ./eval_output] \
        [--batch_size 8] \
        [--max_samples 0]   # 0 = all
"""

import argparse
import json
import os
import warnings

import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from peft import PeftModel
from sacrebleu.metrics import BLEU, CHRF
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

warnings.filterwarnings("ignore")

LANG_CODES = {
    "Hindi":    "hin_Deva",
    "Kannada":  "kan_Knda",
    "Sanskrit": "san_Deva",
}
LANG_NAMES_BY_CODE = {v: k for k, v in LANG_CODES.items()}

DIRECTIONS = [
    ("Sanskrit", "Hindi"),
    ("Sanskrit", "Kannada"),
    ("Hindi",    "Sanskrit"),
    ("Hindi",    "Kannada"),
    ("Kannada",  "Sanskrit"),
    ("Kannada",  "Hindi"),
]


# ── XSA patch (must mirror xsa_intervention.py exactly) ───────────────────

class XSAPatch(nn.Module):
    def __init__(self, attn_module):
        super().__init__()
        self.attn = attn_module

    def forward(self, hidden_states, attention_mask=None, layer_head_mask=None,
                past_key_value=None, output_attentions=False, use_cache=True, **kwargs):
        out = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        attn_output = out[0]
        B, T, D = hidden_states.shape
        v = self.attn.v_proj(hidden_states)
        v_norm = F.normalize(v, dim=-1)
        coeff = (attn_output * v_norm).sum(dim=-1, keepdim=True)
        attn_output_xsa = attn_output - coeff * v_norm
        return (attn_output_xsa,) + out[1:]


def apply_xsa(model, enc=True, dec=False):
    if enc:
        for layer in model.model.encoder.layers:
            layer.self_attn = XSAPatch(layer.self_attn)
        print("  XSA applied to encoder self-attention.")
    if dec:
        for layer in model.model.decoder.layers:
            layer.self_attn = XSAPatch(layer.self_attn)
        print("  XSA applied to decoder self-attention.")


# ── Data loading ───────────────────────────────────────────────────────────

def load_jsonl(path):
    df = pd.read_json(path, lines=True)
    required = {"src_lang", "tgt_lang", "src_text", "tgt_text"}
    if not required.issubset(df.columns):
        raise ValueError(f"JSONL must have columns: {required}. Found: {set(df.columns)}")
    df = df[list(required)].dropna().reset_index(drop=True)
    df["src_text"] = df["src_text"].astype(str)
    df["tgt_text"] = df["tgt_text"].astype(str)
    return df


# ── Inference ──────────────────────────────────────────────────────────────

def predict_batch(model, tokenizer, src_texts, src_lang, tgt_lang,
                  device, max_src_len=128, max_new_tokens=128, batch_size=8):
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    tokenizer.src_lang = src_lang
    hypotheses = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(src_texts), batch_size):
            batch = src_texts[i: i + batch_size]
            enc = tokenizer(
                batch,
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
    return hypotheses


def evaluate_and_predict(model, tokenizer, df, device,
                          max_src_len=128, batch_size=8, max_samples=0):
    bleu_metric = BLEU(effective_order=True)
    chrf_metric = CHRF()

    all_results = {}   # per-direction metrics
    all_rows = []      # flat prediction rows

    for src_name, tgt_name in DIRECTIONS:
        src_code = LANG_CODES[src_name]
        tgt_code = LANG_CODES[tgt_name]
        mask = (df["src_lang"] == src_code) & (df["tgt_lang"] == tgt_code)
        subset = df[mask].reset_index(drop=True)

        direction_key = f"{src_name[:3]}→{tgt_name[:3]}"

        if len(subset) == 0:
            print(f"  [{direction_key}] skipped (0 samples)")
            all_results[direction_key] = {"bleu": None, "chrf": None, "n": 0}
            continue

        if max_samples > 0:
            subset = subset.iloc[:max_samples]

        src_texts = subset["src_text"].tolist()
        tgt_texts = subset["tgt_text"].tolist()

        print(f"  [{direction_key}] predicting {len(src_texts)} sentences...")
        hypotheses = predict_batch(
            model, tokenizer, src_texts, src_code, tgt_code,
            device, max_src_len=max_src_len, batch_size=batch_size,
        )

        bleu = bleu_metric.corpus_score(hypotheses, [tgt_texts]).score
        chrf = chrf_metric.corpus_score(hypotheses, [tgt_texts]).score
        all_results[direction_key] = {
            "bleu": round(bleu, 2),
            "chrf": round(chrf, 2),
            "n":    len(src_texts),
        }
        print(f"    BLEU={bleu:.2f}  chrF={chrf:.2f}")

        for src, ref, hyp in zip(src_texts, tgt_texts, hypotheses):
            all_rows.append({
                "src_lang":    src_code,
                "tgt_lang":    tgt_code,
                "direction":   direction_key,
                "src_text":    src,
                "reference":   ref,
                "hypothesis":  hyp,
            })

    return all_results, all_rows


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lora_path",   required=True,
                   help="Path to saved LoRA adapter directory")
    p.add_argument("--data",        required=True,
                   help="Path to evaluation JSONL file")
    p.add_argument("--model",       default=None,
                   help="Base model ID (auto-read from adapter_config.json if omitted)")
    p.add_argument("--output_dir",  default="./eval_output")
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--max_src_len", type=int,   default=128)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--max_samples", type=int,   default=0,
                   help="Cap samples per direction (0 = all)")
    p.add_argument("--xsa_enc",     action="store_true", default=False,
                   help="Apply XSA patch to encoder SA (must match training)")
    p.add_argument("--xsa_dec",     action="store_true", default=False,
                   help="Apply XSA patch to decoder SA (must match training)")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-read base model from adapter config if not provided
    adapter_cfg_path = os.path.join(args.lora_path, "adapter_config.json")
    if args.model is None:
        with open(adapter_cfg_path) as f:
            adapter_cfg = json.load(f)
        args.model = adapter_cfg["base_model_name_or_path"]
    print(f"Base model : {args.model}")
    print(f"LoRA path  : {args.lora_path}")
    print(f"Data       : {args.data}")
    print(f"Device     : {args.device}")
    print(f"xsa_enc={args.xsa_enc}  xsa_dec={args.xsa_dec}")

    # Load tokenizer + base model
    print("\nLoading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(args.model, torch_dtype=torch.float32)

    # Apply XSA patch before loading LoRA (must match training order)
    if args.xsa_enc or args.xsa_dec:
        print("Applying XSA patch...")
        apply_xsa(base_model, enc=args.xsa_enc, dec=args.xsa_dec)

    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model.to(args.device)
    model.eval()

    # Load evaluation data
    print(f"\nLoading data from {args.data}...")
    df = load_jsonl(args.data)
    print(f"Loaded {len(df)} samples.")

    # Direction counts
    print("\nSamples per direction in eval file:")
    for src_name, tgt_name in DIRECTIONS:
        src_code = LANG_CODES[src_name]
        tgt_code = LANG_CODES[tgt_name]
        n = ((df["src_lang"] == src_code) & (df["tgt_lang"] == tgt_code)).sum()
        print(f"  {src_name[:3]}→{tgt_name[:3]}: {n}")

    # Evaluate + predict
    print("\nRunning evaluation...")
    metrics, predictions = evaluate_and_predict(
        model, tokenizer, df, args.device,
        max_src_len=args.max_src_len,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )

    # Print summary
    print("\n── Summary ──────────────────────────────────────")
    for direction, scores in metrics.items():
        if scores["bleu"] is None:
            print(f"  {direction}: skipped")
        else:
            print(f"  {direction}: BLEU={scores['bleu']:.2f}  chrF={scores['chrf']:.2f}  n={scores['n']}")

    # Macro-average over available directions
    valid = [s for s in metrics.values() if s["bleu"] is not None]
    if valid:
        avg_bleu = sum(s["bleu"] for s in valid) / len(valid)
        avg_chrf = sum(s["chrf"] for s in valid) / len(valid)
        print(f"\n  Macro-avg ({len(valid)} directions): BLEU={avg_bleu:.2f}  chrF={avg_chrf:.2f}")

    # Save metrics JSON
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"lora_path": args.lora_path, "data": args.data,
                   "xsa_enc": args.xsa_enc, "xsa_dec": args.xsa_dec,
                   "scores": metrics}, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved → {metrics_path}")

    # Save predictions JSONL
    pred_path = os.path.join(args.output_dir, "predictions.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Predictions saved → {pred_path}")


if __name__ == "__main__":
    main()
