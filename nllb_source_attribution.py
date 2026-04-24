"""
NLLB-600M Source Attribution Analysis
======================================
For each encoder/decoder layer, measures how much attention goes to source
*content* tokens vs special/padding tokens.

Three panels per plot:
  1. Encoder self-attn  — fraction going to source content keys (per layer)
  2. Decoder cross-attn — fraction going to source content positions (per layer)
                          ← answers "when does the decoder look at the source?"
  3. Decoder self-attn  — attention entropy per layer (how spread vs focused)

Usage:
    python nllb_source_attribution.py \\
        --data ../NIOS_Trilingual_Joint/val_multilingual.jsonl \\
        --model facebook/nllb-200-distilled-600M \\
        --n_samples 100 \\
        --output_dir ./attribution_output/nllb
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

warnings.filterwarnings("ignore")

LANG_CODES = {
    "Hindi":    "hin_Deva",
    "Kannada":  "kan_Knda",
    "Sanskrit": "san_Deva",
}
DIRECTIONS = [
    ("Sanskrit", "Hindi"),
    ("Sanskrit", "Kannada"),
    ("Hindi",    "Sanskrit"),
    ("Hindi",    "Kannada"),
    ("Kannada",  "Sanskrit"),
    ("Kannada",  "Hindi"),
]


# ── Span masks ────────────────────────────────────────────────────────────

def source_content_mask(attention_mask_1d):
    """
    Returns (content_mask, special_mask) as boolean tensors.
    NLLB encoder format: [lang_token, w1..wN, EOS, PAD...]
    Content = positions 1..N  (excluding lang_token at 0 and EOS at real_len-1)
    Special = positions 0 and real_len-1
    """
    T = attention_mask_1d.shape[0]
    real_len = int(attention_mask_1d.sum().item())
    content = torch.zeros(T, dtype=torch.bool)
    special = torch.zeros(T, dtype=torch.bool)
    if real_len >= 3:
        content[1:real_len - 1] = True
    special[0] = True
    if real_len >= 2:
        special[real_len - 1] = True
    return content, special


# ── Per-sample metrics ─────────────────────────────────────────────────────

def attn_entropy(A):
    """Mean entropy of attention rows. A: (H, T_q, T_k) already softmaxed."""
    eps = 1e-9
    # entropy per head per query position: -sum(p log p)
    ent = -(A * (A + eps).log()).sum(dim=-1)   # (H, T_q)
    return ent.mean().item()


def compute_sample_metrics(model, tokenizer, src_text, tgt_text,
                            src_lang, tgt_lang, device, max_len):
    """
    Teacher-forced forward pass; returns dicts keyed by layer index:
      enc_content_frac[l]  — fraction of enc self-attn mass on content keys
      dec_cross_frac[l] — fraction of dec cross-attn mass on source content
      dec_cross_special[l] — fraction of dec cross-attn mass on source special tokens
      dec_self_ent[l]      — entropy of dec self-attn

    Key metric definition:
      "Fraction on content" = sum of attention weights over content key positions,
      averaged over heads and (real) query positions.
      This gives a true [0,1] fraction of attention mass, not a per-token mean.
    """
    tokenizer.src_lang = src_lang
    src_enc = tokenizer(
        src_text, return_tensors="pt",
        max_length=max_len, truncation=True, padding="max_length",
    )
    input_ids = src_enc["input_ids"].to(device)
    attention_mask = src_enc["attention_mask"].to(device)

    content_mask, special_mask = source_content_mask(attention_mask[0].cpu())
    content_mask = content_mask.to(device)
    special_mask = special_mask.to(device)
    # Real (non-padding) query positions for encoder
    real_query_mask = attention_mask[0].bool()

    tokenizer.tgt_lang = tgt_lang
    tgt_enc = tokenizer(
        text_target=tgt_text, return_tensors="pt",
        max_length=max_len, truncation=True, padding=False,
    )
    labels = tgt_enc["input_ids"].to(device)
    bos_id = model.config.decoder_start_token_id
    decoder_input_ids = torch.cat(
        [torch.tensor([[bos_id]], device=device), labels[:, :-1]], dim=1
    )

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            output_attentions=True,
            return_dict=True,
        )

    enc_content_frac, dec_cross_frac, dec_cross_special, dec_self_ent = {}, {}, {}, {}

    # ── Encoder self-attention ──────────────────────────────────────────
    # Shape: (1, H, T_src, T_src) — attention from each source position to all source positions
    # Fix: restrict to real query positions, then SUM over content/special key positions
    # so the result is the fraction of attention mass going to content keys.
    for l, A in enumerate(outputs.encoder_attentions):
        A = A[0]                                # (H, T_src, T_src)
        A_real = A[:, real_query_mask, :]       # (H, real_len, T_src) — drop padded queries
        # Sum over content key positions → (H, real_len), then mean over heads & queries
        content_frac = A_real[:, :, content_mask].sum(dim=-1).mean().item()
        enc_content_frac[l] = content_frac

    # ── Decoder cross-attention ─────────────────────────────────────────
    # Shape: (1, H, T_tgt, T_src)
    # Decoder attends to encoder output; padding positions have 0 weight (masked).
    # SUM over content/special source positions → fraction of cross-attn mass.
    for l, A in enumerate(outputs.cross_attentions):
        A = A[0]                                                        # (H, T_tgt, T_src)
        content_frac = A[:, :, content_mask].sum(dim=-1).mean().item()  # sum content keys
        special_frac = A[:, :, special_mask].sum(dim=-1).mean().item()  # sum special keys
        dec_cross_frac[l] = content_frac
        dec_cross_special[l] = special_frac

    # ── Decoder self-attention ──────────────────────────────────────────
    for l, A in enumerate(outputs.decoder_attentions):
        A = A[0]  # (H, T_tgt, T_tgt)
        dec_self_ent[l] = attn_entropy(A)

    return enc_content_frac, dec_cross_frac, dec_cross_special, dec_self_ent


# ── Direction-level aggregation ────────────────────────────────────────────

def accumulate(running, sample_dict):
    for k, v in sample_dict.items():
        running.setdefault(k, []).append(v)


def mean_over_layers(running):
    return {k: float(np.mean(v)) for k, v in running.items()}


def run_direction(model, tokenizer, df, src_name, tgt_name, n_samples, device, max_len):
    src_code = LANG_CODES[src_name]
    tgt_code = LANG_CODES[tgt_name]
    mask = (df["src_lang"] == src_code) & (df["tgt_lang"] == tgt_code)
    rows = df[mask].reset_index(drop=True)
    if len(rows) == 0:
        return None
    rows = rows.iloc[:n_samples]

    run_enc, run_cross_content, run_cross_special, run_dec = {}, {}, {}, {}
    key = f"{src_name[:3]}→{tgt_name[:3]}"

    for _, row in tqdm(rows.iterrows(), total=len(rows), desc=f"  {key}", leave=False):
        try:
            ef, ccf, csf, df_ = compute_sample_metrics(
                model, tokenizer,
                row["src_text"], row["tgt_text"],
                src_code, tgt_code, device, max_len,
            )
        except Exception:
            continue
        accumulate(run_enc, ef)
        accumulate(run_cross_content, ccf)
        accumulate(run_cross_special, csf)
        accumulate(run_dec, df_)

    return {
        "enc_content_frac":   mean_over_layers(run_enc),
        "dec_cross_frac":  mean_over_layers(run_cross_content),
        "dec_cross_special":  mean_over_layers(run_cross_special),
        "dec_self_ent":       mean_over_layers(run_dec),
        "n": len(rows),
    }


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_results(results_by_direction, output_dir, n_enc, n_dec):
    os.makedirs(output_dir, exist_ok=True)
    import seaborn as sns
    palette = sns.color_palette("tab10", len(results_by_direction))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        "NLLB-600M — Source Content Attribution Across Layers",
        fontsize=13, fontweight="bold",
    )

    panels = [
        ("enc_content_frac", axes[0],
         "Encoder self-attention\nFraction of attn to source content keys",
         range(1, n_enc + 1), "Encoder layer"),
        ("dec_cross_frac", axes[1],
         "Decoder cross-attention\nFraction of attn to source content positions",
         range(1, n_dec + 1), "Decoder layer"),
        ("dec_self_ent", axes[2],
         "Decoder self-attention\nMean entropy (nats) of attn distribution",
         range(1, n_dec + 1), "Decoder layer"),
    ]

    for metric_key, ax, title, x_range, xlabel in panels:
        for (direction, data), color in zip(results_by_direction.items(), palette):
            if metric_key not in data:
                continue
            y_dict = data[metric_key]
            n_layers = len(y_dict)
            x = list(range(1, n_layers + 1))
            y = [y_dict[i] for i in sorted(y_dict.keys())]
            ax.plot(x, y, marker="o", markersize=4, linewidth=1.8,
                    label=direction, color=color)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    axes[0].set_ylabel("Fraction of attention mass", fontsize=9)
    axes[1].set_ylabel("Fraction of attention mass", fontsize=9)
    axes[2].set_ylabel("Mean entropy (nats)", fontsize=9)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "nllb_source_attribution.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Plot → {fig_path}")
    plt.close()

    # Save CSV
    rows = []
    for direction, data in results_by_direction.items():
        for metric_key in ("enc_content_frac", "dec_cross_frac", "dec_self_ent"):
            component = metric_key.split("_")[0] + "_" + metric_key.split("_")[1]
            for layer, val in data.get(metric_key, {}).items():
                rows.append({"direction": direction, "metric": metric_key,
                             "layer": layer + 1, "value": round(val, 4)})
    csv_path = os.path.join(output_dir, "nllb_source_attribution.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  CSV  → {csv_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       required=True, help="JSONL eval file")
    p.add_argument("--model",      default="facebook/nllb-200-distilled-600M")
    p.add_argument("--n_samples",  type=int, default=100,
                   help="Samples per direction to process")
    p.add_argument("--max_len",    type=int, default=128)
    p.add_argument("--output_dir", default="./attribution_output/nllb")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.model}...")
    print("  (using attn_implementation='eager' — required for output_attentions)")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    ).eval().to(args.device)

    n_enc = len(model.model.encoder.layers)
    n_dec = len(model.model.decoder.layers)
    print(f"Encoder layers: {n_enc}   Decoder layers: {n_dec}")

    df = pd.read_json(args.data, lines=True)
    required = {"src_lang", "tgt_lang", "src_text", "tgt_text"}
    df = df[list(required)].dropna().reset_index(drop=True)
    print(f"Loaded {len(df)} samples from {args.data}")

    results_by_direction = {}
    for src_name, tgt_name in DIRECTIONS:
        src_code = LANG_CODES[src_name]
        tgt_code = LANG_CODES[tgt_name]
        n_avail = ((df["src_lang"] == src_code) & (df["tgt_lang"] == tgt_code)).sum()
        if n_avail == 0:
            print(f"  [{src_name[:3]}→{tgt_name[:3]}] skipped (0 samples)")
            continue
        print(f"\n[{src_name[:3]}→{tgt_name[:3]}]  {min(n_avail, args.n_samples)} samples...")
        result = run_direction(
            model, tokenizer, df, src_name, tgt_name,
            args.n_samples, args.device, args.max_len,
        )
        if result:
            key = f"{src_name[:3]}→{tgt_name[:3]}"
            results_by_direction[key] = result
            ec = np.mean(list(result["enc_content_frac"].values()))
            dc = np.mean(list(result["dec_cross_frac"].values()))
            print(f"    Enc content attn (avg): {ec:.3f}")
            print(f"    Dec source attn  (avg): {dc:.3f}")

    print("\nPlotting...")
    plot_results(results_by_direction, args.output_dir, n_enc, n_dec)
    print("Done.")


if __name__ == "__main__":
    main()
