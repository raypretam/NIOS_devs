"""
Script 1 (v3): XSA Diagnostic — correct for NLLB-200 / M2M100Attention
=======================================================================
Fixes two bugs in v2:
  1. NLLB uses M2M100Attention (in modeling_m2m_100.py), not MBartAttention.
  2. The encoder layer caller unpacks exactly 2 values from self_attn():
         hidden_states, _ = self.self_attn(...)
     so our patched forward must also return a 2-tuple.
     The decoder self_attn call in m2m_100 also returns 2-tuple for the
     self-attention path (past_key_value handled separately in the layer).

Strategy: instead of monkey-patching forward(), we register a hook on
v_proj to capture V, and a separate hook on out_proj to capture the input
to out_proj (which is exactly concat(Y_heads) before projection).
This avoids touching the forward() signature entirely.

Measurement point:
    V   = v_proj(hidden_states)          → (B, T, D), reshape to (B,H,T,Dh)
    Y   = input_to_out_proj              → (B, T, D), reshape to (B,H,T,Dh)
    cos(Y_i, V_i) per head per token, then averaged.

This is the correct pre-out_proj measurement from the XSA paper.

Usage:
    python xsa_diagnostic_v3.py \
        --csv nios_test.csv \
        --model facebook/nllb-200-distilled-600M \
        --n_samples 100 \
        --output_dir ./xsa_diagnostic_v3_output
"""

import argparse
import os
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

warnings.filterwarnings("ignore")

LANG_CODES = {
    "Hindi":    "hin_Deva",
    "Kannada":  "kan_Knda",
    "Sanskrit": "san_Deva",
}


# ── Hook-based measurement ─────────────────────────────────────────────────
#
# For each M2M100Attention (self-attn only):
#   Hook A on v_proj  : captures output V = (B, T, D)
#   Hook B on out_proj: captures input  Y = (B, T, D)  [pre-proj, post-bmm]
#
# Both are stored keyed by a shared buffer dict, then consumed after each
# forward pass to compute cos(Y_i, V_i).

class AttentionBiasProbe:
    def __init__(self, model):
        self.model   = model
        self.hooks   = []
        self.results = {}   # key -> list of float

        n_enc = len(model.model.encoder.layers)
        n_dec = len(model.model.decoder.layers)

        for idx in range(n_enc):
            self._register_pair(
                model.model.encoder.layers[idx].self_attn,
                f"enc_{idx}",
            )
        for idx in range(n_dec):
            self._register_pair(
                model.model.decoder.layers[idx].self_attn,
                f"dec_{idx}",
            )

    def _register_pair(self, attn_module, key):
        """Register v_proj and out_proj hooks that share a buffer."""
        buf = {}   # shared between the two hooks for this layer

        def hook_v(module, inp, out):
            # out: (B, T, D)
            buf["V"] = out.detach()

        def hook_out(module, inp, out):
            # inp[0]: (B, T, D)  — input to out_proj = concat of all heads
            Y = inp[0].detach()
            V = buf.get("V")
            if V is None or Y.shape != V.shape:
                return
            with torch.no_grad():
                num_heads = attn_module.num_heads
                head_dim  = attn_module.head_dim
                B, T, D   = Y.shape
                # reshape to (B, H, T, Dh)
                Yr = Y.view(B, T, num_heads, head_dim).permute(0, 2, 1, 3)
                Vr = V.view(B, T, num_heads, head_dim).permute(0, 2, 1, 3)
                # per-token, per-head cosine similarity
                Yn = F.normalize(Yr, dim=-1)
                Vn = F.normalize(Vr, dim=-1)
                cos = (Yn * Vn).sum(dim=-1).mean().item()
            if key not in self.results:
                self.results[key] = []
            self.results[key].append(cos)
            buf.clear()

        h1 = attn_module.v_proj.register_forward_hook(hook_v)
        h2 = attn_module.out_proj.register_forward_hook(hook_out)
        self.hooks.extend([h1, h2])

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_means(self, n_enc, n_dec):
        enc = np.array([np.mean(self.results.get(f"enc_{i}", [0.0])) for i in range(n_enc)])
        dec = np.array([np.mean(self.results.get(f"dec_{i}", [0.0])) for i in range(n_dec)])
        return enc, dec


# ── Inference ─────────────────────────────────────────────────────────────

def run_sentences(model, tokenizer, sentences, src_lang, tgt_lang, device, max_len):
    tokenizer.src_lang = src_lang
    tgt_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    model.eval()
    for sent in tqdm(sentences, desc=f"  {src_lang}", leave=False):
        enc = tokenizer(
            sent, return_tensors="pt",
            max_length=max_len, truncation=True, padding=False,
        ).to(device)
        with torch.no_grad():
            model.generate(**enc, forced_bos_token_id=tgt_id, max_new_tokens=5)


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_and_save(enc_by_lang, dec_by_lang, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    langs   = list(enc_by_lang.keys())
    palette = sns.color_palette("tab10", len(langs))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Attention Similarity Bias: cos(yᵢ, vᵢ) per layer — NLLB-200\n"
        "(measured at out_proj input, per-head space — correct XSA replication)",
        fontsize=11, fontweight="bold",
    )
    for ax, (data, title) in zip(
        axes,
        [(enc_by_lang, "Encoder self-attention"),
         (dec_by_lang, "Decoder masked self-attention")],
    ):
        for lang, color in zip(langs, palette):
            y = data[lang]
            x = np.arange(1, len(y) + 1)
            ax.plot(x, y, marker="o", markersize=4, linewidth=1.8,
                    label=lang, color=color)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Layer index", fontsize=10)
        ax.set_ylabel("Mean cos(yᵢ, vᵢ)", fontsize=10)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.legend(fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "attention_similarity_bias_v3.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Plot → {fig_path}")
    plt.show()

    rows = []
    for lang in langs:
        for i, v in enumerate(enc_by_lang[lang]):
            rows.append({"lang": lang, "component": "encoder", "layer": i+1, "mean_cos": round(v, 4)})
        for i, v in enumerate(dec_by_lang[lang]):
            rows.append({"lang": lang, "component": "decoder", "layer": i+1, "mean_cos": round(v, 4)})
    csv_path = os.path.join(output_dir, "attention_similarity_bias_v3.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  CSV  → {csv_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",         default="nios_test.csv")
    p.add_argument("--model",       default="facebook/nllb-200-distilled-600M")
    p.add_argument("--n_samples",   type=int, default=100)
    p.add_argument("--output_dir",  default="./xsa_diagnostic_v3_output")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_src_len", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Loading {args.model} on {args.device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval().to(args.device)

    n_enc = len(model.model.encoder.layers)
    n_dec = len(model.model.decoder.layers)
    print(f"Encoder layers: {n_enc}   Decoder layers: {n_dec}")

    # Verify we found M2M100Attention
    attn_cls = type(model.model.encoder.layers[0].self_attn).__name__
    print(f"Attention class: {attn_cls}")
    assert hasattr(model.model.encoder.layers[0].self_attn, "v_proj"), \
        "Expected v_proj attribute — wrong attention class?"
    assert hasattr(model.model.encoder.layers[0].self_attn, "out_proj"), \
        "Expected out_proj attribute — wrong attention class?"

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")

    enc_by_lang, dec_by_lang = {}, {}

    for src_col, src_lang in LANG_CODES.items():
        tgt_lang = "hin_Deva" if src_col != "Hindi" else "kan_Knda"
        sentences = (
            df[src_col].dropna()
            .sample(min(args.n_samples, len(df)), random_state=42)
            .tolist()
        )
        print(f"\n[{src_col}]  {len(sentences)} sentences → probing...")

        probe = AttentionBiasProbe(model)
        run_sentences(model, tokenizer, sentences, src_lang, tgt_lang,
                      args.device, args.max_src_len)
        enc_means, dec_means = probe.get_means(n_enc, n_dec)
        probe.remove()

        enc_by_lang[src_col] = enc_means
        dec_by_lang[src_col] = dec_means

        print(f"  Encoder cos(y,v): "
              f"min={enc_means.min():.3f}  max={enc_means.max():.3f}  mean={enc_means.mean():.3f}")
        print(f"  Decoder cos(y,v): "
              f"min={dec_means.min():.3f}  max={dec_means.max():.3f}  mean={dec_means.mean():.3f}")

    print("\nPlotting results...")
    plot_and_save(enc_by_lang, dec_by_lang, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()