"""
Qwen3 XSA Diagnostic — measures attention similarity bias cos(y_i, v_i) per layer
==================================================================================
Adapted from the NLLB-200 diagnostic for decoder-only Qwen3 models.

Key differences vs the NLLB version:
  1. Qwen3 is a decoder-only causal LM (one stack of self-attention layers,
     no encoder).  Module path: model.model.layers[i].self_attn
  2. Module names follow LLaMA-style: q_proj / k_proj / v_proj / o_proj
     (not out_proj as in M2M100/MBart).
  3. Qwen3 uses Grouped-Query Attention (GQA).  For Qwen3-4B:
         num_attention_heads = 32   (Q heads)
         num_key_value_heads = 8    (KV heads)
     So V has shape (B, T, 8*Dh) while the input to o_proj has shape
     (B, T, 32*Dh).  We must repeat each KV head `num_attention_heads /
     num_key_value_heads = 4` times to align them with Q heads before
     computing cos(y_i, v_i) per head — this is the standard `repeat_kv`
     operation used inside the model itself.
  4. Qwen3 has Q/K-Norm (RMSNorm on queries and keys) but this is applied
     INSIDE the attention computation between proj and bmm.  It does not
     affect our measurement: v_proj output is what gets averaged by
     attention, and o_proj input is exactly that average — so cos(y, v)
     measures self-position bias the same way as in the XSA paper.

Strategy: register hooks on v_proj (output → V) and o_proj (input → Y),
exactly mirroring the NLLB script.  No forward() patching needed.

Usage:
    # Wide CSV with one column per group (e.g. nios_test.csv with
    # Hindi/Kannada/Sanskrit columns) — one curve per language:
    python qwen3_xsa_diagnostic.py \
        --data nios_test.csv \
        --text_columns Hindi Kannada Sanskrit \
        --model Qwen/Qwen3-4B \
        --n_samples 100 \
        --output_dir ./qwen3_xsa_output

    # Same wide CSV but pooled into a single curve:
    python qwen3_xsa_diagnostic.py \
        --data nios_test.csv \
        --text_columns Hindi Kannada Sanskrit --pool \
        --model Qwen/Qwen3-4B

    # Long-format CSV with a 'text' column and optional 'group' column:
    python qwen3_xsa_diagnostic.py \
        --data prompts.csv --text_column text --group_column domain \
        --model Qwen/Qwen3-4B

    # Plain .txt — one sentence per line:
    python qwen3_xsa_diagnostic.py \
        --data prompts.txt --model Qwen/Qwen3-4B
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")


# ── Hook-based measurement ─────────────────────────────────────────────────
#
# For each Qwen3Attention (decoder self-attn — the only kind here):
#   Hook A on v_proj : captures output V = (B, T, n_kv_heads * head_dim)
#   Hook B on o_proj : captures input  Y = (B, T, n_q_heads  * head_dim)
#
# Because of GQA, V is "narrower" than Y: each KV head serves
# n_q_heads/n_kv_heads query heads.  We replicate V along the head axis
# to align with Y, then compute cos(Y_i, V_i) per head per token.

class Qwen3AttentionBiasProbe:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.results = {}  # key (layer index str) -> list of float

        # Qwen3ForCausalLM:  model -> Qwen3Model -> .layers[i].self_attn
        self.layers = model.model.layers
        self.n_layers = len(self.layers)

        for idx in range(self.n_layers):
            self._register_pair(self.layers[idx].self_attn, f"layer_{idx}")

    def _register_pair(self, attn_module, key):
        """Register v_proj and o_proj hooks that share a buffer."""
        buf = {}  # shared between the two hooks for this layer

        # Read GQA shape info once (set on the module by Qwen3Attention.__init__)
        head_dim = attn_module.head_dim
        # num_attention_heads is on config; num_key_value_heads is also on config.
        # The module exposes num_key_value_groups = n_q_heads / n_kv_heads.
        n_kv_groups = attn_module.num_key_value_groups
        # Total q heads = (kv heads) * (groups).  We don't know kv heads
        # directly from the module, but we can infer from V's shape at runtime.

        def hook_v(module, inp, out):
            # out: (B, T, n_kv_heads * head_dim)
            buf["V"] = out.detach()

        def hook_o(module, inp, out):
            # inp[0]: (B, T, n_q_heads * head_dim) — input to o_proj
            Y = inp[0].detach()
            V = buf.get("V")
            if V is None:
                return
            with torch.no_grad():
                B, T, D_q = Y.shape
                _, _, D_kv = V.shape
                # Number of KV heads inferred from V; number of Q heads from Y
                n_kv_heads = D_kv // head_dim
                n_q_heads = D_q // head_dim

                # Reshape:
                #   Y → (B, n_q_heads, T, head_dim)
                #   V → (B, n_kv_heads, T, head_dim)
                Yr = Y.view(B, T, n_q_heads, head_dim).permute(0, 2, 1, 3)
                Vr = V.view(B, T, n_kv_heads, head_dim).permute(0, 2, 1, 3)

                # Repeat KV heads to align with Q heads (GQA expansion).
                # Each KV head serves `n_kv_groups = n_q_heads / n_kv_heads`
                # query heads consecutively.
                if n_q_heads != n_kv_heads:
                    Vr = Vr.repeat_interleave(n_kv_groups, dim=1)
                    # Now Vr: (B, n_q_heads, T, head_dim)

                # Per-token, per-head cosine similarity, then mean
                Yn = F.normalize(Yr, dim=-1)
                Vn = F.normalize(Vr, dim=-1)
                cos = (Yn * Vn).sum(dim=-1).mean().item()

            self.results.setdefault(key, []).append(cos)
            buf.clear()

        h1 = attn_module.v_proj.register_forward_hook(hook_v)
        h2 = attn_module.o_proj.register_forward_hook(hook_o)
        self.hooks.extend([h1, h2])

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_means(self):
        return np.array([
            np.mean(self.results.get(f"layer_{i}", [0.0]))
            for i in range(self.n_layers)
        ])


# ── Inference ─────────────────────────────────────────────────────────────

def run_sentences(model, tokenizer, sentences, device, max_len):
    """Forward each sentence once.  We just need a single forward pass per
    input — no need to actually generate beyond a few tokens."""
    model.eval()
    for sent in tqdm(sentences, desc="  probing", leave=False):
        enc = tokenizer(
            sent,
            return_tensors="pt",
            max_length=max_len,
            truncation=True,
            padding=False,
        ).to(device)
        with torch.no_grad():
            # A single forward pass triggers all hooks once per layer.
            # We don't need .generate() — that would fire hooks many times
            # (once per generated token) and inflate the per-sentence cost.
            model(**enc)


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_and_save(data_by_group, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)
    groups = list(data_by_group.keys())
    palette = sns.color_palette("tab10", max(len(groups), 3))

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        f"Attention Similarity Bias: cos(yᵢ, vᵢ) per layer — {model_name}\n"
        "(decoder self-attention, measured at o_proj input, per-head GQA-aligned)",
        fontsize=11, fontweight="bold",
    )
    for group, color in zip(groups, palette):
        y = data_by_group[group]
        x = np.arange(1, len(y) + 1)
        ax.plot(x, y, marker="o", markersize=3.5, linewidth=1.6,
                label=group, color=color)

    ax.set_xlabel("Layer index", fontsize=10)
    ax.set_ylabel("Mean cos(yᵢ, vᵢ)", fontsize=10)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=12))
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "qwen3_attention_similarity_bias.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Plot → {fig_path}")
    plt.close()

    rows = []
    for group in groups:
        for i, v in enumerate(data_by_group[group]):
            rows.append({
                "group": group,
                "component": "decoder",
                "layer": i + 1,
                "mean_cos": round(float(v), 4),
            })
    csv_path = os.path.join(output_dir, "qwen3_attention_similarity_bias.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  CSV  → {csv_path}")


# ── Data loading ───────────────────────────────────────────────────────────

def load_sentences(args):
    """
    Returns a dict: group_name -> list[str]
    Supports three CSV/TXT shapes:

      1. .txt — one sentence per line, single group called 'all'

      2. Wide CSV with one column per group (e.g. Hindi/Kannada/Sanskrit).
         Pass --text_columns Hindi Kannada Sanskrit — each becomes its own
         curve.  This matches the NLLB diagnostic.  If --pool is set, all
         columns are merged into a single 'all' group.

      3. Long CSV with one text column and one group column.  Pass
         --text_column text --group_column lang.

      4. Long CSV with just a text column — single curve called 'all'.
         Pass --text_column text.
    """
    ext = os.path.splitext(args.data)[1].lower()

    if ext == ".txt":
        with open(args.data, encoding="utf-8") as f:
            sents = [line.strip() for line in f if line.strip()]
        return {"all": sents}

    if ext != ".csv":
        raise ValueError(f"Unsupported file extension '{ext}'. Use .csv or .txt")

    df = pd.read_csv(args.data)

    # Wide format: --text_columns col1 col2 ...
    if args.text_columns:
        missing = [c for c in args.text_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"--text_columns has columns not in CSV: {missing}. "
                f"Available: {list(df.columns)}"
            )
        if args.pool:
            sents = []
            for c in args.text_columns:
                sents.extend(df[c].dropna().astype(str).tolist())
            return {"all": sents}
        return {
            c: df[c].dropna().astype(str).tolist()
            for c in args.text_columns
        }

    # Long format
    if args.text_column not in df.columns:
        raise ValueError(
            f"--text_column '{args.text_column}' not in CSV columns: "
            f"{list(df.columns)}.  If your CSV is wide (one column per group), "
            f"use --text_columns instead, e.g. --text_columns Hindi Kannada Sanskrit"
        )

    if args.group_column is None:
        return {"all": df[args.text_column].dropna().astype(str).tolist()}

    if args.group_column not in df.columns:
        raise ValueError(
            f"--group_column '{args.group_column}' not in CSV columns: "
            f"{list(df.columns)}"
        )

    out = {}
    for group, sub in df.groupby(args.group_column):
        out[str(group)] = sub[args.text_column].dropna().astype(str).tolist()
    return out


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True,
                   help="Path to .csv (use --text_column or --text_columns) "
                        "or .txt (one sentence per line)")
    p.add_argument("--text_columns", nargs="+", default=None,
                   help="Wide-format CSV: one column per group, each plotted "
                        "as a separate curve. E.g. --text_columns Hindi Kannada Sanskrit")
    p.add_argument("--pool", action="store_true",
                   help="With --text_columns, merge all columns into a single "
                        "'all' group instead of plotting them separately.")
    p.add_argument("--text_column", default="text",
                   help="Long-format CSV: column with the sentences. "
                        "Ignored if --text_columns is set.")
    p.add_argument("--group_column", default=None,
                   help="Long-format CSV: column to group sentences by. "
                        "Each group is plotted as a separate curve.")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--n_samples", type=int, default=100,
                   help="Sentences sampled per group")
    p.add_argument("--output_dir", default="./qwen3_xsa_output")
    p.add_argument("--max_src_len", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"],
                   help="Compute dtype. 'auto' = bf16 on CUDA, fp32 on CPU.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def resolve_dtype(arg, device):
    if arg == "fp32":
        return torch.float32
    if arg == "fp16":
        return torch.float16
    if arg == "bf16":
        return torch.bfloat16
    # auto
    return torch.bfloat16 if device.startswith("cuda") else torch.float32


def main():
    args = parse_args()
    dtype = resolve_dtype(args.dtype, args.device)

    print(f"Loading {args.model} on {args.device} ({dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval().to(args.device)

    n_layers = len(model.model.layers)
    attn0 = model.model.layers[0].self_attn
    print(f"Decoder layers      : {n_layers}")
    print(f"Attention class     : {type(attn0).__name__}")
    print(f"head_dim            : {attn0.head_dim}")
    print(f"num_key_value_groups: {attn0.num_key_value_groups}  "
          f"(GQA: each KV head serves this many Q heads)")

    # Sanity checks for the projection names we hook
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert hasattr(attn0, name), \
            f"Expected attribute '{name}' on Qwen3Attention — got: " \
            f"{[n for n, _ in attn0.named_children()]}"

    print(f"\nLoading sentences from {args.data}")
    sentences_by_group = load_sentences(args)
    print(f"Found groups: {list(sentences_by_group.keys())}")

    # Sample n_samples per group, deterministically
    rng = np.random.default_rng(args.seed)
    for g, sents in sentences_by_group.items():
        if len(sents) > args.n_samples:
            idx = rng.choice(len(sents), args.n_samples, replace=False)
            sentences_by_group[g] = [sents[i] for i in idx]
        print(f"  {g}: {len(sentences_by_group[g])} sentences")

    data_by_group = {}
    for group, sents in sentences_by_group.items():
        print(f"\n[{group}]  probing {len(sents)} sentences...")
        probe = Qwen3AttentionBiasProbe(model)
        run_sentences(model, tokenizer, sents, args.device, args.max_src_len)
        means = probe.get_means()
        probe.remove()
        data_by_group[group] = means

        print(f"  cos(y,v) per layer: "
              f"min={means.min():.3f}  max={means.max():.3f}  mean={means.mean():.3f}")

    print("\nPlotting results...")
    plot_and_save(data_by_group, args.output_dir, args.model)
    print("\nDone.")


if __name__ == "__main__":
    main()