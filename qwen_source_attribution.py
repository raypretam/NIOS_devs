"""
Qwen3-4B Source Attribution Analysis (decoder-only)
=====================================================
For a decoder-only model doing translation via a chat prompt, measures per layer:
  "How much attention do generated target tokens pay to source sentence positions?"

Approach: teacher-forced forward pass with the full (prompt + source + target)
sequence. For each layer, compute the fraction of attention from target positions
to source positions — averaged over all target positions, heads, and samples.

This directly answers: "Do later layers focus MORE or LESS on the source?"

Prompt structure (Qwen3 chat template):
    <|im_start|>user
    Translate the following {src_lang} text to {tgt_lang}:
    {src_text}<|im_end|>
    <|im_start|>assistant
    <think>

    </think>

    {tgt_text}<|im_end|>

Source positions = token positions of {src_text} inside user turn.
Target positions = token positions of {tgt_text} inside assistant turn.

Usage:
    python qwen_source_attribution.py \\
        --data ../NIOS_Trilingual_Joint/val_multilingual.jsonl \\
        --model Qwen/Qwen3-4B \\
        --n_samples 50 \\
        --output_dir ./attribution_output/qwen3
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

LANG_CODES = {
    "Hindi":    "hin_Deva",
    "Kannada":  "kan_Knda",
    "Sanskrit": "san_Deva",
}
LANG_NAMES = {v: k for k, v in LANG_CODES.items()}

DIRECTIONS = [
    ("Sanskrit", "Hindi"),
    ("Sanskrit", "Kannada"),
    ("Hindi",    "Sanskrit"),
    ("Hindi",    "Kannada"),
    ("Kannada",  "Sanskrit"),
    ("Kannada",  "Hindi"),
]


# ── Span detection ─────────────────────────────────────────────────────────

def find_subseq(seq, subseq):
    """Return start index of first occurrence of subseq in seq, or -1."""
    n, m = len(seq), len(subseq)
    for i in range(n - m + 1):
        if seq[i: i + m] == subseq:
            return i
    return -1


def build_and_locate(tokenizer, src_lang_name, tgt_lang_name, src_text, tgt_text):
    """
    Build the full token sequence using the chat template and return:
      full_ids        — list[int], complete input token ids
      src_positions   — list[int], positions of source text tokens
      tgt_positions   — list[int], positions of target text tokens
    Returns None if spans can't be located.
    """
    user_content = (
        f"Translate the following {src_lang_name} text to {tgt_lang_name}:\n"
        f"{src_text}"
    )
    messages = [
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": tgt_text},
    ]
    out = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=True
    )
    full_ids = out["input_ids"]

    # Find source span: tokenize the source text alone (no special tokens)
    src_ids = tokenizer.encode(src_text, add_special_tokens=False)
    src_start = find_subseq(full_ids, src_ids)
    if src_start == -1:
        return None
    src_positions = list(range(src_start, src_start + len(src_ids)))

    # Find target span: tokenize the target text alone
    tgt_ids = tokenizer.encode(tgt_text, add_special_tokens=False)
    tgt_start = find_subseq(full_ids, tgt_ids)
    if tgt_start == -1:
        return None
    tgt_positions = list(range(tgt_start, tgt_start + len(tgt_ids)))

    # Sanity: target must come after source
    if tgt_positions[0] <= src_positions[-1]:
        return None

    return full_ids, src_positions, tgt_positions


# ── Per-sample metrics ─────────────────────────────────────────────────────

def compute_sample_metrics(model, tokenizer, src_lang_name, tgt_lang_name,
                            src_text, tgt_text, device, max_len):
    """
    Returns dict: layer_index -> source_attribution_fraction
      where source_attribution_fraction = mean attention from target positions
      to source positions, averaged over heads.
    Returns None if span detection fails or sequence too long.
    """
    result = build_and_locate(tokenizer, src_lang_name, tgt_lang_name, src_text, tgt_text)
    if result is None:
        return None

    full_ids, src_positions, tgt_positions = result

    if len(full_ids) > max_len:
        return None

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    src_pos = torch.tensor(src_positions, dtype=torch.long)
    tgt_pos = torch.tensor(tgt_positions, dtype=torch.long)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_attentions=True,
            return_dict=True,
        )

    # outputs.attentions: tuple of (1, n_heads, T, T) per layer
    layer_metrics = {}
    for l, A in enumerate(outputs.attentions):
        A = A[0]  # (n_heads, T, T)
        # Attention from each target position to all positions
        # A[:, tgt_pos, :] — shape (n_heads, n_tgt, T)
        tgt_attn = A[:, tgt_pos, :]  # (H, n_tgt, T)
        # Fraction going to source positions
        src_attn_mass = tgt_attn[:, :, src_pos].sum(dim=-1)  # (H, n_tgt)
        layer_metrics[l] = src_attn_mass.mean().item()

    return layer_metrics


# ── Direction-level aggregation ────────────────────────────────────────────

def run_direction(model, tokenizer, df, src_name, tgt_name,
                  n_samples, device, max_len):
    src_code = LANG_CODES[src_name]
    tgt_code = LANG_CODES[tgt_name]
    mask = (df["src_lang"] == src_code) & (df["tgt_lang"] == tgt_code)
    rows = df[mask].reset_index(drop=True).iloc[:n_samples]

    if len(rows) == 0:
        return None

    key = f"{src_name[:3]}→{tgt_name[:3]}"
    running = {}

    for _, row in tqdm(rows.iterrows(), total=len(rows), desc=f"  {key}", leave=False):
        metrics = compute_sample_metrics(
            model, tokenizer,
            src_name, tgt_name,
            str(row["src_text"]), str(row["tgt_text"]),
            device, max_len,
        )
        if metrics is None:
            continue
        for l, v in metrics.items():
            running.setdefault(l, []).append(v)

    if not running:
        return None

    return {l: float(np.mean(vals)) for l, vals in running.items()}


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_results(results_by_direction, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)
    import seaborn as sns
    palette = sns.color_palette("tab10", len(results_by_direction))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Qwen3-4B — Source Sentence Attribution Across Layers\n"
        f"Fraction of attention from target token positions to source sentence positions",
        fontsize=12, fontweight="bold",
    )

    # Left: all directions overlaid
    ax = axes[0]
    for (direction, layer_dict), color in zip(results_by_direction.items(), palette):
        x = [l + 1 for l in sorted(layer_dict.keys())]
        y = [layer_dict[l] for l in sorted(layer_dict.keys())]
        ax.plot(x, y, marker="o", markersize=3, linewidth=1.8,
                label=direction, color=color)
    ax.set_title("Per direction", fontsize=11)
    ax.set_xlabel("Layer index", fontsize=10)
    ax.set_ylabel("Source attribution (fraction)", fontsize=10)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2)

    # Right: macro-average with std-dev band
    ax2 = axes[1]
    all_layers = sorted(next(iter(results_by_direction.values())).keys())
    mean_vals, std_vals = [], []
    for l in all_layers:
        vals = [d[l] for d in results_by_direction.values() if l in d]
        mean_vals.append(np.mean(vals))
        std_vals.append(np.std(vals))

    x = [l + 1 for l in all_layers]
    mean_vals = np.array(mean_vals)
    std_vals = np.array(std_vals)
    ax2.plot(x, mean_vals, color="steelblue", linewidth=2, marker="o", markersize=4, label="Macro avg")
    ax2.fill_between(x, mean_vals - std_vals, mean_vals + std_vals,
                     alpha=0.25, color="steelblue", label="±1 std dev")

    # Annotate peak layer
    peak_idx = int(np.argmax(mean_vals))
    ax2.annotate(
        f"Peak: layer {x[peak_idx]}\n({mean_vals[peak_idx]:.3f})",
        xy=(x[peak_idx], mean_vals[peak_idx]),
        xytext=(x[peak_idx] + 1, mean_vals[peak_idx] + 0.01),
        fontsize=8, arrowprops=dict(arrowstyle="->", lw=1),
    )

    ax2.set_title("Macro-average across all directions", fontsize=11)
    ax2.set_xlabel("Layer index", fontsize=10)
    ax2.set_ylabel("Source attribution (fraction)", fontsize=10)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "qwen3_source_attribution.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Plot → {fig_path}")
    plt.close()

    # Save CSV
    rows = []
    for direction, layer_dict in results_by_direction.items():
        for l, v in layer_dict.items():
            rows.append({"direction": direction, "layer": l + 1, "src_attribution": round(v, 5)})
    csv_path = os.path.join(output_dir, "qwen3_source_attribution.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  CSV  → {csv_path}")

    # Print summary
    print(f"\n  Peak source attribution: layer {x[peak_idx]} "
          f"(mean={mean_vals[peak_idx]:.4f})")
    print(f"  Early layers (1-8) avg : {mean_vals[:8].mean():.4f}")
    mid = len(x) // 3
    print(f"  Mid layers avg         : {mean_vals[mid:2*mid].mean():.4f}")
    print(f"  Late layers avg        : {mean_vals[-8:].mean():.4f}")


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       required=True, help="JSONL eval file")
    p.add_argument("--model",      default="Qwen/Qwen3-4B")
    p.add_argument("--n_samples",  type=int, default=50,
                   help="Samples per direction (keep low for memory/speed)")
    p.add_argument("--max_len",    type=int, default=512,
                   help="Skip samples longer than this many tokens")
    p.add_argument("--output_dir", default="./attribution_output/qwen3")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.model}...")
    print("  (using attn_implementation='eager' — required for output_attentions)")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",   # flash_attn doesn't support output_attentions
        trust_remote_code=True,
    ).eval().to(args.device)

    n_layers = model.config.num_hidden_layers
    print(f"Layers: {n_layers}   Device: {args.device}")

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

        n_use = min(n_avail, args.n_samples)
        print(f"\n[{src_name[:3]}→{tgt_name[:3]}]  {n_use} samples...")
        layer_dict = run_direction(
            model, tokenizer, df, src_name, tgt_name,
            args.n_samples, args.device, args.max_len,
        )
        if layer_dict:
            key = f"{src_name[:3]}→{tgt_name[:3]}"
            results_by_direction[key] = layer_dict
            avg = np.mean(list(layer_dict.values()))
            peak_l = max(layer_dict, key=layer_dict.get)
            print(f"    Avg source attribution: {avg:.4f}  Peak at layer {peak_l + 1}: {layer_dict[peak_l]:.4f}")

    if not results_by_direction:
        print("No results to plot.")
        return

    print("\nPlotting...")
    plot_results(results_by_direction, args.output_dir, args.model)
    print("Done.")


if __name__ == "__main__":
    main()
