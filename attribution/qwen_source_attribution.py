"""
Qwen3-4B Full Attention Attribution Analysis (decoder-only)
=============================================================
Measures, per layer and per target position, how attention mass from generated
target tokens is distributed across every meaningful span in the input:

    pre_instr   — <|im_start|>user\\n  (chat scaffolding before the instruction)
    instruction — "Translate the following X to Y:\\n"
    src         — the source sentence itself
    post_src    — <|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n
    tgt_self    — attention from target tokens to earlier target tokens

These five fractions sum to ~1.0 per layer per target token (sanity-checked).

Produces four analysis panels:
  1. Stacked area — span composition across layers (macro average)
  2. Multi-line — each span as its own curve (shows peaks & collapses)
  3. Per-direction source attribution (the original metric)
  4. First-token vs continuation — how attention shifts across target position

Usage:
    python qwen_source_attribution_full.py \\
        --data ../NIOS_Trilingual_Joint/val_multilingual.jsonl \\
        --model Qwen/Qwen3-4B \\
        --n_samples 50 \\
        --output_dir ./attribution_output/qwen3_full
"""

import argparse
import json
import os
import warnings
from collections import defaultdict

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
DIRECTIONS = [
    ("Sanskrit", "Hindi"),
    ("Sanskrit", "Kannada"),
    ("Hindi",    "Sanskrit"),
    ("Hindi",    "Kannada"),
    ("Kannada",  "Sanskrit"),
    ("Kannada",  "Hindi"),
]

SPAN_NAMES = ["pre_instr", "instruction", "src", "post_src", "tgt_self"]
SPAN_COLORS = {
    "pre_instr":   "#888888",
    "instruction": "#ff8c42",
    "src":         "#2e86ab",
    "post_src":    "#a23b72",
    "tgt_self":    "#6b9a3a",
}
SPAN_LABELS = {
    "pre_instr":   "Pre-instruction scaffolding",
    "instruction": "Instruction (\"Translate X to Y:\")",
    "src":         "Source sentence",
    "post_src":    "Post-source scaffolding (think block)",
    "tgt_self":    "Target self-attention",
}


# ── Span detection (all spans, not just src / tgt) ────────────────────────

def find_subseq(seq, subseq):
    n, m = len(seq), len(subseq)
    if m == 0:
        return -1
    for i in range(n - m + 1):
        if seq[i: i + m] == subseq:
            return i
    return -1


def build_and_locate_all_spans(tokenizer, src_lang_name, tgt_lang_name,
                                src_text, tgt_text):
    """
    Build chat-templated sequence and locate every span.

    Returns dict:
        full_ids:   list[int]
        spans:      dict[str, list[int]] with keys
                    pre_instr, instruction, src, post_src, tgt
    Returns None if src or tgt span can't be located.
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
    T = len(full_ids)

    # Source span: tokenize src alone; find it inside full_ids.
    src_ids = tokenizer.encode(src_text, add_special_tokens=False)
    src_start = find_subseq(full_ids, src_ids)
    if src_start == -1:
        return None
    src_end = src_start + len(src_ids)  # exclusive
    src_positions = list(range(src_start, src_end))

    # Target span
    tgt_ids = tokenizer.encode(tgt_text, add_special_tokens=False)
    tgt_start = find_subseq(full_ids, tgt_ids)
    if tgt_start == -1:
        return None
    tgt_end = tgt_start + len(tgt_ids)
    tgt_positions = list(range(tgt_start, tgt_end))

    if tgt_start <= src_end - 1:
        return None

    # Instruction: the fixed wrapper text "Translate the following X to Y:\n"
    instruction_text = f"Translate the following {src_lang_name} text to {tgt_lang_name}:\n"
    instr_ids = tokenizer.encode(instruction_text, add_special_tokens=False)
    instr_start = find_subseq(full_ids, instr_ids)
    if instr_start == -1:
        # Fall back: everything in user turn before src is "instruction"
        # (happens if tokenization splits across the instruction/src boundary)
        instr_start = 0
        instr_end = src_start
    else:
        instr_end = instr_start + len(instr_ids)

    # Pre-instruction scaffolding: [0 ... instr_start)
    pre_instr_positions = list(range(0, instr_start))
    instruction_positions = list(range(instr_start, min(instr_end, src_start)))

    # Post-source scaffolding: [src_end ... tgt_start)
    post_src_positions = list(range(src_end, tgt_start))

    # Sanity: spans should partition [0, tgt_start) completely, no overlap
    assigned = set()
    for p in (pre_instr_positions + instruction_positions + src_positions + post_src_positions):
        if p in assigned:
            return None  # overlap — skip
        assigned.add(p)
    if assigned != set(range(0, tgt_start)):
        # gap or misalignment — try to patch by extending post_src / pre_instr
        missing = set(range(0, tgt_start)) - assigned
        if missing:
            # allocate any missing positions to post_src (most likely cause:
            # whitespace-merging across the src/post-src boundary)
            post_src_positions = sorted(set(post_src_positions) | missing)

    return {
        "full_ids":     full_ids,
        "pre_instr":    pre_instr_positions,
        "instruction":  instruction_positions,
        "src":          src_positions,
        "post_src":     post_src_positions,
        "tgt":          tgt_positions,
    }


# ── Per-sample metrics (all spans, all target positions) ──────────────────

def compute_sample_metrics(model, tokenizer, src_lang_name, tgt_lang_name,
                           src_text, tgt_text, device, max_len):
    """
    Returns:
        layer_span_frac[l][span_name]         — mean over heads & tgt tokens
        layer_span_frac_first[l][span_name]   — measured at the first 1 tgt token
        layer_span_frac_later[l][span_name]   — measured at all but first 3 tgt tokens
        n_tgt                                 — how many target tokens we had
        sanity_sum[l]                         — should be ~1.0; for QA
    Returns None if spans can't be built or sequence too long.
    """
    result = build_and_locate_all_spans(
        tokenizer, src_lang_name, tgt_lang_name, src_text, tgt_text
    )
    if result is None:
        return None
    if len(result["full_ids"]) > max_len:
        return None

    full_ids = result["full_ids"]
    tgt_positions = result["tgt"]
    if len(tgt_positions) < 2:
        return None

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    tgt_pos = torch.tensor(tgt_positions, dtype=torch.long, device=device)

    # Build a position tensor for each input span
    span_pos = {}
    for s in ("pre_instr", "instruction", "src", "post_src"):
        if len(result[s]) > 0:
            span_pos[s] = torch.tensor(result[s], dtype=torch.long, device=device)
        else:
            span_pos[s] = None

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_attentions=True,
            return_dict=True,
        )

    # First-token index and later-token indices (within the tgt span)
    first_tgt_idx = torch.tensor([0], device=device)
    later_mask = torch.arange(len(tgt_positions), device=device) >= 3

    layer_all   = {}
    layer_first = {}
    layer_later = {}
    sanity      = {}

    for l, A in enumerate(outputs.attentions):
        A = A[0]  # (H, T, T)
        # Rows for target positions: (H, n_tgt, T)
        tgt_attn = A[:, tgt_pos, :]

        # Target self-attention: mass on positions strictly before each tgt token
        # that are also inside the tgt span. Because the model is causal, every
        # position ≥ first tgt and < current tgt is a previous target token.
        # Build a mask (n_tgt, T) where entry (i, j) = 1 iff j is in tgt and j < tgt_pos[i]
        n_tgt = len(tgt_positions)
        T     = A.shape[-1]
        tgt_set = set(tgt_positions)
        tgt_self_mask = torch.zeros(n_tgt, T, dtype=torch.bool, device=device)
        for i, tp in enumerate(tgt_positions):
            for j in range(tp):
                if j in tgt_set:
                    tgt_self_mask[i, j] = True

        fractions = {}

        # Input spans: sum over their positions
        for s, pos in span_pos.items():
            if pos is None:
                fractions[s] = 0.0
                continue
            # tgt_attn[:, :, pos] → (H, n_tgt, |pos|)
            mass = tgt_attn[:, :, pos].sum(dim=-1)   # (H, n_tgt)
            fractions[s] = mass   # keep per-tgt for first/later split later

        # Target self-attention: variable set per tgt position
        # Use broadcasting: (H, n_tgt, T) * (n_tgt, T) → sum over T
        tgt_self_mass = (tgt_attn * tgt_self_mask.unsqueeze(0)).sum(dim=-1)  # (H, n_tgt)
        fractions["tgt_self"] = tgt_self_mass

        # Aggregate: mean over H and all tgt positions
        all_frac = {}
        for s, m in fractions.items():
            if isinstance(m, torch.Tensor):
                all_frac[s] = m.mean().item()
            else:
                all_frac[s] = m
        layer_all[l] = all_frac

        # First target token only
        first_frac = {}
        for s, m in fractions.items():
            if isinstance(m, torch.Tensor):
                # m: (H, n_tgt) → take position 0, mean over heads
                first_frac[s] = m[:, 0].mean().item()
            else:
                first_frac[s] = m
        layer_first[l] = first_frac

        # Later target tokens (≥ 3)
        later_frac = {}
        has_later = later_mask.any().item()
        for s, m in fractions.items():
            if isinstance(m, torch.Tensor) and has_later:
                later_frac[s] = m[:, later_mask].mean().item()
            else:
                later_frac[s] = 0.0 if not has_later else m
        layer_later[l] = later_frac

        # Sanity: total mass per target token (all spans + self) should be ~1
        total = sum(all_frac[s] for s in SPAN_NAMES)
        sanity[l] = total

    return {
        "layer_all":   layer_all,
        "layer_first": layer_first,
        "layer_later": layer_later,
        "sanity":      sanity,
        "n_tgt":       len(tgt_positions),
    }


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
    acc_all   = defaultdict(lambda: defaultdict(list))
    acc_first = defaultdict(lambda: defaultdict(list))
    acc_later = defaultdict(lambda: defaultdict(list))
    acc_sanity = defaultdict(list)
    n_valid  = 0
    n_skipped = 0

    for _, row in tqdm(rows.iterrows(), total=len(rows), desc=f"  {key}", leave=False):
        metrics = compute_sample_metrics(
            model, tokenizer,
            src_name, tgt_name,
            str(row["src_text"]), str(row["tgt_text"]),
            device, max_len,
        )
        if metrics is None:
            n_skipped += 1
            continue
        n_valid += 1
        for l, d in metrics["layer_all"].items():
            for s, v in d.items():
                acc_all[l][s].append(v)
        for l, d in metrics["layer_first"].items():
            for s, v in d.items():
                acc_first[l][s].append(v)
        for l, d in metrics["layer_later"].items():
            for s, v in d.items():
                acc_later[l][s].append(v)
        for l, v in metrics["sanity"].items():
            acc_sanity[l].append(v)

    if n_valid == 0:
        return None

    def reduce_acc(acc):
        out = {}
        for l, sd in acc.items():
            out[l] = {s: float(np.mean(vs)) for s, vs in sd.items()}
        return out

    return {
        "all":      reduce_acc(acc_all),
        "first":    reduce_acc(acc_first),
        "later":    reduce_acc(acc_later),
        "sanity":   {l: float(np.mean(vs)) for l, vs in acc_sanity.items()},
        "n_valid":  n_valid,
        "n_skipped": n_skipped,
    }


# ── Plotting ───────────────────────────────────────────────────────────────

def macro_avg_span_matrix(results_by_direction, key="all"):
    """Returns (layers_sorted, dict[span] -> np.array of per-layer macro means)."""
    layers = sorted(next(iter(results_by_direction.values()))[key].keys())
    per_span = {s: [] for s in SPAN_NAMES}
    for l in layers:
        for s in SPAN_NAMES:
            vals = [results_by_direction[d][key][l][s]
                    for d in results_by_direction
                    if l in results_by_direction[d][key]]
            per_span[s].append(np.mean(vals) if vals else 0.0)
    for s in SPAN_NAMES:
        per_span[s] = np.array(per_span[s])
    return layers, per_span


def plot_full_attribution(results_by_direction, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    layers, per_span = macro_avg_span_matrix(results_by_direction, "all")
    x = [l + 1 for l in layers]

    # Figure layout: 2x2
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Qwen3-4B — Full Attention Attribution Across Layers\n"
        "Where target tokens send their attention at each layer",
        fontsize=13, fontweight="bold",
    )

    # ── Panel 1: Stacked area of span composition ─────────────────────────
    ax1 = axes[0, 0]
    stack_order = ["pre_instr", "instruction", "src", "post_src", "tgt_self"]
    colors = [SPAN_COLORS[s] for s in stack_order]
    ys = [per_span[s] for s in stack_order]
    labels = [SPAN_LABELS[s] for s in stack_order]
    ax1.stackplot(x, *ys, labels=labels, colors=colors, alpha=0.85)
    ax1.set_title("Span composition (macro avg across 6 directions)",
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel("Layer index")
    ax1.set_ylabel("Fraction of attention mass")
    ax1.set_ylim(0, 1.02)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Per-span curves ──────────────────────────────────────────
    ax2 = axes[0, 1]
    for s in SPAN_NAMES:
        ax2.plot(x, per_span[s], marker="o", markersize=3, linewidth=1.8,
                 color=SPAN_COLORS[s], label=SPAN_LABELS[s])
    # Annotate peaks
    for s in ("src", "instruction", "tgt_self"):
        peak_idx = int(np.argmax(per_span[s]))
        ax2.annotate(
            f"peak L{x[peak_idx]}\n{per_span[s][peak_idx]:.2f}",
            xy=(x[peak_idx], per_span[s][peak_idx]),
            xytext=(6, 6), textcoords="offset points",
            fontsize=7, color=SPAN_COLORS[s],
            arrowprops=dict(arrowstyle="->", lw=0.8, color=SPAN_COLORS[s]),
        )
    ax2.set_title("Each span as its own curve", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Layer index")
    ax2.set_ylabel("Fraction of attention mass")
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Per-direction src attribution ────────────────────────────
    ax3 = axes[1, 0]
    import seaborn as sns
    palette = sns.color_palette("tab10", len(results_by_direction))
    for (direction, data), color in zip(results_by_direction.items(), palette):
        y = [data["all"][l]["src"] for l in sorted(data["all"].keys())]
        ax3.plot(x, y, marker="o", markersize=3, linewidth=1.6,
                 label=direction, color=color)
    ax3.set_title("Source-span attention per direction", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Layer index")
    ax3.set_ylabel("Fraction on source sentence")
    ax3.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax3.legend(fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: First-token vs continuation ──────────────────────────────
    ax4 = axes[1, 1]
    _, first_span = macro_avg_span_matrix(results_by_direction, "first")
    _, later_span = macro_avg_span_matrix(results_by_direction, "later")

    ax4.plot(x, first_span["src"], color=SPAN_COLORS["src"], linewidth=2,
             linestyle="-", marker="o", markersize=3,
             label="Source — first tgt token")
    ax4.plot(x, later_span["src"], color=SPAN_COLORS["src"], linewidth=2,
             linestyle="--", marker="s", markersize=3,
             label="Source — later tgt tokens (≥3)")
    ax4.plot(x, first_span["instruction"], color=SPAN_COLORS["instruction"],
             linewidth=2, linestyle="-", marker="o", markersize=3,
             label="Instruction — first tgt token")
    ax4.plot(x, later_span["instruction"], color=SPAN_COLORS["instruction"],
             linewidth=2, linestyle="--", marker="s", markersize=3,
             label="Instruction — later tgt tokens (≥3)")
    ax4.plot(x, later_span["tgt_self"], color=SPAN_COLORS["tgt_self"],
             linewidth=2, linestyle="--", marker="s", markersize=3,
             label="Target self — later tgt tokens (≥3)")
    ax4.set_title("First token vs continuation (macro avg)",
                  fontsize=11, fontweight="bold")
    ax4.set_xlabel("Layer index")
    ax4.set_ylabel("Fraction of attention mass")
    ax4.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax4.legend(fontsize=8, loc="upper right")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "qwen3_full_attribution.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Plot → {fig_path}")
    plt.close()


def save_csv_and_sanity(results_by_direction, output_dir):
    rows = []
    for direction, data in results_by_direction.items():
        for view in ("all", "first", "later"):
            for l, sd in data[view].items():
                for s, v in sd.items():
                    rows.append({
                        "direction": direction,
                        "view":      view,
                        "layer":     l + 1,
                        "span":      s,
                        "fraction":  round(v, 5),
                    })
    csv_path = os.path.join(output_dir, "qwen3_full_attribution.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  CSV  → {csv_path}")

    # Sanity report
    sanity_rows = []
    for direction, data in results_by_direction.items():
        for l, v in data["sanity"].items():
            sanity_rows.append({"direction": direction, "layer": l + 1,
                                "total_mass": round(v, 5)})
    sanity_df = pd.DataFrame(sanity_rows)
    san_path = os.path.join(output_dir, "qwen3_sanity_total_mass.csv")
    sanity_df.to_csv(san_path, index=False)
    mn, mx = sanity_df["total_mass"].min(), sanity_df["total_mass"].max()
    print(f"  Sanity (per-layer total mass across all 5 spans): min={mn:.4f}  max={mx:.4f}")
    if mx - mn > 0.05 or abs(1.0 - (mn + mx) / 2) > 0.03:
        print(f"  ⚠️  Sanity total deviates noticeably from 1.0 — inspect {san_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       required=True, help="JSONL eval file")
    p.add_argument("--model",      default="Qwen/Qwen3-4B")
    p.add_argument("--n_samples",  type=int, default=50)
    p.add_argument("--max_len",    type=int, default=512)
    p.add_argument("--output_dir", default="./attribution_output/qwen3_full")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    ).eval().to(args.device)
    n_layers = model.config.num_hidden_layers
    print(f"Layers: {n_layers}   Device: {args.device}")

    df = pd.read_json(args.data, lines=True)
    required = {"src_lang", "tgt_lang", "src_text", "tgt_text"}
    df = df[list(required)].dropna().reset_index(drop=True)
    print(f"Loaded {len(df)} samples from {args.data}")

    results_by_direction = {}
    total_skipped = 0
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
            total_skipped += result["n_skipped"]
            print(f"    n_valid={result['n_valid']}  n_skipped={result['n_skipped']}")

    if not results_by_direction:
        print("No results to plot.")
        return

    if total_skipped > 0:
        print(f"\n⚠️  Skipped {total_skipped} samples total due to span-detection failure.")
        print("   If this is > ~5% of your data, inspect a few failures manually.")

    print("\nPlotting...")
    plot_full_attribution(results_by_direction, args.output_dir)
    save_csv_and_sanity(results_by_direction, args.output_dir)

    # Compact summary
    layers, per_span = macro_avg_span_matrix(results_by_direction, "all")
    x = [l + 1 for l in layers]
    print("\n── Macro-average summary ───────────────────────────────────")
    for s in SPAN_NAMES:
        peak_idx = int(np.argmax(per_span[s]))
        early = per_span[s][:8].mean()
        late  = per_span[s][-8:].mean()
        print(f"  {s:14s} peak L{x[peak_idx]:<3d} ({per_span[s][peak_idx]:.3f})  "
              f"early={early:.3f}  late={late:.3f}")
    print("Done.")


if __name__ == "__main__":
    main()