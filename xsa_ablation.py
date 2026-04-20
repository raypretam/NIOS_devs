"""
Script 3: XSA Ablation Study — Full Comparison Across All Conditions
======================================================================
Runs and compares four conditions:
  [A] Baseline   : standard SA + LoRA
  [B] XSA-Enc    : XSA on encoder SA only + LoRA
  [C] XSA-Dec    : XSA on decoder SA only + LoRA
  [D] XSA-Both   : XSA on encoder + decoder SA + LoRA

For each condition, evaluates BLEU and chrF on all six translation directions
and produces a consolidated comparison table and bar chart.

This script calls xsa_intervention.py as a subprocess, or you can run the
conditions independently and then just call the analysis portion.

Usage (full run):
    python xsa_ablation.py \
        --csv nios_test.csv \
        --output_dir ./xsa_ablation_output \
        --epochs 3 \
        --run_all        # trains all 4 conditions sequentially

Usage (analysis only, if you've already run script 2 for some conditions):
    python xsa_ablation.py \
        --output_dir ./xsa_ablation_output \
        --analyze_only \
        --results_dir ./xsa_intervention_output

Dependencies: same as script 2 + matplotlib seaborn
"""

import argparse
import os
import json
import subprocess
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

warnings.filterwarnings("ignore")

DIRECTIONS = [
    "San→Hin", "San→Kan",
    "Hin→San", "Hin→Kan",
    "Kan→San", "Kan→Hin",
]

CONDITIONS = [
    {"name": "baseline",  "xsa_enc": False, "xsa_dec": False, "label": "Baseline\n(standard SA)"},
    {"name": "xsa_enc",   "xsa_enc": True,  "xsa_dec": False, "label": "XSA-Enc\n(enc only)"},
    {"name": "xsa_dec",   "xsa_enc": False, "xsa_dec": True,  "label": "XSA-Dec\n(dec only)"},
    {"name": "xsa_both",  "xsa_enc": True,  "xsa_dec": True,  "label": "XSA-Both\n(enc+dec)"},
]


# ─── Training runner ──────────────────────────────────────────────────────

def run_condition(cond, args):
    """Launch xsa_intervention.py for one condition as a subprocess."""
    cmd = [
        sys.executable, "xsa_intervention.py",
        "--csv",        args.csv,
        "--model",      args.model,
        "--output_dir", args.output_dir,
        "--train_frac", str(args.train_frac),
        "--epochs",     str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr",         str(args.lr),
        "--lora_r",     str(args.lora_r),
        "--max_src_len",str(args.max_src_len),
    ]
    if cond["xsa_enc"]:
        cmd.append("--xsa_enc")
    if cond["xsa_dec"]:
        cmd.append("--xsa_dec")

    print(f"\n{'='*60}")
    print(f"Running condition: {cond['name']}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"WARNING: condition {cond['name']} exited with code {result.returncode}")


# ─── Load results ─────────────────────────────────────────────────────────

def load_results(output_dir):
    """Load all results_*.json files from output_dir."""
    all_results = {}
    for cond in CONDITIONS:
        path = os.path.join(output_dir, f"results_{cond['name']}.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            all_results[cond["name"]] = data["scores"]
            print(f"  Loaded: {cond['name']}")
        else:
            print(f"  Missing: {path}")
    return all_results


# ─── Analysis & Plotting ──────────────────────────────────────────────────

def build_summary_table(all_results):
    """Build a DataFrame with BLEU and chrF for each condition × direction."""
    rows = []
    for cond in CONDITIONS:
        name = cond["name"]
        if name not in all_results:
            continue
        scores = all_results[name]
        for direction, s in scores.items():
            rows.append({
                "condition": name,
                "label":     cond["label"],
                "direction": direction,
                "BLEU":      s["bleu"],
                "chrF":      s["chrf"],
            })
    return pd.DataFrame(rows)


def print_latex_table(df):
    """Print a LaTeX-ready comparison table."""
    pivot_bleu = df.pivot(index="direction", columns="condition", values="BLEU")
    pivot_chrf = df.pivot(index="direction", columns="condition", values="chrF")

    cond_order = [c["name"] for c in CONDITIONS if c["name"] in pivot_bleu.columns]
    pivot_bleu = pivot_bleu[cond_order]
    pivot_chrf = pivot_chrf[cond_order]

    print("\n--- BLEU scores ---")
    print(pivot_bleu.to_string())
    print("\n--- chrF scores ---")
    print(pivot_chrf.to_string())

    # Δ vs baseline
    if "baseline" in pivot_bleu.columns:
        print("\n--- BLEU Δ vs baseline ---")
        delta = pivot_bleu.drop(columns="baseline").subtract(pivot_bleu["baseline"], axis=0)
        print(delta.to_string(float_format=lambda x: f"{x:+.2f}"))

    # Mean scores
    print("\n--- Mean BLEU across directions ---")
    print(pivot_bleu.mean().to_string(float_format=lambda x: f"{x:.2f}"))
    print("\n--- Mean chrF across directions ---")
    print(pivot_chrf.mean().to_string(float_format=lambda x: f"{x:.2f}"))

    return pivot_bleu, pivot_chrf


def plot_grouped_bar(df, metric, output_dir):
    """Grouped bar chart: directions on x-axis, bars per condition."""
    cond_order = [c["name"] for c in CONDITIONS if c["name"] in df["condition"].unique()]
    labels     = {c["name"]: c["label"] for c in CONDITIONS}
    palette    = sns.color_palette("Set2", len(cond_order))

    fig, ax = plt.subplots(figsize=(13, 5))
    directions = sorted(df["direction"].unique())
    x = np.arange(len(directions))
    n = len(cond_order)
    width = 0.8 / n

    for i, cond in enumerate(cond_order):
        vals = []
        for d in directions:
            row = df[(df["condition"] == cond) & (df["direction"] == d)]
            vals.append(row[metric].values[0] if len(row) else 0)
        bars = ax.bar(x + (i - n/2 + 0.5) * width, vals, width * 0.9,
                      label=labels[cond], color=palette[i])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(directions, fontsize=9)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(
        f"{metric} scores by translation direction — Baseline vs XSA variants\n"
        f"(NLLB-200 + LoRA, nios_test.csv, {len(directions)} directions)",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, df[metric].max() * 1.15)

    out_path = os.path.join(output_dir, f"ablation_{metric.lower()}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()


def plot_delta_heatmap(pivot_bleu, pivot_chrf, output_dir):
    """Heatmap of BLEU and chrF deltas vs baseline."""
    if "baseline" not in pivot_bleu.columns:
        print("No baseline found — skipping delta heatmap.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (pivot, metric) in zip(axes, [(pivot_bleu, "BLEU"), (pivot_chrf, "chrF")]):
        delta = pivot.drop(columns="baseline").subtract(pivot["baseline"], axis=0)
        conds = [c["label"].replace("\n", " ") for c in CONDITIONS
                 if c["name"] in delta.columns]
        delta.columns = conds
        sns.heatmap(
            delta, ax=ax, annot=True, fmt="+.2f", cmap="RdYlGn",
            center=0, linewidths=0.5, linecolor="gray",
            cbar_kws={"label": f"Δ {metric} vs baseline"},
        )
        ax.set_title(f"Δ {metric} vs baseline", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=8)

    plt.suptitle("XSA ablation — improvement over standard SA (NLLB-200 + LoRA)",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "ablation_delta_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()


def plot_radar(df, metric, output_dir):
    """Radar chart: each condition's scores across all directions."""
    from matplotlib.patches import FancyArrowPatch
    cond_order = [c["name"] for c in CONDITIONS if c["name"] in df["condition"].unique()]
    labels_map = {c["name"]: c["label"] for c in CONDITIONS}
    palette    = sns.color_palette("Set2", len(cond_order))
    directions = sorted(df["direction"].unique())
    N = len(directions)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for cond, color in zip(cond_order, palette):
        vals = []
        for d in directions:
            row = df[(df["condition"] == cond) & (df["direction"] == d)]
            vals.append(row[metric].values[0] if len(row) else 0)
        vals += vals[:1]
        ax.plot(angles, vals, color=color, linewidth=2, label=labels_map[cond])
        ax.fill(angles, vals, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(directions, fontsize=9)
    ax.set_title(f"{metric} radar — XSA ablation", fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

    out_path = os.path.join(output_dir, f"ablation_radar_{metric.lower()}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",          default="nios_test.csv")
    p.add_argument("--model",        default="facebook/nllb-200-distilled-600M")
    p.add_argument("--output_dir",   default="./xsa_ablation_output")
    p.add_argument("--train_frac",   type=float, default=0.8)
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--lora_r",       type=int,   default=16)
    p.add_argument("--max_src_len",  type=int,   default=128)
    p.add_argument("--run_all",      action="store_true",
                   help="Run all 4 training conditions sequentially")
    p.add_argument("--analyze_only", action="store_true",
                   help="Skip training, only run analysis on existing result JSONs")
    p.add_argument("--results_dir",  default=None,
                   help="Directory containing results_*.json (defaults to output_dir)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    results_dir = args.results_dir or args.output_dir

    if args.run_all and not args.analyze_only:
        for cond in CONDITIONS:
            run_condition(cond, args)

    print("\nLoading results...")
    all_results = load_results(results_dir)

    if not all_results:
        print("\nNo results found. Run with --run_all first, or check --results_dir.")
        return

    df = build_summary_table(all_results)

    print("\n" + "="*60)
    print("ABLATION SUMMARY")
    print("="*60)
    pivot_bleu, pivot_chrf = print_latex_table(df)

    print("\nGenerating plots...")
    plot_grouped_bar(df, "BLEU", args.output_dir)
    plot_grouped_bar(df, "chrF", args.output_dir)
    plot_delta_heatmap(pivot_bleu, pivot_chrf, args.output_dir)
    plot_radar(df, "BLEU", args.output_dir)
    plot_radar(df, "chrF", args.output_dir)

    # Save consolidated CSV
    out_csv = os.path.join(args.output_dir, "ablation_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nConsolidated results → {out_csv}")
    print("\nDone.")


if __name__ == "__main__":
    main()
