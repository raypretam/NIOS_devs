# XSA × NLLB-200 Experiment Suite

Three scripts implementing the diagnostic → intervention → ablation pipeline
described in the XSA paper follow-up for encoder-decoder MT models.

## Setup

```bash
pip install transformers torch peft sacrebleu sentencepiece \
            pandas matplotlib seaborn tqdm
```

Your CSV should have three columns: `Hindi`, `Kannada`, `Sanskrit`
(one parallel sentence per row, as in nios_test.csv).

---

## Script 1 — Diagnostic (`xsa_diagnostic.py`)

**What it does:** Replicates Figure 1 of the XSA paper for NLLB-200.
Plots cos(y_i, v_i) per layer for encoder self-attention and decoder
masked self-attention, for each of the three languages.

**Interpret the output:**
- High cosine similarity (>0.4) and increasing trend with layer depth → bias is present
- Low or flat cosine similarity → attention is already contextual

```bash
python xsa_diagnostic.py \
    --csv nios_test.csv \
    --model facebook/nllb-200-distilled-600M \
    --n_samples 100 \
    --output_dir ./xsa_diagnostic_output
```

**Output:**
- `xsa_diagnostic_output/attention_similarity_bias.png` — plot per language × component
- `xsa_diagnostic_output/attention_similarity_bias.csv` — raw numbers

---

## Script 2 — Intervention (`xsa_intervention.py`)

**What it does:** Applies XSA to encoder and/or decoder self-attention,
trains LoRA adapters on the training split of your CSV across all 6
translation directions, and evaluates BLEU + chrF on the test split.

**Run baseline first:**
```bash
python xsa_intervention.py \
    --csv nios_test.csv \
    --output_dir ./xsa_intervention_output \
    --train_frac 0.8 \
    --epochs 3 \
    --lora_r 16
```

**Then run with XSA on encoder only:**
```bash
python xsa_intervention.py \
    --csv nios_test.csv \
    --output_dir ./xsa_intervention_output \
    --train_frac 0.8 \
    --epochs 3 \
    --lora_r 16 \
    --xsa_enc
```

**Output:**
- `xsa_intervention_output/lora_baseline/`   — saved LoRA weights
- `xsa_intervention_output/lora_xsa/`        — saved LoRA weights
- `xsa_intervention_output/results_baseline.json`
- `xsa_intervention_output/results_xsa.json`

---

## Script 3 — Ablation (`xsa_ablation.py`)

**What it does:** Orchestrates all four conditions and produces a full
comparison with grouped bar charts, delta heatmaps, and radar plots.

**Run all 4 conditions sequentially (takes time):**
```bash
python xsa_ablation.py \
    --csv nios_test.csv \
    --output_dir ./xsa_ablation_output \
    --epochs 3 \
    --run_all
```

**Or, if you've already run script 2 for some conditions:**
```bash
python xsa_ablation.py \
    --analyze_only \
    --results_dir ./xsa_intervention_output \
    --output_dir ./xsa_ablation_output
```

**Output:**
- `ablation_BLEU.png`, `ablation_chrF.png`    — grouped bar charts
- `ablation_delta_heatmap.png`                — Δ vs baseline heatmap
- `ablation_radar_bleu.png`                   — radar chart
- `ablation_summary.csv`                      — full table

---

## Experimental design

| Condition   | Encoder SA | Decoder SA |
|-------------|-----------|------------|
| baseline    | standard  | standard   |
| xsa_enc     | **XSA**   | standard   |
| xsa_dec     | standard  | **XSA**    |
| xsa_both    | **XSA**   | **XSA**    |

LoRA targets `q_proj` and `v_proj` in all attention layers, r=16.
Train/test split: 80/20 (stratified by dataset order, not shuffled,
to preserve domain coherence within split).

---

## Notes

- The XSA patch in Script 2 operates in the full embedding dimension D
  rather than per head. This is mathematically equivalent when out_proj
  is applied after aggregation, and avoids needing to crack open the
  internal head computation of MBartAttention.
- Cross-attention in the decoder is NOT patched — the XSA paper's
  argument about self-position bias does not apply there (queries and
  keys come from different sequences).
- For CPU-only machines, set `--n_samples 20` in script 1 and
  `--epochs 1` in scripts 2/3 for a quick sanity check.
