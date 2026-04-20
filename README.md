# NIOS Trilingual — Technique Development Log

Fine-tuning experiments on **NLLB-200-distilled-600M** for trilingual translation
across Sanskrit (san_Deva), Hindi (hin_Deva), and Kannada (kan_Knda) — all 6 directions.

**Dataset:** NIOS Trilingual Joint (`NIOS_Trilingual_Joint/`)
- `train_multilingual.jsonl` / `val_multilingual.jsonl`
- Format: `{src_lang, tgt_lang, src_text, tgt_text}` (directional JSONL)
- ~12K training pairs per direction (80/20 split)

---

## Techniques

### 1. Baseline LoRA

Uniform LoRA on `q_proj` + `v_proj` across all encoder and decoder attention layers.
Standard symmetric rank — no architectural insight applied.

| Script | `run_multilingual_translation.py` |
|--------|-----------------------------------|
| Rank   | r=16, α=32, dropout=0.1           |
| Target | `q_proj`, `v_proj` (all layers)   |

**Results (BLEU, val set):**

| Direction        | BLEU  |
|------------------|-------|
| Sanskrit→Hindi   | 21.63 |
| Sanskrit→Kannada | 9.05  |
| Hindi→Sanskrit   | 11.75 |
| Hindi→Kannada    | 16.68 |
| Kannada→Sanskrit | 5.86  |
| Kannada→Hindi    | 25.52 |
| **Avg**          | **15.08** |

---

### 2. Decoder-Biased LoRA

**Motivation:** Diagnostic analysis showed the encoder already forms decent
representations for all three scripts. The bottleneck is the decoder —
especially cross-attention, which transfers encoder context into target-language
generation. Sanskrit generation in particular suffered (Hindi→Sanskrit BLEU 11.75,
Kannada→Sanskrit 5.86).

**Solution:** Asymmetric LoRA ranks — higher capacity where it matters most.

| Script  | `run_multilingual_translation_decoder_biased_lora.py` |
|---------|-------------------------------------------------------|
| Enc SA  | r=8, α=16   (already works well)                      |
| Dec SA  | r=32, α=64  (target-language generation)              |
| Dec CA  | r=64, α=128 (src→tgt transfer — most critical)        |

Three named PEFT adapters are applied simultaneously (PEFT doesn't support
per-module rank in a single `LoraConfig`). Additional training details:
- Label smoothing: 0.1
- Replay buffer: 5% of previous data mixed in per epoch (anti-forgetting)
- LR: 2e-4, 10 epochs, warmup 500 steps, fp16

**Results (BLEU, val set):**

| Direction        | Baseline | Decoder-Biased | Δ     |
|------------------|----------|----------------|-------|
| Sanskrit→Hindi   | 21.63    | 23.69          | +2.06 |
| Sanskrit→Kannada | 9.05     | 10.35          | +1.30 |
| Hindi→Sanskrit   | 11.75    | 13.00          | +1.25 |
| Hindi→Kannada    | 16.68    | 19.12          | +2.44 |
| Kannada→Sanskrit | 5.86     | 7.26           | +1.40 |
| Kannada→Hindi    | 25.52    | 27.98          | +2.46 |
| **Avg**          | **15.08**| **16.90**      | **+1.82** |

---

### 3. XSA — Exclusive-Self-Attention Projection

**Paper:** [arxiv 2603.09078](https://arxiv.org/pdf/2603.09078)

**Motivation:** The XSA paper shows that in standard self-attention, each token's
output `y_i` retains a strong component along its own value vector `v_i`
(self-position bias). This reduces effective use of context. The fix is a
single parameter-free projection step applied to the attention output before `out_proj`:

```
z_i = y_i − (y_i · v̂_i) v̂_i      where v̂_i = v_i / ‖v_i‖
```

**Diagnostic:** `xsa_diagnostic.py` plots `cos(y_i, v_i)` per layer for each
language×component. Encoder self-attention showed increasing cosine similarity
with depth, confirming self-position bias in NLLB-200.

**Implementation:** `XSAPatch` wraps each `MBartAttention` module in-place,
applied before LoRA wrapping so adapters learn on top of the debiased attention.
Operates in full embedding dimension D (mathematically equivalent to per-head).
Cross-attention is deliberately not patched — XSA's argument doesn't apply there
(Q and K come from different sequences).

| Script   | `xsa_intervention.py`, `xsa_ablation.py`                        |
|----------|-----------------------------------------------------------------|
| Patch    | Encoder SA (`--xsa_enc`); optionally decoder SA (`--xsa_dec`)  |
| LoRA     | r=16, α=32, targets `q_proj` + `v_proj`                        |
| Ablation | 4 conditions: baseline / xsa_enc / xsa_dec / xsa_both           |


**Results (BLEU + chrF, test subset , `eval_predict_lora.py`):**

| Direction        | BLEU  | chrF  |
|------------------|-------|-------|
| Sanskrit→Hindi   | 31.52 | 65.0 |
| Sanskrit→Kannada | 20.18 | 58.9 |
| Hindi→Sanskrit   | 18.74 | 61.39 |
| Hindi→Kannada    | 25.2 | 64.5 |
| Kannada→Sanskrit | 14.15 | 55.63 |
| Kannada→Hindi    | 32.48 | 64.95 |
| **Avg**          | **23.72** |
Checkpoint: `xsa/xsa_intervention_output/lora_xsa/`

---

## Overall Comparison (Avg BLEU)

| Technique             | Avg BLEU | Δ vs Baseline |
|-----------------------|----------|---------------|
| Baseline LoRA         | 15.08    | —             |
| Decoder-Biased LoRA   | 16.90    | +1.82         |
| XSA + LoRA (enc+dec)  | 23.71    | +8.63         |

---

## Repository Layout

```
NIOS_devs/
├── run_multilingual_translation_decoder_biased_lora.py  # Technique 2
├── xsa_diagnostic.py          # XSA bias diagnostic — plots cos(y_i, v_i) per layer
├── xsa_intervention.py        # XSA patch + LoRA training (Technique 3)
├── xsa_ablation.py            # 4-condition ablation runner + visualisations
├── eval_predict_lora.py       # Load any LoRA checkpoint → BLEU/chrF + predictions.jsonl
├── xsa_intervention_output/
│   ├── lora_xsa/              # Saved XSA+LoRA adapter weights
│   └── results_xsa.json
├── eval_output_xsa/
│   ├── metrics.json
│   └── predictions.jsonl
└── README_xsa_experiments.md  # XSA-specific pipeline notes
```

---

## Evaluation

To reproduce evaluation on the val set using the XSA checkpoint:

```bash
python eval_predict_lora.py \
    --lora_path ./xsa_intervention_output/lora_xsa \
    --data ../NIOS_Trilingual_Joint/val_multilingual.jsonl \
    --xsa_enc --xsa_dec \
    --output_dir ./eval_output_xsa
```

Outputs `eval_output_xsa/metrics.json` and `eval_output_xsa/predictions.jsonl`.
