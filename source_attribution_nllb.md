# NLLB-600M Source Attribution Analysis

Layer-wise diagnostic of attention behaviour in `facebook/nllb-200-distilled-600M`
during translation across six NIOS directions (Sanskrit / Hindi / Kannada, both ways).
For every layer of both the encoder and decoder, the script measures *where
attention mass is going* and plots it across layers.

```bash
python nllb_source_attribution.py \
    --data ../NIOS_Trilingual_Joint/val_multilingual.jsonl \
    --model facebook/nllb-200-distilled-600M \
    --n_samples 100 \
    --output_dir ./attribution_output/nllb
```

Produces two artefacts in `--output_dir`:

- `nllb_source_attribution.png` — 3-panel figure (one line per direction)
- `nllb_source_attribution.csv` — the raw numbers used to draw the figure

---

## How the metrics are computed

For each sample, a teacher-forced forward pass is run with
`output_attentions=True` and `attn_implementation="eager"` (required so
transformers actually materialises attention weights). The encoder side is
padded to `max_len=128`; the decoder side is **not** padded (target length is
the real label length).

For each source sequence, token positions are partitioned into three groups
based on the NLLB encoder format `[lang_token, w_1 ... w_N, EOS, PAD ...]`:

| Group       | Positions                        |
|-------------|----------------------------------|
| `special`   | position 0 (language tag) and position `real_len − 1` (EOS) |
| `content`   | positions 1 … `real_len − 2` (the actual words) |
| padding     | everything after `real_len − 1`  |

Attention tensors coming out of the model are already post-softmax, so each
row sums to 1. The script then sums the relevant columns to get a mass
fraction in `[0, 1]` and averages over heads and real query positions.

---

## Panel 1 — Encoder self-attention

**Metric:** `enc_content_frac[l]` — for encoder layer `l`, the fraction of
self-attention mass that real (non-padded) source tokens send to *content*
keys (positions 1 … `real_len − 2`).

```
For each encoder layer l:
    A_l shape: (H, T_src, T_src)     — post-softmax
    A_real   = A_l[:, real_queries, :]
    frac     = A_real[:, :, content_mask].sum(dim=-1).mean()
```

**What the shape of the curve tells you:**

- Rising with depth → deeper encoder layers are pooling information across
  content tokens and building mixed representations.
- Flat or falling → later layers concentrate attention on the sentence-level
  summary slot (EOS) instead of mixing content. This is common in NLLB because
  the EOS position ends up acting as a "sentence vector."
- Similar curves across the three source languages → the encoder treats all
  three scripts roughly the same (which is also what the XSA diagnostic found).

Note that the complement of `enc_content_frac` is not padding — it's the mass
going to the two special positions (language tag + EOS). Padded keys are
already zero after softmax thanks to the encoder attention mask.

---

## Panel 2 — Decoder cross-attention

**Metric:** `dec_cross_frac[l]` — for decoder layer `l`, the fraction of
cross-attention mass that decoder tokens send to *content* source positions.

```
For each decoder layer l:
    A_l shape: (H, T_tgt, T_src)     — post-softmax, queries = decoder, keys = encoder
    frac = A_l[:, :, content_mask].sum(dim=-1).mean()
```

This is intended to answer **"when does the decoder look at the source?"** —
but there's an important caveat in how to read it.

**Important interpretation note.** The script also computes
`dec_cross_special[l]` (mass on the language tag and EOS positions) but
currently only plots the *content* line. In NLLB the EOS position carries an
outsized share of sentence-level information, so cross-attention mass is
naturally split between content tokens and the EOS "summary slot."
`dec_cross_frac` is therefore a **lower bound** on "how much the decoder is
reading from the source" — not a measure of total source-attribution. A value
in the 0.05 – 0.3 range does not mean the decoder is ignoring the encoder;
it means most of the mass is concentrated on EOS/language-tag positions
(which are themselves functions of the source).

To get total source attribution, sum `dec_cross_frac + dec_cross_special` or
sum attention over all non-padded source positions — that quantity should be
≈ 1.0 at every layer (useful as a sanity check that masking is working).

**What the shape of the curve tells you:**

- Shape rising then falling (peak in middle layers) → classic encoder-decoder
  pattern: early decoder layers set up target-side structure, middle layers do
  the heavy source-reading, late layers refine with self-attention and LM head.
- Monotonically falling → decoder is moving attention onto the EOS summary
  slot in deeper layers rather than individual content words.
- Large gap between directions (e.g. `San→Kan` vs `Kan→Hin`) → the decoder
  relies on content-position reading more for some directions than others,
  which often correlates with morphological alignment between the two
  languages.

---

## Panel 3 — Decoder self-attention entropy

**Metric:** `dec_self_ent[l]` — mean entropy (in nats) of the decoder's
masked self-attention distribution at layer `l`.

```
For each decoder layer l:
    A_l shape: (H, T_tgt, T_tgt)
    ent  = -(A_l * log(A_l)).sum(dim=-1)
    metric = ent.mean()
```

Entropy is a scale-free way of asking "how focused is this attention?"

| Value       | Interpretation                                      |
|-------------|-----------------------------------------------------|
| ≈ 0         | Attention hard-selects a single previous token      |
| ≈ log(k)    | Attention is uniform over roughly k tokens          |
| ≈ log(T_tgt)| Attention is uniform over the whole target prefix   |

For NLLB on these sentences you should expect values in the 0.5 – 1.5 nat
range, corresponding to effective attention over 1.6 – 4.5 tokens. Interpret
the curve as:

- Low and flat → decoder self-attention is tightly focused (usually on the
  most recent 1–2 tokens), which is typical late-layer behaviour for
  autoregressive generation.
- Higher / rising → decoder is aggregating more context from the partial
  target sequence, typical of middle layers.
- A distinctive dip at a particular layer → that layer is specialising into
  a single-token copy/select head.

Because the decoder is autoregressive, the entropy at query position `t` can
only range over the first `t` tokens (causal mask). The mean entropy
therefore naturally grows during generation; the *per-layer* pattern is what
carries the signal.

---

## CSV schema

`nllb_source_attribution.csv` has one row per (direction, metric, layer):

| column      | description                                               |
|-------------|-----------------------------------------------------------|
| `direction` | e.g. `San→Hin`, `Kan→San`                                 |
| `metric`    | one of `enc_content_frac`, `dec_cross_frac`, `dec_self_ent` |
| `layer`     | 1-indexed layer number                                    |
| `value`     | the aggregated mean for that (direction, metric, layer)   |

Values for `enc_content_frac` and `dec_cross_frac` are in `[0, 1]`; values
for `dec_self_ent` are in nats.

Note: `dec_cross_special` is computed per-sample but intentionally not
written to the CSV. If you want the total-source-mass sanity-check line,
add it to both `plot_results` and the CSV-writing loop.

---
