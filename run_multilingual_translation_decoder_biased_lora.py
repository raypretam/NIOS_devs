#!/usr/bin/env python
# Joint Multilingual Fine-tuning of NLLB-600M
# All 6 directions: san↔hin, san↔kan, hin↔kan
# Decoder-biased LoRA: encoder r=8, decoder self-attn r=32, decoder cross-attn r=64
#
# Dependencies:
#   pip install transformers>=4.40.0 accelerate>=0.26.0 datasets>=2.18.0
#   pip install peft>=0.10.0 sentencepiece protobuf sacrebleu evaluate torch

"""
═══════════════════════════════════════════════════════════════════════════════
DECODER-BIASED LORA RATIONALE
═══════════════════════════════════════════════════════════════════════════════
From inference results:
  Kannada→Hindi  25.52  (encoder reads Kannada fine)
  Sanskrit→Hindi 21.63  (encoder reads Sanskrit fine)
  Hindi→Sanskrit 11.75  (DECODER can't generate Sanskrit well)
  Kannada→Sanskrit 5.86 (DECODER can't generate Sanskrit well)

The encoder already learned decent representations for all three scripts.
The decoder — especially cross-attention which reads encoder output while
generating target tokens — is the bottleneck.

Asymmetric LoRA ranks address this directly:
  Encoder self-attention   → r=8   (already works well, small adaptation)
  Decoder self-attention   → r=32  (target language generation)
  Decoder cross-attention  → r=64  (most critical: src→tgt transfer)

PEFT does not support per-module rank in a single LoraConfig, so we apply
three separate named adapters and activate all three simultaneously during
training.

═══════════════════════════════════════════════════════════════════════════════
USAGE
═══════════════════════════════════════════════════════════════════════════════
python run_multilingual_translation.py \
    --model_name_or_path facebook/nllb-200-distilled-600M \
    --train_file data/train_multilingual.jsonl \
    --validation_file data/val_multilingual.jsonl \
    --output_dir ./nllb_decoder_biased_lora \
    --encoder_lora_r 8 \
    --decoder_sa_lora_r 32 \
    --decoder_ca_lora_r 64 \
    --lora_alpha_multiplier 2 \
    --lora_dropout 0.1 \
    --replay_buffer_ratio 0.05 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --warmup_steps 500 \
    --label_smoothing_factor 0.1 \
    --fp16 True \
    --predict_with_generate True \
    --do_train True \
    --do_eval True \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_steps 500 \
    --load_best_model_at_end True \
    --metric_for_best_model avg_bleu
"""

import json
import logging
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import datasets
import evaluate
import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

# ──────────────────────────────────────────────────────────────────────────────
# Language codes
# ──────────────────────────────────────────────────────────────────────────────
LANG_CODES = {
    "san": "san_Deva",
    "hin": "hin_Deva",
    "kan": "kan_Knda",
}

ALL_DIRECTIONS = [
    ("san_Deva", "hin_Deva"),
    ("hin_Deva", "san_Deva"),
    ("san_Deva", "kan_Knda"),
    ("kan_Knda", "san_Deva"),
    ("hin_Deva", "kan_Knda"),
    ("kan_Knda", "hin_Deva"),
]

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# § 1  DECODER-BIASED LORA
# ══════════════════════════════════════════════════════════════════════════════

def apply_decoder_biased_lora(
    model,
    encoder_r: int = 8,
    decoder_sa_r: int = 32,
    decoder_ca_r: int = 64,
    lora_alpha_multiplier: int = 2,
    lora_dropout: float = 0.1,
):
    """
    Apply decoder-biased LoRA using a SINGLE LoraConfig with rank_pattern
    and alpha_pattern — the correct PEFT API for per-module rank overrides.

    This avoids all multi-adapter API issues (enable_adapters, add_adapter,
    set_adapter) which have broken signatures across PEFT versions.

    rank_pattern: maps a substring of the full module path to an override rank.
    Any module whose full dotted name contains the key gets that rank.
    Modules not matched by any key get the base rank r=encoder_r.

    NLLB-600M (M2M100) module paths:
      model.encoder.layers.{i}.self_attn.{q,k,v,out}_proj   → base r
      model.decoder.layers.{i}.self_attn.{q,k,v,out}_proj   → decoder_sa_r
      model.decoder.layers.{i}.encoder_attn.{q,k,v,out}_proj → decoder_ca_r
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        raise ImportError("pip install peft>=0.10.0")

    # ── Enumerate all target module full paths ────────────────────────────────
    # We build rank_pattern and alpha_pattern keyed on the FULL dotted module
    # name (e.g. "model.decoder.layers.3.encoder_attn.q_proj").
    # PEFT checks: any(key in full_module_name for key in rank_pattern).
    # Using full paths eliminates any ambiguity between encoder/decoder modules.
    rank_pattern  = {}
    alpha_pattern = {}

    enc_sa_count = dec_sa_count = dec_ca_count = 0

    for name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue
        is_proj = any(name.endswith(p) for p in
                      ["q_proj", "k_proj", "v_proj", "out_proj"])
        if not is_proj:
            continue

        if "decoder" in name and "encoder_attn" in name:
            # Decoder cross-attention — highest rank
            rank_pattern[name]  = decoder_ca_r
            alpha_pattern[name] = decoder_ca_r * lora_alpha_multiplier
            dec_ca_count += 1
        elif "decoder" in name and "self_attn" in name:
            # Decoder self-attention — medium rank
            rank_pattern[name]  = decoder_sa_r
            alpha_pattern[name] = decoder_sa_r * lora_alpha_multiplier
            dec_sa_count += 1
        elif "encoder" in name and "self_attn" in name:
            # Encoder self-attention — base rank (no entry needed, but log it)
            enc_sa_count += 1

    logger.info(f"Encoder SA modules : {enc_sa_count} → r={encoder_r}  (base)")
    logger.info(f"Decoder SA modules : {dec_sa_count} → r={decoder_sa_r}")
    logger.info(f"Decoder CA modules : {dec_ca_count} → r={decoder_ca_r}")

    if dec_ca_count == 0:
        raise RuntimeError(
            "No decoder cross-attention modules found. "
            "Check model architecture — expected 'encoder_attn' in decoder layer names."
        )

    # ── Single LoraConfig with rank_pattern ───────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=encoder_r,                               # base rank → encoder SA
        lora_alpha=encoder_r * lora_alpha_multiplier,
        lora_dropout=lora_dropout,
        # Target all attention projections in both encoder and decoder
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        rank_pattern=rank_pattern,                 # per-module rank overrides
        alpha_pattern=alpha_pattern,               # per-module alpha overrides
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    # ── Freeze embeddings — protect NLLB's 200-language vocab ────────────────
    for name, param in model.named_parameters():
        if "embed" in name:
            param.requires_grad = False

    # ── Parameter summary ─────────────────────────────────────────────────────
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"\n{'═'*58}\n"
        f"  Decoder-Biased LoRA (rank_pattern API)\n"
        f"  Encoder SA     r={encoder_r:<4}  α={encoder_r * lora_alpha_multiplier}\n"
        f"  Decoder SA     r={decoder_sa_r:<4}  α={decoder_sa_r * lora_alpha_multiplier}\n"
        f"  Decoder CA     r={decoder_ca_r:<4}  α={decoder_ca_r * lora_alpha_multiplier}\n"
        f"  Trainable : {trainable:>12,}  ({100*trainable/total:.3f}%)\n"
        f"  Total     : {total:>12,}\n"
        f"{'═'*58}"
    )
    return model


def verify_adapter_ranks(model):
    """
    Print effective LoRA rank per module type.
    Confirms rank_pattern wired encoder_SA/decoder_SA/decoder_CA correctly.
    """
    try:
        from peft.tuners.lora import Linear as LoraLinear
    except ImportError:
        return

    logger.info("\n── Effective LoRA rank per module ───────────────────────────")
    seen = set()
    for name, module in model.named_modules():
        if isinstance(module, LoraLinear):
            for adapter_name, lora_A in module.lora_A.items():
                r = lora_A.weight.shape[0]
                if "encoder_attn" in name:
                    mtype = "decoder_CA"
                elif "decoder" in name and "self_attn" in name:
                    mtype = "decoder_SA"
                else:
                    mtype = "encoder_SA"
                # Show one representative row per (type, proj_name) pair
                proj = name.split(".")[-1]
                key  = f"{mtype}.{proj}"
                if key not in seen:
                    seen.add(key)
                    logger.info(f"  {key:40s}  r={r}")
    logger.info("─────────────────────────────────────────────────────────────")


# ══════════════════════════════════════════════════════════════════════════════
# § 2  ARGUMENT DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="facebook/nllb-200-distilled-600M",
    )
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)

    # Decoder-biased LoRA ranks
    encoder_lora_r: int = field(
        default=8,
        metadata={"help": "LoRA rank for encoder self-attention."},
    )
    decoder_sa_lora_r: int = field(
        default=32,
        metadata={"help": "LoRA rank for decoder self-attention."},
    )
    decoder_ca_lora_r: int = field(
        default=64,
        metadata={"help": "LoRA rank for decoder cross-attention (highest priority)."},
    )
    lora_alpha_multiplier: int = field(
        default=2,
        metadata={"help": "lora_alpha = r * multiplier. 2 is standard."},
    )
    lora_dropout: float = field(default=0.1)


@dataclass
class DataTrainingArguments:
    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    test_file: Optional[str] = field(default=None)
    max_source_length: int = field(default=128)
    max_target_length: int = field(default=128)
    val_max_target_length: Optional[int] = field(default=None)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    pad_to_max_length: bool = field(default=False)
    ignore_pad_token_for_loss: bool = field(default=True)
    num_beams: int = field(default=5)
    replay_buffer_ratio: float = field(default=0.05)
    temperature: float = field(default=3.0)

    def __post_init__(self):
        if self.train_file is None and self.validation_file is None:
            raise ValueError("Provide --train_file and/or --validation_file.")
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


# ══════════════════════════════════════════════════════════════════════════════
# § 3  PER-EXAMPLE TOKENIZATION
# ══════════════════════════════════════════════════════════════════════════════

def make_preprocess_fn(tokenizer, max_source_length, max_target_length, padding):
    def preprocess_function(examples):
        all_input_ids, all_attention_mask, all_labels = [], [], []
        for src_text, tgt_text, src_lang, tgt_lang in zip(
            examples["src_text"], examples["tgt_text"],
            examples["src_lang"], examples["tgt_lang"],
        ):
            tokenizer.src_lang = src_lang
            enc = tokenizer(src_text, max_length=max_source_length,
                            padding=padding, truncation=True)
            tokenizer.tgt_lang = tgt_lang
            dec = tokenizer(text_target=tgt_text, max_length=max_target_length,
                            padding=padding, truncation=True)
            label_ids = [
                tok if tok != tokenizer.pad_token_id else -100
                for tok in dec["input_ids"]
            ]
            all_input_ids.append(enc["input_ids"])
            all_attention_mask.append(enc["attention_mask"])
            all_labels.append(label_ids)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
            "src_lang": examples["src_lang"],
            "tgt_lang": examples["tgt_lang"],
        }
    return preprocess_function


# ══════════════════════════════════════════════════════════════════════════════
# § 4  FLORES-200 REPLAY BUFFER
# ══════════════════════════════════════════════════════════════════════════════

def build_replay_buffer(n_target, tokenizer, max_source_length, max_target_length):
    FLORES_LOCAL_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "flores200_dataset", "dev"
    )
    lang_files = {}
    for lang_code in ["san_Deva", "hin_Deva", "kan_Knda"]:
        path = os.path.join(FLORES_LOCAL_DIR, f"{lang_code}.dev")
        if not os.path.exists(path):
            logger.warning(f"Flores file not found: {path}. Skipping replay buffer.")
            return None
        with open(path, encoding="utf-8") as f:
            lang_files[lang_code] = [l.strip() for l in f if l.strip()]
        logger.info(f"Loaded {len(lang_files[lang_code])} Flores sentences for {lang_code}")

    n_sentences = min(len(v) for v in lang_files.values())
    flores_rows = [
        {"src_lang": src, "tgt_lang": tgt,
         "src_text": lang_files[src][i], "tgt_text": lang_files[tgt][i]}
        for i in range(n_sentences)
        for src, tgt in ALL_DIRECTIONS
    ]
    random.shuffle(flores_rows)
    flores_rows = flores_rows[:n_target]

    preprocess_fn = make_preprocess_fn(tokenizer, max_source_length, max_target_length, False)
    replay_ds = Dataset.from_list(flores_rows)
    replay_ds = replay_ds.map(preprocess_fn, batched=True, batch_size=64,
                               num_proc=1, desc="Tokenizing replay buffer")
    logger.info(f"Replay buffer: {len(replay_ds):,} examples from Flores-200")
    return replay_ds


# ══════════════════════════════════════════════════════════════════════════════
# § 5  DATA COLLATOR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimpleSeq2SeqDataCollator:
    """
    Pads inputs and labels only.
    Does NOT inject decoder_input_ids to avoid the M2M100/NLLB conflict:
      'You cannot specify both decoder_input_ids and decoder_inputs_embeds'
    The model computes decoder_input_ids from labels internally.
    """
    tokenizer: Any
    label_pad_token_id: int = -100
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids  = [f["input_ids"]      for f in features]
        attn_masks = [f["attention_mask"] for f in features]
        labels     = [f["labels"]         for f in features]

        max_inp = max(len(x) for x in input_ids)
        max_lbl = max(len(x) for x in labels)

        if self.pad_to_multiple_of:
            max_inp = -(-max_inp // self.pad_to_multiple_of) * self.pad_to_multiple_of
            max_lbl = -(-max_lbl // self.pad_to_multiple_of) * self.pad_to_multiple_of

        padded_inp, padded_attn, padded_lbl = [], [], []
        for inp, attn, lbl in zip(input_ids, attn_masks, labels):
            pi = max_inp - len(inp)
            pl = max_lbl - len(lbl)
            padded_inp.append(inp   + [self.tokenizer.pad_token_id] * pi)
            padded_attn.append(attn + [0]                           * pi)
            padded_lbl.append(lbl   + [self.label_pad_token_id]     * pl)

        batch = {
            "input_ids":      torch.tensor(padded_inp,  dtype=torch.long),
            "attention_mask": torch.tensor(padded_attn, dtype=torch.long),
            "labels":         torch.tensor(padded_lbl,  dtype=torch.long),
        }
        if "src_lang" in features[0]:
            batch["src_lang"] = [f["src_lang"] for f in features]
        if "tgt_lang" in features[0]:
            batch["tgt_lang"] = [f["tgt_lang"] for f in features]
        return batch


# ══════════════════════════════════════════════════════════════════════════════
# § 6  CUSTOM TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class MultilingualSeq2SeqTrainer(Seq2SeqTrainer):

    def __init__(self, *args, tokenizer_ref=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokenizer_ref = tokenizer_ref

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Pass only the three keys NLLB expects — no decoder_input_ids
        model_inputs = {
            "input_ids":      inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels":         inputs["labels"],
        }
        outputs = model(**model_inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        cleaned = {
            "input_ids":      inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        if "labels" in inputs:
            cleaned["labels"] = inputs["labels"]
        return super().prediction_step(model, cleaned, prediction_loss_only, ignore_keys)


# ══════════════════════════════════════════════════════════════════════════════
# § 7  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    set_seed(training_args.seed)

    # ── 7.1  Tokenizer & base model ───────────────────────────────────────────
    config    = AutoConfig.from_pretrained(model_args.model_name_or_path,
                                           cache_dir=model_args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                              cache_dir=model_args.cache_dir,
                                              use_fast=model_args.use_fast_tokenizer)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path,
                                                      config=config,
                                                      cache_dir=model_args.cache_dir)

    for lang_code in LANG_CODES.values():
        tok_id = tokenizer.convert_tokens_to_ids(lang_code)
        if tok_id == tokenizer.unk_token_id:
            raise ValueError(f"Language token '{lang_code}' missing from vocab.")
        logger.info(f"  {lang_code} → token id {tok_id}  ✓")

    # ── 7.2  Decoder-biased LoRA ──────────────────────────────────────────────
    model = apply_decoder_biased_lora(
        model,
        encoder_r             = model_args.encoder_lora_r,
        decoder_sa_r          = model_args.decoder_sa_lora_r,
        decoder_ca_r          = model_args.decoder_ca_lora_r,
        lora_alpha_multiplier = model_args.lora_alpha_multiplier,
        lora_dropout          = model_args.lora_dropout,
    )
    verify_adapter_ranks(model)

    # ── 7.3  Datasets ─────────────────────────────────────────────────────────
    data_files = {k: v for k, v in [
        ("train",      data_args.train_file),
        ("validation", data_args.validation_file),
        ("test",       data_args.test_file),
    ] if v is not None}
    raw_datasets = load_dataset("json", data_files=data_files,
                                cache_dir=model_args.cache_dir)

    # ── 7.4  Tokenize ─────────────────────────────────────────────────────────
    padding      = "max_length" if data_args.pad_to_max_length else False
    preprocess_fn = make_preprocess_fn(tokenizer, data_args.max_source_length,
                                       data_args.max_target_length, padding)
    keep_cols = {"input_ids", "attention_mask", "labels", "src_lang", "tgt_lang"}

    def map_split(name, max_samples=None):
        ds = raw_datasets[name]
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        return ds.map(
            preprocess_fn, batched=True, batch_size=256, num_proc=1,
            remove_columns=[c for c in ds.column_names if c not in keep_cols],
            desc=f"Tokenizing {name}",
        )

    train_dataset   = None
    eval_dataset    = None
    predict_dataset = None

    if training_args.do_train and "train" in raw_datasets:
        train_dataset = map_split("train", data_args.max_train_samples)
        if data_args.replay_buffer_ratio > 0:
            n_replay = int(len(train_dataset) * data_args.replay_buffer_ratio)
            replay_ds = build_replay_buffer(n_replay, tokenizer,
                                            data_args.max_source_length,
                                            data_args.max_target_length)
            if replay_ds is not None:
                for col in train_dataset.column_names:
                    if col not in replay_ds.column_names:
                        replay_ds = replay_ds.add_column(col, [None] * len(replay_ds))
                train_dataset = concatenate_datasets([train_dataset, replay_ds])
                train_dataset = train_dataset.shuffle(seed=training_args.seed)
                logger.info(f"Train set: {len(train_dataset):,} "
                            f"(incl. {len(replay_ds):,} Flores replay)")

    if training_args.do_eval and "validation" in raw_datasets:
        eval_dataset = map_split("validation", data_args.max_eval_samples)

    if training_args.do_predict and "test" in raw_datasets:
        predict_dataset = map_split("test")

    # ── 7.5  Metrics ──────────────────────────────────────────────────────────
    sacrebleu = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds  = np.where(preds  != -100, preds,  tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        dp = [p.strip() for p in tokenizer.batch_decode(preds,  skip_special_tokens=True)]
        dl = [[l.strip()] for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        result = sacrebleu.compute(predictions=dp, references=dl)
        pred_lens = [np.count_nonzero(p != tokenizer.pad_token_id) for p in preds]
        return {
            "avg_bleu": round(result["score"], 4),
            "gen_len":  round(float(np.mean(pred_lens)), 4),
        }

    # ── 7.6  Collator ─────────────────────────────────────────────────────────
    label_pad_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = SimpleSeq2SeqDataCollator(
        tokenizer=tokenizer,
        label_pad_token_id=label_pad_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # ── 7.7  Trainer ──────────────────────────────────────────────────────────
    trainer = MultilingualSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        tokenizer_ref=tokenizer,
    )

    # ── 7.8  Train ────────────────────────────────────────────────────────────
    if training_args.do_train:
        result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        metrics = result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # ── 7.9  Eval ─────────────────────────────────────────────────────────────
    if training_args.do_eval and eval_dataset is not None:
        max_len = training_args.generation_max_length or data_args.val_max_target_length
        metrics = trainer.evaluate(max_length=max_len, num_beams=data_args.num_beams,
                                   metric_key_prefix="eval")
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # ── 7.10  Predict + per-direction BLEU ────────────────────────────────────
    if training_args.do_predict and predict_dataset is not None:
        max_len = training_args.generation_max_length or data_args.val_max_target_length
        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict",
            max_length=max_len, num_beams=data_args.num_beams,
        )
        preds = np.where(predict_results.predictions != -100,
                         predict_results.predictions, tokenizer.pad_token_id)
        decoded_preds = [p.strip() for p in
                         tokenizer.batch_decode(preds, skip_special_tokens=True)]

        if "src_lang" in predict_dataset.column_names:
            tgt_texts_raw = (predict_dataset["tgt_text"]
                             if "tgt_text" in predict_dataset.column_names else None)
            dir_results: dict = defaultdict(lambda: ([], []))
            for idx, (pred, src, tgt) in enumerate(zip(
                decoded_preds, predict_dataset["src_lang"], predict_dataset["tgt_lang"]
            )):
                key = f"{src}→{tgt}"
                dir_results[key][0].append(pred)
                if tgt_texts_raw:
                    dir_results[key][1].append([tgt_texts_raw[idx]])

            logger.info("\n══ Per-Direction BLEU ══════════════════════════")
            dir_bleus = {}
            for direction, (dp, dr) in sorted(dir_results.items()):
                if dr:
                    score = sacrebleu.compute(predictions=dp, references=dr)
                    dir_bleus[direction] = round(score["score"], 2)
                    logger.info(f"  {direction:30s}  BLEU = {score['score']:6.2f}  (n={len(dp)})")
            avg = np.mean(list(dir_bleus.values()))
            logger.info(f"  {'Average':30s}  BLEU = {avg:6.2f}")
            logger.info("═" * 50)
            with open(os.path.join(training_args.output_dir, "per_direction_bleu.json"), "w") as f:
                json.dump(dir_bleus, f, indent=2)

        if trainer.is_world_process_zero():
            with open(os.path.join(training_args.output_dir,
                                   "generated_predictions.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(decoded_preds))

    if training_args.push_to_hub:
        trainer.push_to_hub(finetuned_from=model_args.model_name_or_path,
                            tasks="translation", language=list(LANG_CODES.values()))


if __name__ == "__main__":
    main()