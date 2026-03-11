#!/usr/bin/env python3
"""
METIS SFT Seed Training — Cognitive Routing Format Learning

Trains Qwen2.5-7B-Instruct on the clean 1000-sample seed dataset
to learn FAST/DEEP cognitive routing format + <thinking> closure.

v3 Architecture (fixes v1/v2 failures):
  - modules_to_save=[embed_tokens, lm_head] — trains new token embeddings
  - apply_chat_template(tokenize=True) — guarantees train/inference format parity
  - Prompt masking — loss only on assistant response tokens

Hyperparameter bounds (locked against catastrophic forgetting):
  - LR: 2e-5 peak, cosine annealing, 5% warmup
  - Epochs: 2-3 (strict)
  - Weight decay: 0.01
  - LoRA: r=16, α=32, dropout=0.05
  - Targets: q_proj, v_proj, gate_proj, down_proj
  - modules_to_save: embed_tokens, lm_head (CRITICAL for new tokens)
  - Max length: 1024
  - Gradient checkpointing: ON (GB10 VRAM safety)

Usage:
  python tools/run_sft_seed.py
  python tools/run_sft_seed.py --base-model Qwen/Qwen2.5-7B-Instruct --epochs 3
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metis.training.tokenizer_utils import register_metis_special_tokens, verify_metis_tokens

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════

def load_sft_data(path: str) -> List[Dict[str, str]]:
    """Load SFT data from JSONL format {"system", "user", "assistant"}.

    Also supports legacy JSON [{"text": ...}] for backwards compatibility.
    """
    if path.endswith(".jsonl"):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    logger.info(f"Loaded {len(data)} SFT samples from {path}")

    # Validate
    for i, item in enumerate(data[:3]):
        text = item.get("assistant", item.get("text", ""))
        has_fast = "[COGNITIVE_STATE: FAST]" in text
        has_deep = "[COGNITIVE_STATE: DEEP]" in text
        logger.info(f"  Sample {i}: {'FAST' if has_fast else 'DEEP' if has_deep else 'UNKNOWN'} "
                    f"({len(text)} chars)")

    return data


# ═══════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════

def run_sft(
    base_model_name: str,
    sft_data: List[Dict[str, str]],
    output_dir: str,
    epochs: int = 2,
    learning_rate: float = 2e-5,
    lora_r: int = 16,
    lora_alpha: int = 32,
    weight_decay: float = 0.01,
    max_length: int = 1024,
    batch_size: int = 4,
    grad_accum: int = 8,
    warmup_ratio: float = 0.05,
) -> str:
    """Run SFT training with locked hyperparameter bounds.

    Returns path to the merged final model.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load base model ──
    logger.info(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # ── Register METIS special tokens ──
    tokenizer, model = register_metis_special_tokens(tokenizer, model)
    verify_metis_tokens(tokenizer)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── LoRA config — surgical injection ──
    # CRITICAL: modules_to_save trains embed_tokens and lm_head with full gradients.
    # Without this, the 10 newly added METIS special tokens (e.g. [COGNITIVE_STATE: FAST])
    # have randomly initialized embeddings that LoRA adapters cannot reach,
    # making it physically impossible for the model to output them.
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "gate_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Gradient checkpointing requires input embeddings to have grad
    model.enable_input_require_grads()

    # ── Tokenize with apply_chat_template (CRITICAL for train/inference parity) ──
    # v1 FAILURE: raw text concatenation → model never saw chat template during training
    # v2 FAILURE: separate prompt/response tokenization → subtle boundary misalignment
    # v3 FIX: use apply_chat_template with FULL messages list (system+user+assistant)
    #         then mask prompt tokens to compute loss only on assistant response.
    pad_id = tokenizer.pad_token_id
    all_input_ids: List[List[int]] = []
    all_labels: List[List[int]] = []
    all_attention_mask: List[List[int]] = []

    for idx, item in enumerate(sft_data):
        if "system" in item and "user" in item and "assistant" in item:
            # ── Step 1: Tokenize prompt (system + user) to get prompt length ──
            prompt_messages = [
                {"role": "system", "content": item["system"]},
                {"role": "user", "content": item["user"]},
            ]
            prompt_ids = tokenizer.apply_chat_template(
                prompt_messages, tokenize=True, add_generation_prompt=True,
            )
            prompt_len = len(prompt_ids)

            # ── Step 2: Tokenize full conversation (system + user + assistant) ──
            full_messages = [
                {"role": "system", "content": item["system"]},
                {"role": "user", "content": item["user"]},
                {"role": "assistant", "content": item["assistant"]},
            ]
            full_ids = tokenizer.apply_chat_template(
                full_messages, tokenize=True, add_generation_prompt=False,
            )

            # ── Step 3: Truncate to max_length ──
            if len(full_ids) > max_length:
                full_ids = full_ids[:max_length]
                prompt_len = min(prompt_len, max_length)

            # ── Step 4: Build labels — mask prompt with -100 ──
            labels = [-100] * prompt_len + full_ids[prompt_len:]
            assert len(labels) == len(full_ids)

            # ── Step 5: Pad to max_length ──
            seq_len = len(full_ids)
            pad_len = max_length - seq_len
            input_ids = full_ids + [pad_id] * pad_len
            labels = labels + [-100] * pad_len
            attention_mask = [1] * seq_len + [0] * pad_len

        else:
            # Legacy format fallback (raw text)
            encoded = tokenizer(
                item["text"], truncation=True, max_length=max_length,
                padding="max_length",
            )
            input_ids = encoded["input_ids"]
            labels = input_ids.copy()
            attention_mask = encoded["attention_mask"]

        all_input_ids.append(input_ids)
        all_labels.append(labels)
        all_attention_mask.append(attention_mask)

    tokenized = Dataset.from_dict({
        "input_ids": all_input_ids,
        "labels": all_labels,
        "attention_mask": all_attention_mask,
    })
    logger.info(f"Tokenized {len(tokenized)} samples (max_length={max_length}, chat template applied)")

    # ── Verify tokenization on sample 0 ──
    s_ids = all_input_ids[0]
    s_labels = all_labels[0]
    n_masked = sum(1 for l in s_labels if l == -100)
    n_train = sum(1 for l in s_labels if l != -100)
    # Find first trainable token
    first_train_idx = next((i for i, l in enumerate(s_labels) if l != -100), -1)
    logger.info(f"  Sample 0: {n_masked} prompt+pad tokens masked, {n_train} assistant tokens trainable")
    logger.info(f"  Prompt (first 50 tokens): {tokenizer.decode(s_ids[:50])}...")
    if first_train_idx >= 0:
        train_preview = tokenizer.decode(s_ids[first_train_idx:first_train_idx+30])
        logger.info(f"  Assistant starts at token {first_train_idx}: {train_preview}...")

    # ── VRAM auto-tune ──
    # modules_to_save=[embed_tokens, lm_head] adds ~13GB optimizer states for 1.1B params.
    # GB10 unified memory: must be conservative to avoid silent OOM kill.
    try:
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        free_gb = free_mem / (1024**3)
        logger.info(f"GPU VRAM: {free_gb:.1f}GB free (after model load)")
        # Estimate optimizer overhead: 1.1B params × 12 bytes (param + 2 Adam states) ≈ 13GB
        # Need at least 25GB headroom for optimizer + activations + gradients
        if free_gb < 60:
            batch_size = max(1, batch_size // 2)
            grad_accum = max(grad_accum, 32 // batch_size)
            logger.warning(f"VRAM constrained — adjusted batch={batch_size}, grad_accum={grad_accum}")
        if free_gb < 30:
            batch_size = 1
            grad_accum = 32
            logger.warning(f"VRAM critically low — forced batch=1, grad_accum=32")
    except Exception:
        pass

    # ── Optimizer selection ──
    try:
        import bitsandbytes  # noqa: F401
        optim = "paged_adamw_8bit"
    except ImportError:
        optim = "adamw_torch"
        logger.info("bitsandbytes not found, using adamw_torch")

    # ── Training args — locked bounds ──
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        logging_steps=5,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        optim=optim,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    logger.info("=" * 60)
    logger.info("SFT TRAINING CONFIG (LOCKED BOUNDS)")
    logger.info("=" * 60)
    logger.info(f"  LR:             {learning_rate} (cosine, {warmup_ratio:.0%} warmup)")
    logger.info(f"  Epochs:         {epochs}")
    logger.info(f"  Weight decay:   {weight_decay}")
    logger.info(f"  LoRA:           r={lora_r}, α={lora_alpha}")
    logger.info(f"  Targets:        q_proj, v_proj, gate_proj, down_proj")
    logger.info(f"  modules_to_save: embed_tokens, lm_head (CRITICAL for new tokens)")
    logger.info(f"  Batch:          {batch_size} × {grad_accum} = {batch_size * grad_accum} effective")
    logger.info(f"  Max length:     {max_length}")
    logger.info(f"  GC:             ON")
    logger.info(f"  Samples:        {len(tokenized)}")
    total_steps = (len(tokenized) // (batch_size * grad_accum)) * epochs
    logger.info(f"  Est. steps:     ~{total_steps}")

    # ── Train ──
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        processing_class=tokenizer,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    logger.info(f"Training complete in {elapsed/60:.1f} minutes")

    # Save LoRA adapter
    adapter_path = os.path.join(output_dir, "sft_adapter")
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info(f"LoRA adapter saved to {adapter_path}")

    # ── Merge LoRA → full model ──
    logger.info("Merging LoRA weights into base model...")
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    merged_model = model.merge_and_unload()

    # Clean PEFT residuals
    for attr in ("peft_config", "peft_type", "active_adapter", "active_adapters"):
        if hasattr(merged_model, attr):
            try:
                delattr(merged_model, attr)
            except Exception:
                pass

    merged_path = os.path.join(output_dir, "metis_sft_cognitive")
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    logger.info(f"Merged model saved to {merged_path}")

    return merged_path


# ═══════════════════════════════════════════════════════════
# Inference Check
# ═══════════════════════════════════════════════════════════

INFERENCE_TESTS = [
    # FAST-expected: simple factual / linguistic
    {"query": "What is the capital of France?", "expected": "FAST"},
    {"query": "Translate 'good morning' to Spanish.", "expected": "FAST"},
    {"query": "What color do you get when you mix red and blue?", "expected": "FAST"},
    # DEEP-expected: multi-step math / reasoning
    {"query": "A store sells apples for $2 each. If John buys 15 apples and pays with a $50 bill, how much change does he get?", "expected": "DEEP"},
    {"query": "A train travels at 60 km/h for 2.5 hours, then at 80 km/h for 1.5 hours. What is the total distance?", "expected": "DEEP"},
    {"query": "Sarah has 3 times as many stickers as Tom. Together they have 48 stickers. How many stickers does Sarah have?", "expected": "DEEP"},
]

METIS_SYSTEM_PROMPT = (
    "You are METIS, an AI with a dynamic cognitive routing layer. "
    "Analyze the complexity of the user's request and allocate compute accordingly."
)


def run_inference_check(model_path: str) -> None:
    """Run inference on test queries to verify FAST/DEEP routing."""
    logger.info("=" * 60)
    logger.info("INFERENCE CHECK — Cognitive Routing Verification")
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    correct = 0
    total = len(INFERENCE_TESTS)

    for test in INFERENCE_TESTS:
        messages = [
            {"role": "system", "content": METIS_SYSTEM_PROMPT},
            {"role": "user", "content": test["query"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # CRITICAL: skip_special_tokens=False — otherwise cognitive tokens get stripped!
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated = tokenizer.decode(generated_ids, skip_special_tokens=False)

        # Detect routing via token IDs (most reliable) or text fallback
        generated_id_list = generated_ids.tolist()
        fast_id = tokenizer.convert_tokens_to_ids("[COGNITIVE_STATE: FAST]")
        deep_id = tokenizer.convert_tokens_to_ids("[COGNITIVE_STATE: DEEP]")
        if fast_id in generated_id_list:
            detected = "FAST"
        elif deep_id in generated_id_list:
            detected = "DEEP"
        elif "[COGNITIVE_STATE: FAST]" in generated:
            detected = "FAST"
        elif "[COGNITIVE_STATE: DEEP]" in generated:
            detected = "DEEP"
        else:
            detected = "NONE"

        match = "✅" if detected == test["expected"] else "❌"
        if detected == test["expected"]:
            correct += 1

        logger.info(f"\n  {match} Query: {test['query'][:60]}...")
        logger.info(f"     Expected: {test['expected']} | Detected: {detected}")
        logger.info(f"     Output:   {generated[:150]}...")

    accuracy = correct / total * 100
    logger.info(f"\n{'=' * 60}")
    logger.info(f"ROUTING ACCURACY: {correct}/{total} ({accuracy:.0f}%)")
    logger.info(f"{'=' * 60}")

    if accuracy >= 80:
        logger.info("✅ Cognitive routing is functional — ready for Phase 2")
    else:
        logger.warning("⚠️  Routing accuracy below 80% — may need more data or epochs")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="METIS SFT Seed Training")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base model to fine-tune")
    parser.add_argument("--sft-data", default="data/metis_sft_seed.jsonl",
                        help="Path to SFT seed data (JSONL with system/user/assistant fields)")
    parser.add_argument("--output", default="experiment_output_sft_cognitive",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Training epochs (2-3, strict)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Peak learning rate")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size (lower for modules_to_save VRAM overhead)")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Max sequence length")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip inference check after training")
    args = parser.parse_args()

    # ── Phase 1: Train ──
    sft_data = load_sft_data(args.sft_data)
    merged_path = run_sft(
        base_model_name=args.base_model,
        sft_data=sft_data,
        output_dir=args.output,
        epochs=args.epochs,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    # ── Phase 2: Inference Check ──
    if not args.skip_inference:
        # Release training model, reload for inference
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        run_inference_check(merged_path)

    logger.info("\nDone. Next step: use this model for Reject Sampling Engine (Phase 2).")


if __name__ == "__main__":
    main()
