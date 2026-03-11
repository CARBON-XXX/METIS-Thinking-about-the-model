#!/usr/bin/env python3
"""
METIS Dream Training — Headless GRPO Training Script

Phase 18+23: Lightweight, pure-CLI training script for the Dreaming Daemon.
Zero hardcoded paths. Stripped of all experimental data export/visualization.

Features:
  - Mandatory CLI args: --dataset, --base-model, --output-dir
  - Optional tuning: --lr, --epochs, --batch-size, --lora-r, --max-completion-length
  - OOM auto-fallback: catch CUDA OOM → halve batch size → retry (max 2 retries)
  - Crash recovery: detect existing checkpoints → resume_from_checkpoint=True
  - Clean TRL GRPOTrainer loop only — no charts, no export
  - LoRA merge on success
  - Exit code 0 on success, 1 on failure
  - Phase 23: --anchor-model for KL-divergence anchoring against pristine base
    (prevents compounding distribution drift across nightly training cycles)

Usage:
    python tools/run_dream_training.py \\
        --dataset /path/to/gaps.jsonl \\
        --base-model /path/to/model \\
        --output-dir /path/to/output \\
        [--anchor-model Qwen/Qwen2.5-7B-Instruct] \\
        [--lr 1e-6] [--epochs 1] [--batch-size 4] [--lora-r 32]
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

logger = logging.getLogger("metis.dream_training")

# ── Seed for reproducibility ──
SEED = 42


# ═══════════════════════════════════════════════════════════
# Reward Functions (reused pattern from run_grpo_training.py)
# ═══════════════════════════════════════════════════════════

def accuracy_reward(completions: List[List[Dict[str, str]]], **kwargs: Any) -> List[float]:
    """Reward based on answer quality.

    Simple heuristic: reward non-empty, substantive responses.
    Penalize empty, very short, or obviously evasive answers.
    """
    rewards: List[float] = []
    for completion_set in completions:
        text = completion_set[-1].get("content", "") if completion_set else ""
        text = text.strip()

        if not text or len(text) < 10:
            rewards.append(-1.0)
        elif any(
            phrase in text.lower()
            for phrase in ["i don't know", "i cannot", "i'm not sure", "i apologize"]
        ):
            rewards.append(-0.5)
        elif len(text) > 50:
            rewards.append(2.0)
        else:
            rewards.append(1.0)

    return rewards


def verbosity_penalty(completions: List[List[Dict[str, str]]], **kwargs: Any) -> List[float]:
    """Penalize excessive verbosity, reward concise + thoughtful responses.

    - Short (<20 tokens): neutral (0.0)
    - Medium (20-500 tokens): slight bonus (+0.5)
    - Long (>500 tokens): progressive penalty
    - Thinking tags present: bonus (+0.5)
    """
    rewards: List[float] = []
    for completion_set in completions:
        text = completion_set[-1].get("content", "") if completion_set else ""
        n_tokens = len(text.split())

        if n_tokens < 20:
            r = 0.0
        elif n_tokens <= 500:
            r = 0.5
        else:
            r = -0.1 * (n_tokens - 500) / 100.0

        # Bonus for thinking tags (cognitive depth)
        if "<thinking>" in text:
            r += 0.5

        rewards.append(r)

    return rewards


# ═══════════════════════════════════════════════════════════
# Dataset Loading
# ═══════════════════════════════════════════════════════════

def load_dream_dataset(dataset_path: str) -> Dataset:
    """Load JSONL dataset from Dreaming Daemon.

    Expected format per line:
        {"prompt": "...", "_meta": {...}}

    Returns HuggingFace Dataset with 'prompt' column.
    """
    records: List[Dict[str, str]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                prompt = obj.get("prompt", "")
                if prompt:
                    records.append({"prompt": prompt})
            except json.JSONDecodeError:
                continue

    if not records:
        logger.error(f"No valid prompts found in {dataset_path}")
        sys.exit(1)

    logger.info(f"Loaded {len(records)} prompts from {dataset_path}")
    return Dataset.from_list(records)


# ═══════════════════════════════════════════════════════════
# Checkpoint Detection (Crash Recovery)
# ═══════════════════════════════════════════════════════════

def find_latest_checkpoint(output_dir: str) -> str | None:
    """Find the latest checkpoint in output_dir for resume."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    checkpoints = sorted(
        [
            d for d in output_path.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ],
        key=lambda d: int(re.search(r"checkpoint-(\d+)", d.name).group(1))
        if re.search(r"checkpoint-(\d+)", d.name) else 0,
    )

    if checkpoints:
        latest = str(checkpoints[-1])
        logger.info(f"Found checkpoint for resume: {latest}")
        return latest
    return None


# ═══════════════════════════════════════════════════════════
# Main Training Function
# ═══════════════════════════════════════════════════════════

def train(args: argparse.Namespace) -> bool:
    """Execute GRPO training. Returns True on success, False on failure."""
    logger.info("=" * 60)
    logger.info("  METIS Dream Training (Phase 18)")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Base model: {args.base_model}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info("=" * 60)

    # Load dataset
    dataset = load_dream_dataset(args.dataset)

    # Dry-run mode: validate dataset then exit
    if getattr(args, 'dry_run', False):
        logger.info(f"Dry run successful: {len(dataset)} prompts validated.")
        return True

    # LoRA config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        modules_to_save=["embed_tokens", "lm_head"],
    )

    # Check for crash recovery
    resume_from = find_latest_checkpoint(args.output_dir)

    # OOM auto-fallback: try with current batch size, halve on OOM
    batch_size = args.batch_size
    max_retries = 2

    for attempt in range(max_retries + 1):
        try:
            return _run_training(
                args, dataset, peft_config, batch_size, resume_from,
            )
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            if attempt < max_retries:
                old_bs = batch_size
                batch_size = max(1, batch_size // 2)
                logger.warning(
                    f"OOM at batch_size={old_bs}. "
                    f"Retrying with batch_size={batch_size} "
                    f"(attempt {attempt + 2}/{max_retries + 1})"
                )
            else:
                logger.error(
                    f"OOM after {max_retries + 1} attempts. "
                    f"Final batch_size={batch_size}. Aborting."
                )
                return False
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            return False

    return False


def _run_training(
    args: argparse.Namespace,
    dataset: Dataset,
    peft_config: LoraConfig,
    batch_size: int,
    resume_from: str | None,
) -> bool:
    """Inner training loop. Separated for OOM retry logic.

    Phase 23 KL-Anchor Hardening:
        If --anchor-model is provided, it is loaded as a FROZEN ref_model
        for the KL-divergence penalty in GRPO.  This ensures the mutating
        daily checkpoint (--base-model) is always regularized against the
        PRISTINE pre-trained distribution, preventing compounding drift:
            D_KL(π_θ || π_anchor)  ← constant anchor
        Without this, the default ref_model = base_model, meaning each day
        the reference ITSELF drifts, erasing the regularization guarantee.
    """
    # Adjust num_generations to be divisible by batch_size
    num_generations = max(4, min(16, batch_size * 4))
    gen_batch_size = num_generations  # Must equal num_generations

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=max(1, 8 // batch_size),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_generations=num_generations,
        max_completion_length=args.max_completion_length,
        generation_batch_size=gen_batch_size,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        beta=0.04,
        logging_steps=5,
        log_completions=True,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        reward_weights=[1.0, 1.0],
        seed=SEED,
        report_to="none",
        remove_unused_columns=False,
        temperature=0.8,
        torch_empty_cache_steps=10,
    )

    logger.info(
        f"Training config: batch={batch_size}, "
        f"grad_accum={training_args.gradient_accumulation_steps}, "
        f"num_gen={num_generations}, lr={args.lr}"
    )

    # Load model
    logger.info(f"Loading model: {args.base_model}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        logger.info("  Using Flash Attention 2")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        logger.info("  Flash Attention 2 unavailable, using SDPA")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Phase 23: KL-Anchor Model (prevents compounding distribution drift) ──
    ref_model = None
    anchor_model_id = getattr(args, "anchor_model", None)
    if anchor_model_id:
        logger.info(f"Loading KL-anchor model: {anchor_model_id}")
        try:
            ref_model = AutoModelForCausalLM.from_pretrained(
                anchor_model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                trust_remote_code=True,
            )
            ref_model.eval()
            for param in ref_model.parameters():
                param.requires_grad = False
            logger.info(
                f"  KL-anchor loaded and frozen: {anchor_model_id} "
                f"(D_KL penalty anchored to pristine base)"
            )
        except Exception as e:
            logger.warning(
                f"  Failed to load anchor model: {e}. "
                f"Falling back to default ref_model (base_model copy)."
            )
            ref_model = None

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        reward_funcs=[accuracy_reward, verbosity_penalty],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Train (with optional resume)
    logger.info("Starting GRPO training...")
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        train_result = trainer.train(resume_from_checkpoint=resume_from)
    else:
        train_result = trainer.train()

    logger.info(f"Training complete: {train_result.metrics}")

    # Save
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Merge LoRA
    logger.info("Merging LoRA weights...")
    merged_dir = os.path.join(args.output_dir, "merged")
    try:
        from peft import PeftModel

        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()

        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        peft_model = PeftModel.from_pretrained(base_model, args.output_dir)
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        logger.info(f"  Merged model saved to {merged_dir}")
    except Exception as e:
        logger.error(f"  Merge failed: {e}. LoRA adapter at {args.output_dir}")

    logger.info("=" * 60)
    logger.info("  Dream Training COMPLETE")
    logger.info(f"  Output: {args.output_dir}")
    logger.info(f"  Merged: {merged_dir}")
    logger.info("=" * 60)
    return True


# ═══════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="METIS Dream Training — Headless GRPO for Dreaming Daemon"
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to JSONL dataset (one {\"prompt\": ...} per line)",
    )
    parser.add_argument(
        "--base-model", required=True,
        help="Path to base model for training",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for checkpoints and merged model",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-6,
        help="Learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Number of training epochs (default: 1)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Per-device batch size (default: 4, auto-halved on OOM)",
    )
    parser.add_argument(
        "--lora-r", type=int, default=32,
        help="LoRA rank (default: 32)",
    )
    parser.add_argument(
        "--max-completion-length", type=int, default=1024,
        help="Max completion length for GRPO generation (default: 1024)",
    )
    parser.add_argument(
        "--anchor-model", type=str, default=None,
        help="Pristine pre-trained model for KL-divergence anchor (Phase 23). "
             "Prevents compounding distribution drift across nightly cycles. "
             "Example: Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Validate dataset and exit without training (for daemon testing)",
    )
    parser.add_argument(
        "--evolutionary", action="store_true", default=False,
        help="Signal that this training run is part of an evolutionary loop. "
             "Requires --anchor-model (Phase 23.5 rigid constraint).",
    )

    args = parser.parse_args()

    # ── Phase 23.5: Evolutionary Anchor Binding ──
    if args.evolutionary and args.anchor_model is None:
        print(
            "ERROR: Evolutionary mode active: --anchor-model is strictly required "
            "to prevent catastrophic compounding drift.",
            file=sys.stderr,
        )
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    success = train(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
