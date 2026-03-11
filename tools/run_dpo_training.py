#!/usr/bin/env python3
"""
METIS Phase 4 — DPO Training (Direct Preference Optimization)
==============================================================

Loads the SFT v4 merged model as π_ref, attaches LoRA adapters as π_θ,
and trains on 952 high-contrast DPO pairs from Phase 3.

Architecture:
  - Base/Ref model: experiment_output_sft_cognitive_v4/metis_sft_cognitive
  - PEFT LoRA with modules_to_save for METIS special token embeddings
  - DPO β=0.1, lr=5e-6, 1 epoch, cosine schedule with 10% warmup
  - Output: experiment_output_dpo_final/metis_dpo_cognitive

Stability constraints:
  - 1 epoch max on 952 samples to prevent entropy collapse
  - Extremely low LR (5e-6) to preserve SFT format quality
  - Effective batch size 16 via gradient accumulation
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────

DEFAULT_SFT_MODEL_PATH = "experiment_output_sft_cognitive_v4/metis_sft_cognitive"
DEFAULT_DPO_DATA_PATH = "data/metis_dpo_pairs.jsonl"
DEFAULT_OUTPUT_DIR = "experiment_output_dpo_final"
DEFAULT_MERGED_PATH = "experiment_output_dpo_final/metis_dpo_cognitive"


# ─────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────

def load_dpo_dataset(data_path: str) -> Dataset:
    """Load DPO pairs from JSONL and return HF Dataset.

    Expected format per line:
      {"prompt": "...", "chosen": "...", "rejected": "...", "_meta": {...}}
    """
    logger.info(f"Loading DPO dataset from {data_path}...")

    records: List[Dict[str, str]] = []
    with open(data_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            records.append({
                "prompt": item["prompt"],
                "chosen": item["chosen"],
                "rejected": item["rejected"],
            })

    ds = Dataset.from_list(records)
    logger.info(f"  Loaded {len(ds)} DPO pairs")
    logger.info(f"  Columns: {ds.column_names}")
    return ds


# ─────────────────────────────────────────────────────
# Model & Tokenizer Loading
# ─────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_path: str,
) -> tuple:
    """Load the SFT v4 merged model as base for DPO.

    This model serves as both π_ref (frozen base) and the starting
    point for π_θ (LoRA adapters).
    """
    logger.info(f"Loading SFT model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # Log VRAM
    try:
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        logger.info(f"  VRAM after model load: {free_mem / (1024**3):.1f}GB free "
                    f"/ {total_mem / (1024**3):.1f}GB total")
    except Exception:
        pass

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model: {n_params / 1e9:.2f}B params, dtype=bfloat16")
    return model, tokenizer


# ─────────────────────────────────────────────────────
# LoRA Configuration
# ─────────────────────────────────────────────────────

def build_lora_config() -> LoraConfig:
    """Build PEFT LoRA config for DPO.

    modules_to_save includes embed_tokens and lm_head to continue
    optimizing the METIS special token embeddings learned during SFT.
    """
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "gate_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
        bias="none",
    )
    logger.info(f"  LoRA config: r={config.r}, alpha={config.lora_alpha}, "
                f"dropout={config.lora_dropout}")
    logger.info(f"  Target modules: {config.target_modules}")
    logger.info(f"  Modules to save: {config.modules_to_save}")
    return config


# ─────────────────────────────────────────────────────
# DPO Training Configuration
# ─────────────────────────────────────────────────────

def build_dpo_config(
    output_dir: str,
    batch_size: int = 2,
    grad_accum: int = 8,
    learning_rate: float = 5e-6,
    beta: float = 0.1,
    num_epochs: int = 1,
    max_length: int = 1536,
    max_prompt_length: int = 512,
) -> DPOConfig:
    """Build DPO training configuration with extreme stability bounds."""

    effective_bs = batch_size * grad_accum
    warmup_ratio = 0.1

    config = DPOConfig(
        output_dir=output_dir,
        beta=beta,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=max_length,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        report_to="none",
        seed=42,
    )

    logger.info(f"  DPO config:")
    logger.info(f"    β (KL penalty):      {beta}")
    logger.info(f"    Learning rate:        {learning_rate}")
    logger.info(f"    Scheduler:            cosine, {warmup_ratio*100:.0f}% warmup")
    logger.info(f"    Epochs:               {num_epochs}")
    logger.info(f"    Batch size:           {batch_size} × {grad_accum} = {effective_bs}")
    logger.info(f"    Max length:           {max_length}")
    logger.info(f"    Max prompt length:    {max_prompt_length}")
    logger.info(f"    Gradient checkpoint:  True")

    return config


# ─────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────

def run_dpo_training(
    model_path: str = DEFAULT_SFT_MODEL_PATH,
    data_path: str = DEFAULT_DPO_DATA_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    merged_path: str = DEFAULT_MERGED_PATH,
    batch_size: int = 2,
    grad_accum: int = 8,
    learning_rate: float = 5e-6,
    beta: float = 0.1,
    num_epochs: int = 1,
    max_length: int = 1536,
    max_prompt_length: int = 512,
) -> None:
    """Execute full DPO training pipeline."""

    logger.info("=" * 60)
    logger.info("METIS Phase 4: DPO Training")
    logger.info("=" * 60)

    # ── Step 1: Load dataset ──
    dataset = load_dpo_dataset(data_path)

    # ── Step 2: Load model ──
    model, tokenizer = load_model_and_tokenizer(model_path)

    # ── Step 3: Build configs ──
    logger.info("Building training configuration...")
    lora_config = build_lora_config()
    dpo_config = build_dpo_config(
        output_dir=output_dir,
        batch_size=batch_size,
        grad_accum=grad_accum,
        learning_rate=learning_rate,
        beta=beta,
        num_epochs=num_epochs,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
    )

    # ── Step 4: Initialize DPO Trainer ──
    logger.info("Initializing DPOTrainer...")
    logger.info("  π_ref = frozen base weights (SFT v4)")
    logger.info("  π_θ   = LoRA adapters on SFT v4")

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Log trainable params
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    logger.info(f"  Trainable: {trainable/1e6:.1f}M / {total/1e9:.2f}B "
                f"({trainable/total*100:.2f}%)")

    # Log VRAM after trainer init
    try:
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        logger.info(f"  VRAM after trainer init: {free_mem / (1024**3):.1f}GB free "
                    f"/ {total_mem / (1024**3):.1f}GB total")
    except Exception:
        pass

    # ── Step 5: Train ──
    logger.info("Starting DPO training...")
    train_result = trainer.train()

    # ── Step 6: Log metrics ──
    metrics = train_result.metrics
    logger.info("Training complete. Metrics:")
    for k, v in sorted(metrics.items()):
        logger.info(f"  {k}: {v}")

    # Save adapter checkpoint
    trainer.save_model(output_dir)
    logger.info(f"  LoRA adapter saved to {output_dir}")

    # ── Step 7: Merge and save ──
    logger.info(f"Merging LoRA adapters into base model → {merged_path}...")

    # Need to merge on CPU to avoid VRAM issues
    merged_model = trainer.model.merge_and_unload()

    merged_model.save_pretrained(
        merged_path,
        safe_serialization=True,
    )
    tokenizer.save_pretrained(merged_path)
    logger.info(f"  Merged model saved to {merged_path}")

    # ── Step 8: Quick inference check ──
    logger.info("Running quick inference check...")
    try:
        test_messages = [
            {"role": "system", "content": "You are METIS, an AI with a dynamic cognitive routing layer. Analyze the complexity of the user's request and allocate compute accordingly."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        input_ids = tokenizer.apply_chat_template(
            test_messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        ).to(merged_model.device)

        with torch.no_grad():
            out = merged_model.generate(
                input_ids,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=False)
        if tokenizer.eos_token:
            response = response.split(tokenizer.eos_token)[0]
        logger.info(f"  Test prompt: 'What is 2+2?'")
        logger.info(f"  Response: {response[:300]}")

        # Check for cognitive tags
        has_fast = "[COGNITIVE_STATE: FAST]" in response
        has_deep = "[COGNITIVE_STATE: DEEP]" in response
        has_thinking = "<thinking>" in response
        logger.info(f"  Tags: FAST={has_fast}, DEEP={has_deep}, <thinking>={has_thinking}")
    except Exception as e:
        logger.warning(f"  Inference check failed: {e}")

    # ── Final Report ──
    logger.info("=" * 60)
    logger.info("PHASE 4 COMPLETE: DPO Training")
    logger.info("=" * 60)
    logger.info(f"  Dataset:        {len(dataset)} DPO pairs")
    logger.info(f"  Training loss:  {metrics.get('train_loss', 'N/A')}")
    logger.info(f"  Adapter:        {output_dir}")
    logger.info(f"  Merged model:   {merged_path}")
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="METIS Phase 4: DPO Training"
    )
    parser.add_argument(
        "--model-path", type=str, default=DEFAULT_SFT_MODEL_PATH,
        help="Path to SFT v4 merged model (π_ref base)",
    )
    parser.add_argument(
        "--data-path", type=str, default=DEFAULT_DPO_DATA_PATH,
        help="Path to DPO pairs JSONL from Phase 3",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for adapter checkpoint",
    )
    parser.add_argument(
        "--merged-path", type=str, default=DEFAULT_MERGED_PATH,
        help="Output path for merged DPO model",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Per-device train batch size",
    )
    parser.add_argument(
        "--grad-accum", type=int, default=8,
        help="Gradient accumulation steps (effective_bs = batch_size × grad_accum)",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-6,
        help="Learning rate (extremely low for DPO stability)",
    )
    parser.add_argument(
        "--beta", type=float, default=0.1,
        help="DPO β (KL divergence penalty coefficient)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Number of training epochs (DO NOT exceed 1)",
    )
    args = parser.parse_args()

    run_dpo_training(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        merged_path=args.merged_path,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        num_epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
