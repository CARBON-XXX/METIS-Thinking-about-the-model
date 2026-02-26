#!/usr/bin/env python3
"""
Resume experiment: Phase 2 (DPO) + Phase 3 (Eval) using existing Phase 1 data.
Uses multi-pair matching logic for more DPO training data.

Usage:
    python resume_experiment.py
"""
import json
import os
import sys
import gc
import time
import math
import random
import logging

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Import from run_experiment ──
sys.path.insert(0, os.path.dirname(__file__))
from run_experiment import (
    ExperimentConfig,
    EvalMetrics,
    _build_metis_pairs,
    _build_random_pairs,
    _evaluate_model,
    _format_chat,
    phase4_report,
    EVAL_PROMPTS,
    C,
)


def main():
    config = ExperimentConfig(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        device="cuda",
        output_dir="experiment_v4b",
        n_train_prompts=300,
        n_samples_per_prompt=8,
        max_new_tokens=200,
        dpo_epochs=3,
        # ── OOM fix: reduce batch size, shorter max_length ──
        dpo_batch_size=1,
        dpo_max_length=384,
        dpo_gradient_accumulation=8,
        eval_max_tokens=200,
    )

    # Load Phase 1 data from original experiment
    data_path = os.path.join("experiment_full", "phase1_scored_data.json")
    os.makedirs(config.output_dir, exist_ok=True)
    metis_path = os.path.join(config.output_dir, "metis_dpo")
    random_path = os.path.join(config.output_dir, "random_dpo")

    # ── Load Phase 1 data ──
    logger.info(f"Loading Phase 1 data from {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        scored_data = json.load(f)
    logger.info(f"Loaded {len(scored_data)} samples")

    # ── Load model ──
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(config.device)
    model.eval()

    # ══════════════════════════════════════════
    # Phase 2a: METIS DPO (multi-pair matching)
    # ══════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 2a: METIS DPO Training (multi-pair)")
    logger.info("=" * 60)

    metis_pairs = _build_metis_pairs(scored_data)
    logger.info(f"METIS pairs: {len(metis_pairs)} (multi-pair matching)")

    if len(metis_pairs) >= 1:
        gc.collect()
        torch.cuda.empty_cache()

        from peft import LoraConfig, get_peft_model, TaskType
        from trl import DPOTrainer, DPOConfig
        from datasets import Dataset

        dataset = Dataset.from_list(metis_pairs)
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        train_model = get_peft_model(model, lora_config)
        train_model.print_trainable_parameters()

        training_args = DPOConfig(
            output_dir=metis_path,
            num_train_epochs=config.dpo_epochs,
            per_device_train_batch_size=config.dpo_batch_size,
            gradient_accumulation_steps=config.dpo_gradient_accumulation,
            learning_rate=config.dpo_learning_rate,
            beta=config.dpo_beta,
            max_length=config.dpo_max_length,
            max_prompt_length=config.dpo_max_length // 2,
            logging_steps=1,
            save_strategy="epoch",
            remove_unused_columns=False,
            fp16=True,
            gradient_checkpointing=config.gradient_checkpointing,
            report_to="none",
        )
        trainer = DPOTrainer(
            model=train_model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        trainer.train()
        trainer.save_model(metis_path)
        logger.info(f"Saved METIS DPO checkpoint to {metis_path}")

        del trainer
        train_model.unload()
        del train_model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        logger.warning("No METIS pairs — skipping")
        os.makedirs(metis_path, exist_ok=True)

    # ══════════════════════════════════════════
    # Phase 2b: Random DPO
    # ══════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 2b: Random DPO Training")
    logger.info("=" * 60)

    random_pairs = _build_random_pairs(scored_data)
    logger.info(f"Random pairs: {len(random_pairs)}")

    if len(random_pairs) >= 1:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        from peft import LoraConfig, get_peft_model, TaskType
        from trl import DPOTrainer, DPOConfig
        from datasets import Dataset

        dataset = Dataset.from_list(random_pairs)

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )

        train_model = get_peft_model(model, lora_config)
        train_model.print_trainable_parameters()

        training_args = DPOConfig(
            output_dir=random_path,
            num_train_epochs=config.dpo_epochs,
            per_device_train_batch_size=config.dpo_batch_size,
            gradient_accumulation_steps=config.dpo_gradient_accumulation,
            learning_rate=config.dpo_learning_rate,
            beta=config.dpo_beta,
            max_length=config.dpo_max_length,
            max_prompt_length=config.dpo_max_length // 2,
            logging_steps=1,
            save_strategy="epoch",
            remove_unused_columns=False,
            fp16=True,
            gradient_checkpointing=config.gradient_checkpointing,
            report_to="none",
        )

        trainer = DPOTrainer(
            model=train_model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        trainer.train()
        trainer.save_model(random_path)
        logger.info(f"Saved Random DPO checkpoint to {random_path}")

        del trainer
        train_model.unload()
        del train_model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        logger.warning("No random pairs — skipping")
        os.makedirs(random_path, exist_ok=True)

    # ══════════════════════════════════════════
    # Phase 3: Evaluation
    # ══════════════════════════════════════════
    logger.info("=" * 60)
    logger.info(f"PHASE 3: Evaluation ({config.n_eval_prompts} held-out prompts)")
    logger.info("=" * 60)

    from peft import PeftModel
    from metis.training.rewards import CognitiveRewardComputer

    eval_prompts = EVAL_PROMPTS[:config.n_eval_prompts]
    reward_computer = CognitiveRewardComputer()

    # Reload clean base model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    device = config.device
    logger.info("Reloading clean base model for evaluation...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)
    base_model.eval()

    # ── Evaluate Base Model ──
    logger.info("Evaluating: Base Model")
    base_metrics = _evaluate_model(
        config, base_model, tokenizer, eval_prompts, reward_computer, "Base"
    )

    # ── Evaluate METIS DPO ──
    if os.path.exists(os.path.join(metis_path, "adapter_config.json")):
        logger.info("Evaluating: METIS DPO")
        metis_model = PeftModel.from_pretrained(base_model, metis_path)
        metis_model.eval()
        metis_metrics = _evaluate_model(
            config, metis_model, tokenizer, eval_prompts, reward_computer, "METIS-DPO"
        )
        del metis_model, base_model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(device)
        base_model.eval()
    else:
        logger.warning("No METIS DPO checkpoint — using base as fallback")
        metis_metrics = base_metrics

    # ── Evaluate Random DPO ──
    if os.path.exists(os.path.join(random_path, "adapter_config.json")):
        logger.info("Evaluating: Random DPO")
        random_model = PeftModel.from_pretrained(base_model, random_path)
        random_model.eval()
        random_metrics = _evaluate_model(
            config, random_model, tokenizer, eval_prompts, reward_computer, "Random-DPO"
        )
        del random_model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        logger.warning("No Random DPO checkpoint — using base as fallback")
        random_metrics = base_metrics

    # ══════════════════════════════════════════
    # Phase 4: Report
    # ══════════════════════════════════════════
    phase4_report(config, base_metrics, metis_metrics, random_metrics)

    # Save metrics
    import json as _json
    metrics_path = os.path.join(config.output_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        _json.dump({
            "base": base_metrics.to_dict(),
            "metis": metis_metrics.to_dict(),
            "random": random_metrics.to_dict(),
        }, f, indent=2)
    logger.info(f"Saved eval metrics to {metrics_path}")


if __name__ == "__main__":
    main()
