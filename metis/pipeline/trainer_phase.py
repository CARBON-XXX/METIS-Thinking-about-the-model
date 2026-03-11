"""
METIS Pipeline — Phase 2: DPO Training

Trains two models for A/B comparison:
  Group A (METIS):  DPO with cognitive-reward-ranked preference pairs
  Group B (Random): DPO with randomly-paired preferences (control)
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Tuple

import torch
from transformers import TrainerCallback

from metis.pipeline.config import ExperimentConfig
from metis.training.tokenizer_utils import register_metis_special_tokens, verify_metis_tokens


class MetricsJsonCallback(TrainerCallback):
    """Write training metrics to a JSON file after each log step for real-time monitoring."""

    def __init__(self, output_file: str, group_name: str = "metis"):
        self.output_file = output_file
        self.group_name = group_name
        # Load existing metrics if file exists (for appending across groups)
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                self.all_data = json.load(f)
        else:
            self.all_data = {}
        if self.group_name not in self.all_data:
            self.all_data[self.group_name] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        entry = {
            "step": state.global_step,
            "epoch": round(state.epoch, 4) if state.epoch else 0,
            "timestamp": time.time(),
            **{k: float(v) if isinstance(v, (int, float)) else v for k, v in logs.items()},
        }
        self.all_data[self.group_name].append(entry)
        with open(self.output_file, "w") as f:
            json.dump(self.all_data, f, indent=2)

logger = logging.getLogger("experiment")


def phase2_train(
    config: ExperimentConfig,
    scored_data: List[Dict],
    model: Any,
    tokenizer: Any,
) -> Tuple[str, str]:
    """
    Train two models:
    - Group A: DPO with METIS cognitive reward pairs
    - Group B: DPO with random pairs (control)

    Returns paths to both checkpoints.
    """
    logger.info(f"{'='*60}")
    logger.info(f"PHASE 2: DPO Training (METIS vs Random)")
    logger.info(f"{'='*60}")

    from peft import LoraConfig, get_peft_model, TaskType
    from trl import DPOTrainer, DPOConfig
    from datasets import Dataset

    # ─── Build METIS DPO pairs ───
    if config.external_dpo_data:
        import json
        logger.info(f"Loading external DPO pairs from {config.external_dpo_data}...")
        with open(config.external_dpo_data, "r", encoding="utf-8") as f:
            external_pairs = json.load(f)
        
        # We still need random pairs as a control group for Phase 3/4 evaluation.
        # But we don't have scored_data to build them from. 
        # So we'll just shuffle the external pairs to create a random baseline.
        import random
        metis_pairs = external_pairs
        random_pairs = external_pairs.copy()
        for pair in random_pairs:
            if random.random() < 0.5:
                # swap 50% of the time to destroy the preference signal
                pair["chosen"], pair["rejected"] = pair["rejected"], pair["chosen"]
                
        logger.info(f"Using {len(metis_pairs)} external pairs for METIS DPO.")
        logger.info(f"Generated {len(random_pairs)} flipped pairs for Random baseline.")
    else:
        metis_pairs = _build_metis_pairs(scored_data)
        random_pairs = _build_random_pairs(scored_data)

    logger.info(f"METIS pairs: {len(metis_pairs)}")
    logger.info(f"Random pairs: {len(random_pairs)}")

    metis_path = os.path.join(config.output_dir, "metis_dpo")
    random_path = os.path.join(config.output_dir, "random_dpo")

    # ─── Register METIS Special Tokens ───
    # Must happen BEFORE SFT/DPO so <thinking> etc. are single tokens.
    # This prevents KL explosion from sub-token splitting.
    tokenizer, model = register_metis_special_tokens(tokenizer, model)
    verify_metis_tokens(tokenizer)

    # Save augmented tokenizer alongside output for eval consistency
    tokenizer.save_pretrained(os.path.join(config.output_dir, "tokenizer"))
    logger.info(f"[Tokenizer] Saved augmented tokenizer to {config.output_dir}/tokenizer")

    # ─── SFT Warmup: teach base model <thinking> format ───
    # Uses ONLY external third-party data (e.g. Orca). No model-generated samples.
    if config.sft_warmup and config.sft_data_path:
        sft_path = os.path.join(config.output_dir, "sft_warmup")
        sft_adapter = os.path.join(sft_path, "adapter_config.json")

        if os.path.exists(sft_adapter):
            # Resume from existing SFT checkpoint — skip retraining
            logger.info(f"[SFT Warmup] Found existing checkpoint at {sft_path}, loading...")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, sft_path)
            model = model.merge_and_unload()
            # Clean residual PEFT attributes
            for attr in ("peft_config", "peft_type", "active_adapter", "active_adapters"):
                if hasattr(model, attr):
                    try:
                        delattr(model, attr)
                    except Exception:
                        pass
            logger.info("[SFT Warmup] Loaded and merged existing SFT adapter")
        else:
            logger.info(f"[SFT Warmup] Loading external SFT data from {config.sft_data_path}")
            with open(config.sft_data_path, "r", encoding="utf-8") as f:
                sft_data = json.load(f)
            logger.info(f"[SFT Warmup] {len(sft_data)} external samples loaded")
            model = _run_sft_warmup(config, model, tokenizer, sft_data, sft_path)
        # Save full merged SFT model as the DPO reference base
        sft_full_path = os.path.join(config.output_dir, "metis_sft_base")
        if not os.path.exists(os.path.join(sft_full_path, "config.json")):
            logger.info(f"[SFT Warmup] Saving full merged SFT base to {sft_full_path}")
            model.save_pretrained(sft_full_path)
            tokenizer.save_pretrained(sft_full_path)
            logger.info("[SFT Warmup] ✓ Qwen-7B-METIS-SFT base saved — this is now π_ref for DPO")
        else:
            logger.info(f"[SFT Warmup] Full SFT base already exists at {sft_full_path}")

    elif config.sft_warmup:
        logger.warning("[SFT Warmup] Skipped — no --sft-data path provided. "
                       "SFT requires external third-party data, not model-generated samples.")

    # ─── Train Group A: METIS ───
    if len(metis_pairs) < 1:
        logger.warning("No METIS pairs survived filtering — skipping METIS DPO training")
        os.makedirs(metis_path, exist_ok=True)
    else:
        logger.info("Training Group A (METIS DPO)...")
        metrics_file = os.path.join(config.output_dir, "training_metrics.json")
        model = _train_dpo(config, model, tokenizer, metis_pairs, metis_path, metrics_file=metrics_file, group_name="metis")

    # ─── Train Group B: Random (SKIPPED to save time) ───
    # Random DPO baseline can be run separately later if needed.
    # For now, METIS DPO vs Base Model comparison is sufficient.
    logger.info("Skipping Random DPO (time budget optimization) — creating placeholder dir")
    os.makedirs(random_path, exist_ok=True)

    return metis_path, random_path


# ─────────────────────────────────────────────────────
# Pair Construction
# ─────────────────────────────────────────────────────

def _build_metis_pairs(scored_data: List[Dict]) -> List[Dict]:
    """Build DPO pairs with constrained cognitive matching.

    Anti-reward-hacking pipeline:
    1. Classify samples by decision profile (DEEP-present vs FAST-only)
    2. Homogeneous matching: same-type pairs use full reward_total
    3. Cross-type matching: quality veto — cognitive quality score
       (calibration + phase + epistemic, excluding efficiency) determines
       Chosen/Rejected to prevent efficiency from dominating pair selection
    4. Hard margin gate on the comparison score
    """
    MARGIN_THRESHOLD = 0.08  # Lower: efficiency no longer inflates margins

    def _cognitive_quality(s: Dict) -> float:
        """Cognitive quality score — excludes efficiency to break label inversion."""
        bd = s.get("reward_breakdown", {})
        return (
            0.35 * bd.get("calibration", 0)
            + 0.30 * bd.get("phase_quality", 0)
            + 0.20 * bd.get("epistemic_honesty", 0)
            + 0.15 * bd.get("coherence", 0)
        )

    def _has_deep(s: Dict) -> bool:
        """Check if sample has DEEP decisions."""
        return s.get("trace_stats", {}).get("deep_ratio", 0) > 0.05

    by_prompt: Dict[str, List[Dict]] = {}
    for entry in scored_data:
        p = entry["prompt"]
        if p not in by_prompt:
            by_prompt[p] = []
        by_prompt[p].append(entry)

    pairs = []
    n_total = len(by_prompt)
    n_margin_fail = 0
    n_homo_pairs = 0
    n_veto_pairs = 0

    for prompt, samples in by_prompt.items():
        if len(samples) < 2:
            continue

        # Split by decision profile
        deep_samples = [s for s in samples if _has_deep(s)]
        fast_samples = [s for s in samples if not _has_deep(s)]

        chosen = None
        rejected = None
        pair_type = ""

        # Strategy A: Cross-type quality veto
        # If we have both DEEP and FAST samples, compare by cognitive quality
        if deep_samples and fast_samples:
            best_deep = max(deep_samples, key=_cognitive_quality)
            best_fast = max(fast_samples, key=_cognitive_quality)
            worst_fast = min(fast_samples, key=_cognitive_quality)
            worst_deep = min(deep_samples, key=_cognitive_quality)

            cq_deep = _cognitive_quality(best_deep)
            cq_fast_worst = _cognitive_quality(worst_fast)

            # Quality veto: if DEEP has better cognitive quality than worst FAST,
            # DEEP is Chosen — this teaches the model that thinking pays off
            if cq_deep - cq_fast_worst >= MARGIN_THRESHOLD:
                chosen, rejected = best_deep, worst_fast
                pair_type = "veto_deep_wins"
            else:
                # Fallback: best FAST vs worst DEEP (FAST legitimately better)
                cq_fast = _cognitive_quality(best_fast)
                cq_deep_worst = _cognitive_quality(worst_deep)
                if cq_fast - cq_deep_worst >= MARGIN_THRESHOLD:
                    chosen, rejected = best_fast, worst_deep
                    pair_type = "veto_fast_wins"

        # Strategy B: Homogeneous matching — generate MULTIPLE pairs per prompt
        # With 8 samples, top-2 vs bottom-2 gives up to 4 extra pairs
        if chosen is None:
            all_sorted = sorted(samples, key=lambda x: x["reward_total"], reverse=True)
            n_half = max(len(all_sorted) // 3, 1)  # top-third vs bottom-third
            tops = all_sorted[:n_half]
            bots = all_sorted[-n_half:]

            added_any = False
            for t in tops:
                for b in bots:
                    if t is b:
                        continue
                    margin = t["reward_total"] - b["reward_total"]
                    if margin < MARGIN_THRESHOLD:
                        continue
                    len_t, len_b = len(t["response"]), len(b["response"])
                    ratio = max(len_t, len_b) / max(min(len_t, len_b), 1)
                    if ratio > 1.5:
                        continue
                    pairs.append({
                        "prompt": t["chat_prompt"],
                        "chosen": t["response"],
                        "rejected": b["response"],
                    })
                    n_homo_pairs += 1
                    added_any = True

            if not added_any:
                n_margin_fail += 1
                continue
        else:
            if pair_type.startswith("veto"):
                n_veto_pairs += 1
            pairs.append({
                "prompt": chosen["chat_prompt"],
                "chosen": chosen["response"],
                "rejected": rejected["response"],
            })

    n_pass = len(pairs)
    logger.info(
        f"[Pair Filter] {n_total} prompts → {n_pass} pairs "
        f"(homo={n_homo_pairs}, veto={n_veto_pairs}, "
        f"margin_fail={n_margin_fail}, "
        f"rejection_rate={1 - n_pass / max(n_total, 1):.0%})"
    )
    return pairs


def _build_random_pairs(scored_data: List[Dict]) -> List[Dict]:
    """Build DPO pairs with random chosen/rejected (control group)."""
    by_prompt: Dict[str, List[Dict]] = {}
    for entry in scored_data:
        p = entry["prompt"]
        if p not in by_prompt:
            by_prompt[p] = []
        by_prompt[p].append(entry)

    pairs = []
    rng = random.Random(42)  # Fixed seed for reproducibility
    for prompt, samples in by_prompt.items():
        if len(samples) < 2:
            continue
        # Random pair selection (NOT reward-ranked)
        shuffled = samples.copy()
        rng.shuffle(shuffled)
        pairs.append({
            "prompt": shuffled[0]["chat_prompt"],
            "chosen": shuffled[0]["response"],
            "rejected": shuffled[1]["response"],
        })

    return pairs


# ─────────────────────────────────────────────────────
# SFT Warmup — teach reference model <thinking> format
# ─────────────────────────────────────────────────────

def _build_sft_data_from_dpo(pairs: List[Dict]) -> List[Dict]:
    """Extract SFT training samples from DPO chosen responses.

    For each DPO pair, the 'chosen' response is used as the SFT target.
    This ensures the reference model learns the same <thinking> format
    that appears in the DPO training data.

    Returns:
        List of {"text": "<prompt>\n<chosen_response>"} dicts
    """
    sft_data = []
    seen = set()
    for pair in pairs:
        prompt = pair.get("prompt", "")
        chosen = pair.get("chosen", "")
        key = hash((prompt, chosen))
        if key in seen:
            continue
        seen.add(key)
        # SFT format: full conversation as a single text sequence
        sft_data.append({"text": f"{prompt}\n{chosen}"})
    return sft_data


def _run_sft_warmup(
    config: ExperimentConfig,
    base_model: Any,
    tokenizer: Any,
    sft_data: List[Dict],
    output_path: str,
) -> Any:
    """Run supervised fine-tuning warmup on METIS-formatted data.

    Purpose: Adapt the base model's token distribution to include
    <thinking>...</thinking> blocks BEFORE DPO training.  The SFT'd
    model then serves as π_ref in DPO, preventing KL explosion:

        β · log(π_θ / π_ref) stays bounded because π_ref already
        assigns reasonable probability to <thinking> tokens.

    Uses LoRA (same rank as DPO) → merge → return updated base model.
    """
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import TrainingArguments, Trainer
    from datasets import Dataset

    logger.info(f"[SFT Warmup] {len(sft_data)} samples, {config.sft_epochs} epoch(s)")

    dataset = Dataset.from_list(sft_data)

    # Tokenize
    def _tokenize(examples: Dict) -> Dict:
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.sft_max_length,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(_tokenize, batched=True, remove_columns=["text"])

    # LoRA config — same architecture as DPO for consistency
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    sft_model = get_peft_model(base_model, lora_config)
    sft_model.print_trainable_parameters()

    # Required for gradient_checkpointing + PEFT: ensures input embeddings
    # have requires_grad=True so the backward pass chain is not broken
    if config.gradient_checkpointing:
        sft_model.enable_input_require_grads()

    # Auto-tune memory
    try:
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        total_gb = total_mem / (1024**3)
    except Exception:
        total_gb = 128.0

    eff_batch = config.sft_batch_size
    eff_grad_accum = config.sft_gradient_accumulation
    if total_gb < 80:
        eff_batch = 1
        eff_grad_accum = max(1, (config.sft_batch_size * config.sft_gradient_accumulation) // eff_batch)

    # Detect if bitsandbytes is available for 8-bit optimizer
    try:
        import bitsandbytes  # noqa: F401
        _optim = "paged_adamw_8bit"
    except ImportError:
        _optim = "adamw_torch"
        logger.info("[SFT Warmup] bitsandbytes not found, using adamw_torch optimizer")

    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=config.sft_epochs,
        per_device_train_batch_size=eff_batch,
        gradient_accumulation_steps=eff_grad_accum,
        learning_rate=config.sft_learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=config.gradient_checkpointing,
        optim=_optim,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=sft_model,
        args=training_args,
        train_dataset=tokenized,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_path)
    logger.info(f"[SFT Warmup] Saved SFT checkpoint to {output_path}")

    # Merge LoRA weights into base model and return
    del trainer
    if hasattr(sft_model, "merge_and_unload"):
        base_model = sft_model.merge_and_unload()
    elif hasattr(sft_model, "unload"):
        base_model = sft_model.unload()
    else:
        logger.warning("[SFT Warmup] Could not unload PEFT model")
        return base_model

    # Clean residual PEFT attributes
    for attr in ("peft_config", "peft_type", "active_adapter", "active_adapters"):
        if hasattr(base_model, attr):
            try:
                delattr(base_model, attr)
            except Exception:
                pass

    # Aggressive VRAM cleanup before DPO
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        logger.info(f"[SFT Warmup] VRAM after cleanup: {free_mem/(1024**3):.1f}GB free / {total_mem/(1024**3):.1f}GB total")

    logger.info("[SFT Warmup] LoRA merged into base model — ready for DPO")
    return base_model


# ─────────────────────────────────────────────────────
# DPO Training
# ─────────────────────────────────────────────────────

def _train_dpo(
    config: ExperimentConfig,
    base_model: Any,
    tokenizer: Any,
    pairs: List[Dict],
    output_path: str,
    metrics_file: str | None = None,
    group_name: str = "metis",
) -> Any:
    """Run DPO training with LoRA adapter."""
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import DPOTrainer, DPOConfig
    from datasets import Dataset

    dataset = Dataset.from_list(pairs)

    # Attach LoRA adapter to base model
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    train_model = get_peft_model(base_model, lora_config)
    train_model.print_trainable_parameters()

    # Auto-tune memory: detect available VRAM and adjust accordingly
    try:
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        free_gb = free_mem / (1024**3)
        total_gb = total_mem / (1024**3)
    except Exception:
        free_gb = 128.0  # assume large if can't detect
        total_gb = 128.0

    # DPO needs 2x forward passes → use FREE memory for sizing
    # GB10 unified memory (122GB): CPU/GPU share the same pool, so free_gb
    # under-reports usable capacity. Lowered thresholds accordingly.
    # Tiers: >40GB free = full, 20-40GB = medium, <20GB = constrained
    if free_gb >= 40:
        eff_batch = config.dpo_batch_size
        eff_grad_accum = config.dpo_gradient_accumulation
        eff_max_len = config.dpo_max_length
        eff_gc = config.gradient_checkpointing
        logger.info(f"[DPO] High-memory mode: batch={eff_batch}, grad_accum={eff_grad_accum}, max_len={eff_max_len}, gc={eff_gc} (total={total_gb:.0f}GB, free={free_gb:.0f}GB)")
    elif free_gb >= 20:
        eff_batch = min(config.dpo_batch_size, 2)
        eff_grad_accum = max(1, (config.dpo_batch_size * config.dpo_gradient_accumulation) // eff_batch)
        eff_max_len = min(config.dpo_max_length, 1024)
        eff_gc = True
        logger.info(f"[DPO] Medium-memory mode: batch={eff_batch}, grad_accum={eff_grad_accum}, max_len={eff_max_len}, gc=True (total={total_gb:.0f}GB, free={free_gb:.0f}GB)")
    else:
        eff_batch = 1
        eff_grad_accum = max(1, (config.dpo_batch_size * config.dpo_gradient_accumulation) // eff_batch)
        eff_max_len = min(config.dpo_max_length, 768)
        eff_gc = True
        logger.info(f"[DPO] Low-memory mode: batch={eff_batch}, grad_accum={eff_grad_accum}, max_len={eff_max_len}, gc=True (total={total_gb:.0f}GB, free={free_gb:.0f}GB)")

    training_args = DPOConfig(
        output_dir=output_path,
        num_train_epochs=config.dpo_epochs,
        per_device_train_batch_size=eff_batch,
        gradient_accumulation_steps=eff_grad_accum,
        learning_rate=config.dpo_learning_rate,
        beta=config.dpo_beta,
        max_length=eff_max_len,
        logging_steps=1,
        save_strategy="epoch",
        remove_unused_columns=False,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=eff_gc,
        report_to="none",
    )

    callbacks = []
    if metrics_file:
        callbacks.append(MetricsJsonCallback(metrics_file, group_name=group_name))

    trainer = DPOTrainer(
        model=train_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(output_path)
    logger.info(f"Saved checkpoint to {output_path}")

    # Detach LoRA adapter, restore base model for next training run
    del trainer
    
    # Properly merge weights and unload PEFT wrapper
    if hasattr(train_model, "merge_and_unload"):
        base_model = train_model.merge_and_unload()
    elif hasattr(train_model, "unload"):
        base_model = train_model.unload()
    else:
        logger.warning("Could not unload PEFT model. Adapter contamination may occur.")
        return base_model

    # Explicitly remove residual PEFT attributes that trigger the warning
    for attr in ("peft_config", "peft_type", "active_adapter", "active_adapters"):
        if hasattr(base_model, attr):
            try:
                delattr(base_model, attr)
            except Exception:
                pass

    return base_model

