"""
METIS Pipeline — Phase 2: DPO Training

Trains two models for A/B comparison:
  Group A (METIS):  DPO with cognitive-reward-ranked preference pairs
  Group B (Random): DPO with randomly-paired preferences (control)
"""
from __future__ import annotations

import logging
import os
import random
from typing import Any, Dict, List, Tuple

import torch

from metis.pipeline.config import ExperimentConfig

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
    metis_pairs = _build_metis_pairs(scored_data)
    random_pairs = _build_random_pairs(scored_data)

    logger.info(f"METIS pairs: {len(metis_pairs)}")
    logger.info(f"Random pairs: {len(random_pairs)}")

    metis_path = os.path.join(config.output_dir, "metis_dpo")
    random_path = os.path.join(config.output_dir, "random_dpo")

    # ─── Train Group A: METIS ───
    if len(metis_pairs) < 1:
        logger.warning("No METIS pairs survived filtering — skipping METIS DPO training")
        os.makedirs(metis_path, exist_ok=True)
    else:
        logger.info("Training Group A (METIS DPO)...")
        _train_dpo(config, model, tokenizer, metis_pairs, metis_path)

    # ─── Train Group B: Random ───
    if len(random_pairs) < 1:
        logger.warning("No Random pairs — skipping Random DPO training")
        os.makedirs(random_path, exist_ok=True)
    else:
        logger.info("Training Group B (Random DPO)...")
        _train_dpo(config, model, tokenizer, random_pairs, random_path)

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
# DPO Training
# ─────────────────────────────────────────────────────

def _train_dpo(
    config: ExperimentConfig,
    base_model: Any,
    tokenizer: Any,
    pairs: List[Dict],
    output_path: str,
) -> None:
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

    training_args = DPOConfig(
        output_dir=output_path,
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
        bf16=torch.cuda.is_available(),
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
    trainer.save_model(output_path)
    logger.info(f"Saved checkpoint to {output_path}")

    # Detach LoRA adapter, restore base model for next training run
    del trainer
    train_model.unload()
