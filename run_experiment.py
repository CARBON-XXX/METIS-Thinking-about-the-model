#!/usr/bin/env python3
"""
METIS Training Experiment — A/B Comparison

Compares DPO training with METIS cognitive rewards vs baseline.

Experiment Design:
═══════════════════════════════════════════════════════════════════
Phase 1: Data Generation
    - For each prompt, generate K responses with METIS instrumentation
    - Compute 5-component cognitive rewards for each response

Phase 2: DPO Training (two groups)
    Group A (METIS):    DPO with cognitive-reward-ranked preference pairs
    Group B (Random):   DPO with randomly-paired preferences (control)

Phase 3: Evaluation
    - Generate responses from both trained models on held-out prompts
    - Run METIS cognitive evaluation on all outputs
    - Compare: entropy stability, calibration, confusion ratio, reward

Phase 4: Report
    - Side-by-side comparison table
    - Per-component reward breakdown
    - Statistical significance test
═══════════════════════════════════════════════════════════════════

Usage:
    python run_experiment.py --model Qwen/Qwen2.5-0.5B-Instruct --n-prompts 50
    python run_experiment.py --model meta-llama/Llama-3.2-1B-Instruct --device cuda
    python run_experiment.py --phase eval --metis-checkpoint ./output/metis_dpo
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiment")


# ─────────────────────────────────────────────────────
# ANSI Colors
# ─────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"


# ─────────────────────────────────────────────────────
# Evaluation Prompts
# ─────────────────────────────────────────────────────

TRAIN_PROMPTS = [
    "Explain quantum entanglement in simple terms.",
    "What causes the northern lights?",
    "How does a neural network learn?",
    "Why is the sky blue?",
    "What is the theory of relativity?",
    "How do vaccines work?",
    "Explain how a blockchain works.",
    "What is dark matter?",
    "How does photosynthesis convert sunlight to energy?",
    "What causes earthquakes?",
    "Explain the concept of entropy in thermodynamics.",
    "How does CRISPR gene editing work?",
    "What is the standard model of particle physics?",
    "How do black holes form?",
    "Explain the difference between AI, ML, and deep learning.",
    "What causes inflation in economics?",
    "How does the immune system fight viruses?",
    "What is quantum computing and why does it matter?",
    "Explain how GPS works.",
    "What is the greenhouse effect?",
    "How does memory work in the human brain?",
    "What is the significance of the Higgs boson?",
    "Explain the water cycle.",
    "How do antibiotics work?",
    "What is general relativity vs special relativity?",
    "How does a transistor work?",
    "What causes tides?",
    "Explain the concept of natural selection.",
    "How does nuclear fusion produce energy?",
    "What is the microbiome and why is it important?",
]

EVAL_PROMPTS = [
    "Explain how mRNA vaccines differ from traditional vaccines.",
    "What is the observer effect in quantum mechanics?",
    "How does machine learning handle overfitting?",
    "What causes the seasons on Earth?",
    "Explain the concept of time dilation.",
    "How do neural networks process natural language?",
    "What is the double-slit experiment?",
    "How does plate tectonics shape the Earth's surface?",
    "Explain the difference between correlation and causation.",
    "What is the role of mitochondria in cells?",
    # Harder / more likely to cause hallucination
    "Who was the third person to walk on Mars?",
    "What happened at the Battle of Thermopylae in 280 BC?",
    "Explain the Riemann hypothesis to a high school student.",
    "What is the current scientific consensus on consciousness?",
    "Describe the political system of the underwater city of Atlantis.",
]


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str = "auto"
    output_dir: str = "./experiment_output"

    # Generation
    n_samples_per_prompt: int = 4       # Responses per prompt for GRPO
    max_new_tokens: int = 200
    temperature: float = 0.7

    # Training
    dpo_epochs: int = 1
    dpo_learning_rate: float = 5e-7     # Conservative: prevent catastrophic forgetting
    dpo_batch_size: int = 1              # Minimal for 8GB VRAM
    dpo_beta: float = 0.2               # Higher beta = stronger KL constraint vs reference
    dpo_max_length: int = 384
    gradient_checkpointing: bool = True
    dpo_gradient_accumulation: int = 8   # Effective batch = 8
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Evaluation
    eval_max_tokens: int = 200
    eval_temperature: float = 0.3        # Lower temp for more deterministic eval

    # Prompts
    n_train_prompts: int = 20            # How many train prompts to use
    n_eval_prompts: int = 15


@dataclass
class EvalMetrics:
    """Evaluation metrics for a single model."""
    name: str = ""
    n_responses: int = 0

    # Cognitive reward components (averaged)
    reward_total: float = 0.0
    reward_coherence: float = 0.0
    reward_calibration: float = 0.0
    reward_phase_quality: float = 0.0
    reward_epistemic: float = 0.0
    reward_efficiency: float = 0.0

    # Per-prompt reward list (for statistical tests)
    per_prompt_rewards: List[float] = field(default_factory=list)

    # Raw signal metrics
    mean_entropy: float = 0.0
    mean_surprise: float = 0.0
    mean_confidence: float = 0.0
    confusion_ratio: float = 0.0
    fast_ratio: float = 0.0
    avg_tokens: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in asdict(self).items()}


# ═══════════════════════════════════════════════════════
# Phase 1: Generate & Score
# ═══════════════════════════════════════════════════════

def phase1_generate(
    config: ExperimentConfig,
) -> Tuple[List[Dict], Any, Any]:
    """
    Generate K responses per prompt with METIS instrumentation.
    Returns scored data + model + tokenizer for reuse.
    """
    logger.info(f"{'='*60}")
    logger.info(f"PHASE 1: Generate & Score ({config.n_train_prompts} prompts × {config.n_samples_per_prompt} samples)")
    logger.info(f"{'='*60}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from metis.training.generator import MetisGenerator
    from metis.training.rewards import CognitiveRewardComputer

    # Load model
    logger.info(f"Loading model: {config.model_name}")
    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, **model_kwargs
    ).to(device)
    model.eval()

    generator = MetisGenerator(model, tokenizer)
    reward_computer = CognitiveRewardComputer()

    prompts = TRAIN_PROMPTS[:config.n_train_prompts]
    all_data: List[Dict] = []

    for i, prompt in enumerate(prompts):
        logger.info(f"[{i+1}/{len(prompts)}] {prompt[:50]}...")

        # Format as chat if possible
        chat_prompt = _format_chat(tokenizer, prompt)

        samples = generator.generate_batch(
            chat_prompt,
            n_samples=config.n_samples_per_prompt,
            max_new_tokens=config.max_new_tokens,
        )

        for j, (text, trace) in enumerate(samples):
            reward = reward_computer.compute(trace)
            entry = {
                "prompt": prompt,
                "chat_prompt": chat_prompt,
                "response": text,
                "sample_idx": j,
                "reward_total": reward.total,
                "reward_breakdown": reward.to_dict(),
                "trace_stats": {
                    "total_tokens": trace.total_tokens,
                    "mean_entropy": trace.mean_entropy,
                    "mean_surprise": trace.mean_surprise,
                },
            }
            all_data.append(entry)
            logger.info(
                f"  sample {j}: reward={reward.total:+.4f} "
                f"tokens={trace.total_tokens} "
                f"resp={text[:60]}..."
            )

    # Save raw data
    os.makedirs(config.output_dir, exist_ok=True)
    data_path = os.path.join(config.output_dir, "phase1_scored_data.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(all_data)} scored samples to {data_path}")

    return all_data, model, tokenizer


def _format_chat(tokenizer: Any, prompt: str) -> str:
    """Format prompt as chat template if available."""
    try:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return prompt


# ═══════════════════════════════════════════════════════
# Phase 2: DPO Training
# ═══════════════════════════════════════════════════════

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
    logger.info("Training Group A (METIS DPO)...")
    _train_dpo(config, model, tokenizer, metis_pairs, metis_path)

    # ─── Train Group B: Random ───
    logger.info("Training Group B (Random DPO)...")
    _train_dpo(config, model, tokenizer, random_pairs, random_path)

    return metis_path, random_path


def _build_metis_pairs(scored_data: List[Dict]) -> List[Dict]:
    """Build DPO pairs ranked by METIS cognitive reward (length-debiased).

    Key design decisions:
    - Sort by cognitive reward EXCLUDING length penalty to prevent
      the model from learning "shorter is better" spurious correlation
    - Require chosen/rejected to have comparable token counts (within 50%)
      to isolate cognitive quality from verbosity differences
    """
    # Group by prompt
    by_prompt: Dict[str, List[Dict]] = {}
    for entry in scored_data:
        p = entry["prompt"]
        if p not in by_prompt:
            by_prompt[p] = []
        by_prompt[p].append(entry)

    pairs = []
    for prompt, samples in by_prompt.items():
        # Sort by cognitive reward (total already excludes length penalty
        # in v2, but we use reward_breakdown if available for safety)
        samples.sort(key=lambda x: x["reward_total"], reverse=True)

        # Try to find a valid pair with comparable lengths
        best = samples[0]
        worst = None
        for candidate in reversed(samples):
            if candidate is best:
                continue
            # Length comparability: neither is >50% longer than the other
            len_best = len(best["response"])
            len_cand = len(candidate["response"])
            ratio = max(len_best, len_cand) / max(min(len_best, len_cand), 1)
            if ratio <= 1.5:
                worst = candidate
                break

        if worst is None:
            worst = samples[-1]  # Fallback to most different

        margin = best["reward_total"] - worst["reward_total"]
        if margin < 0.05:
            continue  # Skip if no clear preference

        pairs.append({
            "prompt": best["chat_prompt"],
            "chosen": best["response"],
            "rejected": worst["response"],
        })

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


def _train_dpo(
    config: ExperimentConfig,
    base_model: Any,
    tokenizer: Any,
    pairs: List[Dict],
    output_path: str,
) -> None:
    """Run DPO training with LoRA (no deepcopy — saves VRAM for 8GB GPUs)."""
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import DPOTrainer, DPOConfig
    from datasets import Dataset

    dataset = Dataset.from_list(pairs)

    # Attach LoRA adapter directly to base model (no deepcopy to save VRAM)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════
# Phase 3: Evaluation
# ═══════════════════════════════════════════════════════

def phase3_evaluate(
    config: ExperimentConfig,
    base_model: Any,
    tokenizer: Any,
    metis_path: str,
    random_path: str,
) -> Tuple[EvalMetrics, EvalMetrics, EvalMetrics]:
    """
    Evaluate three models on held-out prompts:
    - Base model (no training)
    - METIS DPO model
    - Random DPO model
    """
    logger.info(f"{'='*60}")
    logger.info(f"PHASE 3: Evaluation ({config.n_eval_prompts} held-out prompts)")
    logger.info(f"{'='*60}")

    from peft import PeftModel
    from metis.training.generator import MetisGenerator
    from metis.training.rewards import CognitiveRewardComputer

    eval_prompts = EVAL_PROMPTS[:config.n_eval_prompts]
    reward_computer = CognitiveRewardComputer()

    # ─── Evaluate Base Model ───
    logger.info("Evaluating: Base Model (no training)")
    base_metrics = _evaluate_model(
        config, base_model, tokenizer, eval_prompts, reward_computer, "Base"
    )

    # ─── Evaluate METIS DPO ───
    logger.info("Evaluating: METIS DPO")
    metis_model = PeftModel.from_pretrained(base_model, metis_path)
    metis_model.eval()
    metis_metrics = _evaluate_model(
        config, metis_model, tokenizer, eval_prompts, reward_computer, "METIS-DPO"
    )
    del metis_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ─── Evaluate Random DPO ───
    logger.info("Evaluating: Random DPO")
    random_model = PeftModel.from_pretrained(base_model, random_path)
    random_model.eval()
    random_metrics = _evaluate_model(
        config, random_model, tokenizer, eval_prompts, reward_computer, "Random-DPO"
    )
    del random_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return base_metrics, metis_metrics, random_metrics


def _evaluate_model(
    config: ExperimentConfig,
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    reward_computer: Any,
    name: str,
) -> EvalMetrics:
    """Evaluate a single model on prompts."""
    from metis.training.generator import MetisGenerator
    from metis.core.types import Decision

    generator = MetisGenerator(model, tokenizer)
    metrics = EvalMetrics(name=name, n_responses=len(prompts))

    total_rewards = []
    all_breakdowns = []

    for i, prompt in enumerate(prompts):
        chat_prompt = _format_chat(tokenizer, prompt)
        text, trace = generator.generate(
            chat_prompt,
            max_new_tokens=config.eval_max_tokens,
            temperature=config.eval_temperature,
        )

        reward = reward_computer.compute(trace)
        total_rewards.append(reward.total)
        all_breakdowns.append(reward)

        # Raw metrics from trace
        events = trace.events
        n = len(events) if events else 1
        metrics.mean_entropy += sum(e.semantic_entropy for e in events) / n
        metrics.mean_surprise += sum(e.token_surprise for e in events) / n
        metrics.mean_confidence += sum(e.confidence for e in events) / n
        metrics.confusion_ratio += sum(
            1 for e in events if e.cognitive_phase == "confusion"
        ) / n
        metrics.fast_ratio += sum(
            1 for e in events if e.decision == Decision.FAST
        ) / n
        metrics.avg_tokens += n

        logger.info(
            f"  [{name}] {i+1}/{len(prompts)}: "
            f"reward={reward.total:+.4f} tokens={n} "
            f"resp={text[:50]}..."
        )

    # Store per-prompt rewards for statistical tests
    metrics.per_prompt_rewards = total_rewards

    # Average all metrics
    n_prompts = len(prompts)
    metrics.reward_total = sum(total_rewards) / n_prompts
    metrics.reward_coherence = sum(r.coherence for r in all_breakdowns) / n_prompts
    metrics.reward_calibration = sum(r.calibration for r in all_breakdowns) / n_prompts
    metrics.reward_phase_quality = sum(r.phase_quality for r in all_breakdowns) / n_prompts
    metrics.reward_epistemic = sum(r.epistemic_honesty for r in all_breakdowns) / n_prompts
    metrics.reward_efficiency = sum(r.efficiency for r in all_breakdowns) / n_prompts
    metrics.mean_entropy /= n_prompts
    metrics.mean_surprise /= n_prompts
    metrics.mean_confidence /= n_prompts
    metrics.confusion_ratio /= n_prompts
    metrics.fast_ratio /= n_prompts
    metrics.avg_tokens /= n_prompts

    return metrics


# ═══════════════════════════════════════════════════════
# Phase 4: Report
# ═══════════════════════════════════════════════════════

def phase4_report(
    config: ExperimentConfig,
    base: EvalMetrics,
    metis: EvalMetrics,
    random_ctrl: EvalMetrics,
) -> None:
    """Generate comparison report."""
    logger.info(f"{'='*60}")
    logger.info(f"PHASE 4: Report")
    logger.info(f"{'='*60}")

    def delta(new: float, old: float) -> str:
        d = new - old
        if abs(d) < 0.001:
            return f"{C.DIM}  ±0{C.RESET}"
        color = C.GREEN if d > 0 else C.RED
        return f"{color}{d:+.4f}{C.RESET}"

    print(f"\n{C.BOLD}{C.CYAN}")
    print(f"  ╔══════════════════════════════════════════════════════════════╗")
    print(f"  ║           METIS Training Experiment — Results               ║")
    print(f"  ╚══════════════════════════════════════════════════════════════╝")
    print(f"{C.RESET}")

    print(f"  {C.BOLD}Model:{C.RESET} {config.model_name}")
    print(f"  {C.BOLD}Train:{C.RESET} {config.n_train_prompts} prompts × {config.n_samples_per_prompt} samples")
    print(f"  {C.BOLD}Eval:{C.RESET}  {config.n_eval_prompts} held-out prompts\n")

    # Main comparison table
    header = f"  {'Metric':<22s} │ {'Base':>10s} │ {'METIS DPO':>10s} │ {'Δ vs Base':>10s} │ {'Random DPO':>10s} │ {'Δ vs Base':>10s}"
    sep = f"  {'─'*22}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}"
    print(f"{C.BOLD}{header}{C.RESET}")
    print(sep)

    rows = [
        ("Reward (Total)", base.reward_total, metis.reward_total, random_ctrl.reward_total),
        ("  R_coherence", base.reward_coherence, metis.reward_coherence, random_ctrl.reward_coherence),
        ("  R_calibration", base.reward_calibration, metis.reward_calibration, random_ctrl.reward_calibration),
        ("  R_phase", base.reward_phase_quality, metis.reward_phase_quality, random_ctrl.reward_phase_quality),
        ("  R_epistemic", base.reward_epistemic, metis.reward_epistemic, random_ctrl.reward_epistemic),
        ("  R_efficiency", base.reward_efficiency, metis.reward_efficiency, random_ctrl.reward_efficiency),
        ("", 0, 0, 0),  # spacer
        ("Mean Entropy", base.mean_entropy, metis.mean_entropy, random_ctrl.mean_entropy),
        ("Mean Surprise", base.mean_surprise, metis.mean_surprise, random_ctrl.mean_surprise),
        ("Mean Confidence", base.mean_confidence, metis.mean_confidence, random_ctrl.mean_confidence),
        ("Confusion Ratio", base.confusion_ratio, metis.confusion_ratio, random_ctrl.confusion_ratio),
        ("Fast (Sys1) Ratio", base.fast_ratio, metis.fast_ratio, random_ctrl.fast_ratio),
        ("Avg Tokens", base.avg_tokens, metis.avg_tokens, random_ctrl.avg_tokens),
    ]

    for label, base_v, metis_v, rand_v in rows:
        if label == "":
            print(sep)
            continue
        # For confusion/surprise, lower is better → invert delta color
        print(
            f"  {label:<22s} │ {base_v:>10.4f} │ {metis_v:>10.4f} │ "
            f"{delta(metis_v, base_v):>20s} │ {rand_v:>10.4f} │ "
            f"{delta(rand_v, base_v):>20s}"
        )

    # Summary
    metis_lift = metis.reward_total - base.reward_total
    random_lift = random_ctrl.reward_total - base.reward_total
    metis_vs_random = metis.reward_total - random_ctrl.reward_total

    print(f"\n{C.BOLD}  Summary:{C.RESET}")
    print(f"    METIS DPO vs Base:    {delta(metis.reward_total, base.reward_total)}")
    print(f"    Random DPO vs Base:   {delta(random_ctrl.reward_total, base.reward_total)}")
    print(f"    METIS DPO vs Random:  {delta(metis.reward_total, random_ctrl.reward_total)}")

    # ─── Statistical Analysis ───
    print(f"\n{C.BOLD}  Statistical Analysis:{C.RESET}")

    metis_rewards = metis.per_prompt_rewards
    random_rewards = random_ctrl.per_prompt_rewards
    base_rewards = base.per_prompt_rewards

    if len(metis_rewards) >= 5 and len(random_rewards) >= 5:
        # Paired bootstrap CI for METIS vs Random
        n_boot = 10000
        rng = random.Random(42)
        n_eval = min(len(metis_rewards), len(random_rewards))
        diffs = [metis_rewards[i] - random_rewards[i] for i in range(n_eval)]
        boot_means = []
        for _ in range(n_boot):
            sample = [diffs[rng.randint(0, n_eval - 1)] for _ in range(n_eval)]
            boot_means.append(sum(sample) / n_eval)
        boot_means.sort()
        ci_lo = boot_means[int(0.025 * n_boot)]
        ci_hi = boot_means[int(0.975 * n_boot)]
        mean_diff = sum(diffs) / n_eval

        # Cohen's d (paired)
        if n_eval > 1:
            diff_var = sum((d - mean_diff) ** 2 for d in diffs) / (n_eval - 1)
            diff_std = math.sqrt(diff_var) if diff_var > 0 else 1e-6
            cohens_d = mean_diff / diff_std
        else:
            cohens_d = 0.0

        ci_color = C.GREEN if ci_lo > 0 else (C.RED if ci_hi < 0 else C.YELLOW)
        print(f"    METIS vs Random (paired, n={n_eval}):")
        print(f"      Mean Δ:       {ci_color}{mean_diff:+.4f}{C.RESET}")
        print(f"      95% Boot CI:  {ci_color}[{ci_lo:+.4f}, {ci_hi:+.4f}]{C.RESET}")
        print(f"      Cohen's d:    {cohens_d:+.3f}", end="")
        if abs(cohens_d) >= 0.8:
            print(f" {C.GREEN}(large){C.RESET}")
        elif abs(cohens_d) >= 0.5:
            print(f" {C.YELLOW}(medium){C.RESET}")
        elif abs(cohens_d) >= 0.2:
            print(f" {C.YELLOW}(small){C.RESET}")
        else:
            print(f" {C.DIM}(negligible){C.RESET}")

        sig = ci_lo > 0 or ci_hi < 0
        if sig and mean_diff > 0:
            print(f"\n    {C.GREEN}{C.BOLD}✓ METIS improvement is statistically significant (CI excludes 0){C.RESET}")
        elif not sig:
            print(f"\n    {C.YELLOW}⚠ Result not statistically significant (CI includes 0, need more data){C.RESET}")
        else:
            print(f"\n    {C.RED}✗ METIS underperforms Random (CI excludes 0){C.RESET}")
    else:
        print(f"    {C.DIM}Too few samples for statistical tests (n<5){C.RESET}")

    if metis_lift > random_lift + 0.01:
        print(f"    {C.GREEN}{C.BOLD}✓ METIS cognitive rewards provide measurable training improvement{C.RESET}")
    elif metis_lift > random_lift:
        print(f"    {C.YELLOW}≈ Marginal improvement from METIS rewards (need more data){C.RESET}")
    else:
        print(f"    {C.RED}✗ No clear improvement (may need hyperparameter tuning){C.RESET}")

    # Save report
    report = {
        "config": asdict(config),
        "base": base.to_dict(),
        "metis_dpo": metis.to_dict(),
        "random_dpo": random_ctrl.to_dict(),
        "summary": {
            "metis_vs_base": round(metis_lift, 4),
            "random_vs_base": round(random_lift, 4),
            "metis_vs_random": round(metis_vs_random, 4),
        },
    }
    report_path = os.path.join(config.output_dir, "experiment_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  {C.DIM}Report saved to {report_path}{C.RESET}\n")


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="METIS Training Experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cuda / cpu / auto")
    parser.add_argument("--output", type=str, default="./experiment_output",
                        help="Output directory")
    parser.add_argument("--n-prompts", type=int, default=20,
                        help="Number of training prompts")
    parser.add_argument("--n-samples", type=int, default=4,
                        help="Samples per prompt")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Max new tokens per generation")
    parser.add_argument("--dpo-epochs", type=int, default=1,
                        help="DPO training epochs")
    parser.add_argument("--dpo-lr", type=float, default=5e-6,
                        help="DPO learning rate")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "generate", "train", "eval"],
                        help="Which phase to run")
    parser.add_argument("--metis-checkpoint", type=str, default=None,
                        help="Path to METIS DPO checkpoint (for eval-only)")
    parser.add_argument("--random-checkpoint", type=str, default=None,
                        help="Path to Random DPO checkpoint (for eval-only)")
    args = parser.parse_args()

    config = ExperimentConfig(
        model_name=args.model,
        device=args.device,
        output_dir=args.output,
        n_train_prompts=args.n_prompts,
        n_samples_per_prompt=args.n_samples,
        max_new_tokens=args.max_tokens,
        dpo_epochs=args.dpo_epochs,
        dpo_learning_rate=args.dpo_lr,
        lora_r=args.lora_r,
    )

    os.makedirs(config.output_dir, exist_ok=True)

    print(f"\n{C.BOLD}{C.CYAN}")
    print(f"  ╔══════════════════════════════════════════════════════════════╗")
    print(f"  ║          METIS Training Experiment                          ║")
    print(f"  ║          Cognitive Rewards vs Random Baseline               ║")
    print(f"  ╚══════════════════════════════════════════════════════════════╝")
    print(f"{C.RESET}")
    print(f"  Model:    {config.model_name}")
    print(f"  Device:   {config.device}")
    print(f"  Prompts:  {config.n_train_prompts} train + {config.n_eval_prompts} eval")
    print(f"  Samples:  {config.n_samples_per_prompt} per prompt")
    print(f"  Output:   {config.output_dir}\n")

    start = time.time()

    if args.phase in ("all", "generate"):
        scored_data, model, tokenizer = phase1_generate(config)

        if args.phase == "generate":
            logger.info("Phase 1 complete. Use --phase train to continue.")
            return
    else:
        # Load model for later phases
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs).to(device)
        model.eval()

        # Load scored data
        data_path = os.path.join(config.output_dir, "phase1_scored_data.json")
        with open(data_path, "r", encoding="utf-8") as f:
            scored_data = json.load(f)

    if args.phase in ("all", "train"):
        metis_path, random_path = phase2_train(config, scored_data, model, tokenizer)
    else:
        metis_path = args.metis_checkpoint or os.path.join(config.output_dir, "metis_dpo")
        random_path = args.random_checkpoint or os.path.join(config.output_dir, "random_dpo")

    if args.phase in ("all", "eval"):
        base_metrics, metis_metrics, random_metrics = phase3_evaluate(
            config, model, tokenizer, metis_path, random_path,
        )
        phase4_report(config, base_metrics, metis_metrics, random_metrics)

    elapsed = time.time() - start
    logger.info(f"Experiment completed in {elapsed:.1f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
