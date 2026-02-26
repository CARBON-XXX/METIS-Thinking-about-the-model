"""
METIS Pipeline — Phase 3: Evaluation & Phase 4: Report

Phase 3: Evaluate three models on held-out prompts
  - Base model (no training)
  - METIS DPO model
  - Random DPO model
  + External benchmarks (TruthfulQA, MMLU)

Phase 4: Generate comparison report with statistical analysis
"""
from __future__ import annotations

import gc
import json
import logging
import math
import os
import random
import threading
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import torch

from metis.pipeline.config import (
    C, ExperimentConfig, EvalMetrics, EVAL_PROMPTS, format_chat,
)

logger = logging.getLogger("experiment")


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
    from metis.training.rewards import CognitiveRewardComputer

    eval_prompts = EVAL_PROMPTS[:config.n_eval_prompts]
    reward_computer = CognitiveRewardComputer()

    # ─── Reload clean base model ───
    # Phase 2 LoRA train/unload corrupts base_model weights.
    # Reload from checkpoint to ensure fair base model evaluation.
    # CRITICAL: do NOT use device_map="auto" — after Phase 2 training,
    # VRAM has fragmented cache residue. Accelerate sees "low free VRAM"
    # and offloads layers to CPU RAM, causing PCIe bus bottleneck that
    # degrades from 15s→42s+ per prompt during autoregressive generation.
    from transformers import AutoModelForCausalLM
    logger.info("Reloading clean base model for evaluation...")
    try:
        del base_model
    except NameError:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    device = config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    base_model.eval()

    # ─── Evaluate Base Model ───
    logger.info("Evaluating: Base Model (no training)")
    base_metrics = _evaluate_model(
        config, base_model, tokenizer, eval_prompts, reward_computer, "Base"
    )

    # ─── Evaluate METIS DPO ───
    _has_metis_ckpt = os.path.exists(os.path.join(metis_path, "adapter_config.json"))
    if _has_metis_ckpt:
        logger.info("Evaluating: METIS DPO")
        metis_model = PeftModel.from_pretrained(base_model, metis_path)
        metis_model.eval()
        metis_metrics = _evaluate_model(
            config, metis_model, tokenizer, eval_prompts, reward_computer, "METIS-DPO"
        )
        # CRITICAL: PeftModel.from_pretrained injects LoRA layers INTO base_model's
        # Linear modules. Simply deleting the PeftModel reference does NOT remove
        # the injected layers. We must destroy and reload from scratch to prevent
        # adapter stacking when loading Random DPO next.
        del metis_model, base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
        logger.info("Reloading clean base model after METIS DPO eval...")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(device)
        base_model.eval()
    else:
        logger.warning("No METIS DPO checkpoint found — using base metrics as fallback")
        metis_metrics = base_metrics

    # ─── Evaluate Random DPO ───
    _has_random_ckpt = os.path.exists(os.path.join(random_path, "adapter_config.json"))
    if _has_random_ckpt:
        logger.info("Evaluating: Random DPO")
        random_model = PeftModel.from_pretrained(base_model, random_path)
        random_model.eval()
        random_metrics = _evaluate_model(
            config, random_model, tokenizer, eval_prompts, reward_computer, "Random-DPO"
        )
        del random_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        logger.warning("No Random DPO checkpoint found — using base metrics as fallback")
        random_metrics = base_metrics

    # ─── External Benchmarks (independent validation) ───
    if config.run_benchmarks:
        logger.info(f"{'='*60}")
        logger.info("PHASE 3b: External Benchmarks (TruthfulQA + MMLU)")
        logger.info(f"{'='*60}")

        from metis.training.benchmarks import BenchmarkSuite

        # Base model benchmarks (model already loaded above)
        logger.info("Benchmarking: Base Model")
        _run_benchmarks(config, base_model, tokenizer, base_metrics)

        # METIS DPO benchmarks
        if _has_metis_ckpt:
            logger.info("Benchmarking: METIS DPO")
            metis_model = PeftModel.from_pretrained(base_model, metis_path)
            metis_model.eval()
            _run_benchmarks(config, metis_model, tokenizer, metis_metrics)
            del metis_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Random DPO benchmarks — need clean base model reload
        if _has_random_ckpt:
            logger.info("Reloading clean base model for Random DPO benchmarks...")
            del base_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            base_model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True,
            ).to(device)
            base_model.eval()

            logger.info("Benchmarking: Random DPO")
            random_model = PeftModel.from_pretrained(base_model, random_path)
            random_model.eval()
            _run_benchmarks(config, random_model, tokenizer, random_metrics)
            del random_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return base_metrics, metis_metrics, random_metrics


# ─────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────

def _run_benchmarks(
    config: ExperimentConfig,
    model: Any,
    tokenizer: Any,
    metrics: EvalMetrics,
) -> None:
    """Run external benchmarks and fill in metrics fields."""
    from metis.training.benchmarks import BenchmarkSuite

    suite = BenchmarkSuite(model, tokenizer)

    tqa = suite.run_truthfulqa(max_questions=config.truthfulqa_questions)
    metrics.truthfulqa_mc1 = tqa.accuracy
    metrics.truthfulqa_mc2 = tqa.sub_scores.get("mc2_accuracy", 0.0)
    metrics.benchmark_details["truthfulqa"] = tqa.to_dict()

    mmlu = suite.run_mmlu(
        n_subjects=config.mmlu_subjects,
        max_per_subject=config.mmlu_per_subject,
    )
    metrics.mmlu_accuracy = mmlu.accuracy
    metrics.benchmark_details["mmlu"] = mmlu.to_dict()

    logger.info(
        f"  [{metrics.name}] TruthfulQA MC1={metrics.truthfulqa_mc1:.1%} "
        f"MC2={metrics.truthfulqa_mc2:.1%} | MMLU={metrics.mmlu_accuracy:.1%}"
    )


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
    from metis.core.types import Decision, CognitiveTrace
    from metis.training.rewards import RewardBreakdown

    WALL_CLOCK_TIMEOUT = 120  # seconds — hard circuit breaker per prompt

    generator = MetisGenerator(model, tokenizer)
    metrics = EvalMetrics(name=name, n_responses=len(prompts))

    total_rewards = []
    all_breakdowns = []

    for i, prompt in enumerate(prompts):
        chat_prompt = format_chat(tokenizer, prompt)

        # ── Wall-clock circuit breaker ──
        # Prevents single-prompt pathological loops from blocking the entire eval
        result_container: list = []
        # Capture generator in local var for thread-safety
        gen_ref = generator

        def _generate_with_timeout(g=gen_ref, p=chat_prompt):
            try:
                samples = g.generate_batch(
                    p,
                    n_samples=1,
                    temperatures=[config.eval_temperature],
                    max_new_tokens=config.eval_max_tokens,
                )
                result_container.append(samples[0])
            except Exception as e:
                logger.warning(f"  [{name}] {i+1}/{len(prompts)}: generation error: {e}")

        gen_thread = threading.Thread(target=_generate_with_timeout, daemon=True)
        gen_thread.start()
        gen_thread.join(timeout=WALL_CLOCK_TIMEOUT)

        if gen_thread.is_alive() or not result_container:
            # Timeout or error — produce fallback metrics
            logger.warning(
                f"  [{name}] {i+1}/{len(prompts)}: TIMEOUT ({WALL_CLOCK_TIMEOUT}s) "
                f"— falling back to zero reward"
            )
            text = ""
            trace = CognitiveTrace()
            reward = RewardBreakdown()
            # CRITICAL: orphaned daemon thread still holds old generator reference.
            # Create fresh generator so next prompt doesn't share METIS state
            # with the still-running background thread (race condition).
            generator = MetisGenerator(model, tokenizer)
        else:
            text, trace = result_container[0]
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
            1 for e in events if getattr(e.cognitive_phase, "value", e.cognitive_phase) == "confusion"
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

        # ── VRAM hygiene: prevent CUDA memory fragmentation ──
        # Must run EVERY prompt — 50 prompts × 200 tokens of KV cache alloc/free
        # fragments CUDA memory progressively, causing 100x+ slowdown via swap thrashing
        del trace, text
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (i + 1) % 10 == 0:
            gc.collect()

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

    print(f"""
 {C.BOLD}[SYSTEM::METIS]{C.RESET} {C.CYAN}Experiment Results{C.RESET}
 {C.DIM}═══════════════════════════════════════════════════════════════{C.RESET}
""")

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
        ("", 0, 0, 0),  # spacer
        ("▸ TruthfulQA MC1", base.truthfulqa_mc1, metis.truthfulqa_mc1, random_ctrl.truthfulqa_mc1),
        ("▸ TruthfulQA MC2", base.truthfulqa_mc2, metis.truthfulqa_mc2, random_ctrl.truthfulqa_mc2),
        ("▸ MMLU", base.mmlu_accuracy, metis.mmlu_accuracy, random_ctrl.mmlu_accuracy),
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

    # ─── External Benchmark Verdict ───
    if base.truthfulqa_mc1 > 0 or metis.truthfulqa_mc1 > 0:
        print(f"\n{C.BOLD}  External Benchmark Verdict (independent validation):{C.RESET}")
        tqa_delta = metis.truthfulqa_mc1 - base.truthfulqa_mc1
        mmlu_delta = metis.mmlu_accuracy - base.mmlu_accuracy
        tqa_color = C.GREEN if tqa_delta > 0.01 else (C.RED if tqa_delta < -0.01 else C.YELLOW)
        mmlu_color = C.GREEN if mmlu_delta > -0.01 else C.RED
        print(f"    TruthfulQA MC1: METIS {metis.truthfulqa_mc1:.1%} vs Base {base.truthfulqa_mc1:.1%} "
              f"({tqa_color}{tqa_delta:+.1%}{C.RESET})")
        print(f"    MMLU:           METIS {metis.mmlu_accuracy:.1%} vs Base {base.mmlu_accuracy:.1%} "
              f"({mmlu_color}{mmlu_delta:+.1%}{C.RESET})")

        if tqa_delta > 0.01 and mmlu_delta > -0.02:
            print(f"\n    {C.GREEN}{C.BOLD}✓ INDEPENDENTLY VALIDATED: TruthfulQA↑ + MMLU stable{C.RESET}")
            print(f"    {C.GREEN}  METIS cognitive rewards reduce hallucinations without knowledge loss{C.RESET}")
        elif tqa_delta > 0.01 and mmlu_delta <= -0.02:
            print(f"\n    {C.YELLOW}⚠ TruthfulQA improved but MMLU degraded — possible knowledge-safety tradeoff{C.RESET}")
        elif tqa_delta <= 0.01 and mmlu_delta > -0.02:
            print(f"\n    {C.YELLOW}⚠ No TruthfulQA improvement, but MMLU preserved — DPO signal may be too weak{C.RESET}")
        else:
            print(f"\n    {C.RED}✗ Both benchmarks degraded — check DPO hyperparameters{C.RESET}")

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
            "truthfulqa_mc1_delta": round(metis.truthfulqa_mc1 - base.truthfulqa_mc1, 4),
            "mmlu_delta": round(metis.mmlu_accuracy - base.mmlu_accuracy, 4),
        },
    }
    report_path = os.path.join(config.output_dir, "experiment_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  {C.DIM}Report saved to {report_path}{C.RESET}\n")
