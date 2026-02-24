#!/usr/bin/env python3
"""
METIS Cognitive Reward Demo — Terminal Visualization

Demonstrates the full training pipeline without GPU:
1. Generate synthetic traces with varying quality
2. Compute 5-component cognitive rewards
3. GRPO ranking with normalized advantages
4. DPO preference pair generation
5. KTO sample classification

Run:  python demo_reward.py
"""
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metis.core.types import (
    CognitiveTrace, CognitiveEvent, Decision,
    EpistemicState, BoundaryAction,
)
from metis.training.rewards import CognitiveRewardComputer, RewardConfig
from metis.training.grpo import CognitiveGRPO
from metis.training.dataset import PreferencePairGenerator, GeneratorConfig


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
    WHITE  = "\033[97m"
    BG_GREEN  = "\033[42m"
    BG_RED    = "\033[41m"
    BG_YELLOW = "\033[43m"


def colored_reward(val: float) -> str:
    if val >= 0.5:
        return f"{C.GREEN}{val:+.4f}{C.RESET}"
    elif val >= 0.0:
        return f"{C.YELLOW}{val:+.4f}{C.RESET}"
    else:
        return f"{C.RED}{val:+.4f}{C.RESET}"


def bar(val: float, width: int = 20) -> str:
    """Horizontal bar chart for [-1, 1] range."""
    clamped = max(-1.0, min(1.0, val))
    mid = width // 2
    pos = int((clamped + 1.0) / 2.0 * width)

    chars = list("·" * width)
    if pos >= mid:
        for i in range(mid, pos):
            chars[i] = "█"
    else:
        for i in range(pos, mid):
            chars[i] = "░"

    bar_str = "".join(chars)
    if clamped >= 0.5:
        return f"{C.GREEN}{bar_str}{C.RESET}"
    elif clamped >= 0.0:
        return f"{C.YELLOW}{bar_str}{C.RESET}"
    else:
        return f"{C.RED}{bar_str}{C.RESET}"


# ─────────────────────────────────────────────────────
# Synthetic Trace Generator
# ─────────────────────────────────────────────────────

QUALITY_PROFILES = {
    "excellent": {
        "entropy_std": 0.10, "surprise_base": 1.2, "confusion_p": 0.00,
        "confidence": 0.90, "fast_p": 0.75, "uncertain_p": 0.00,
    },
    "good": {
        "entropy_std": 0.20, "surprise_base": 1.8, "confusion_p": 0.02,
        "confidence": 0.82, "fast_p": 0.60, "uncertain_p": 0.05,
    },
    "mediocre": {
        "entropy_std": 0.45, "surprise_base": 3.0, "confusion_p": 0.12,
        "confidence": 0.58, "fast_p": 0.30, "uncertain_p": 0.15,
    },
    "poor": {
        "entropy_std": 0.70, "surprise_base": 4.2, "confusion_p": 0.35,
        "confidence": 0.42, "fast_p": 0.10, "uncertain_p": 0.30,
    },
    "terrible": {
        "entropy_std": 1.00, "surprise_base": 5.5, "confusion_p": 0.55,
        "confidence": 0.30, "fast_p": 0.05, "uncertain_p": 0.50,
    },
}


def make_trace(quality: str, seed: int, n_tokens: int = 40) -> CognitiveTrace:
    rng = random.Random(seed)
    p = QUALITY_PROFILES[quality]
    trace = CognitiveTrace(query=f"demo_{quality}")

    for i in range(n_tokens):
        phase = "confusion" if rng.random() < p["confusion_p"] else rng.choice(
            ["fluent", "recall", "reasoning", "exploration"]
        )
        decision = Decision.FAST if rng.random() < p["fast_p"] else Decision.DEEP
        uncertain = rng.random() < p["uncertain_p"]

        trace.events.append(CognitiveEvent(
            step=i,
            token_entropy=max(0.01, 1.0 + rng.gauss(0, p["entropy_std"])),
            semantic_entropy=max(0.01, 1.2 + rng.gauss(0, p["entropy_std"] * 1.3)),
            confidence=max(0.01, min(0.99, p["confidence"] + rng.gauss(0, 0.05))),
            z_score=rng.gauss(0, 0.5 if quality in ("excellent", "good") else 1.5),
            token_surprise=max(0.1, p["surprise_base"] + rng.gauss(0, 1.0)),
            entropy_gradient=rng.gauss(0, 0.1),
            entropy_momentum=rng.gauss(0, 0.05),
            cognitive_phase=phase,
            decision=decision,
            epistemic_state=EpistemicState.UNCERTAIN if uncertain else EpistemicState.LIKELY,
            boundary_action=BoundaryAction.HEDGE if uncertain and rng.random() < 0.3 else BoundaryAction.GENERATE,
        ))

    trace.total_tokens = n_tokens
    return trace


# ─────────────────────────────────────────────────────
# Demo Sections
# ─────────────────────────────────────────────────────

def demo_reward_components():
    print(f"\n{C.BOLD}{'═' * 70}{C.RESET}")
    print(f"{C.BOLD}  1. COGNITIVE REWARD COMPONENTS{C.RESET}")
    print(f"{C.BOLD}{'═' * 70}{C.RESET}\n")

    print(f"  {C.DIM}R_total = w₁·R_coh + w₂·R_cal + w₃·R_phase + w₄·R_epist + w₅·R_eff{C.RESET}\n")

    computer = CognitiveRewardComputer()

    header = (
        f"  {'Quality':10s} │ {'Total':>8s} │ {'Coherence':>9s} │ "
        f"{'Calibr.':>8s} │ {'Phase':>8s} │ {'Epistemic':>9s} │ {'Effic.':>8s}"
    )
    print(f"{C.BOLD}{header}{C.RESET}")
    print(f"  {'─' * 10}─┼─{'─' * 8}─┼─{'─' * 9}─┼─{'─' * 8}─┼─{'─' * 8}─┼─{'─' * 9}─┼─{'─' * 8}")

    for quality in ["excellent", "good", "mediocre", "poor", "terrible"]:
        trace = make_trace(quality, seed=42)
        r = computer.compute(trace)
        print(
            f"  {quality:10s} │ {colored_reward(r.total)} │ {colored_reward(r.coherence)} │ "
            f"{colored_reward(r.calibration)} │ {colored_reward(r.phase_quality)} │ "
            f"{colored_reward(r.epistemic_honesty)} │ {colored_reward(r.efficiency)}"
        )

    print()


def demo_reward_bars():
    print(f"\n{C.BOLD}{'═' * 70}{C.RESET}")
    print(f"{C.BOLD}  2. REWARD DISTRIBUTION VISUALIZATION{C.RESET}")
    print(f"{C.BOLD}{'═' * 70}{C.RESET}\n")

    computer = CognitiveRewardComputer()

    for quality in ["excellent", "good", "mediocre", "poor", "terrible"]:
        trace = make_trace(quality, seed=42)
        r = computer.compute(trace)
        print(f"  {quality:10s}  {bar(r.total, 40)}  {colored_reward(r.total)}")

    print()
    print(f"  {C.DIM}Legend: █ positive  · zero  ░ negative  │ = center (0.0){C.RESET}\n")


def demo_grpo_ranking():
    print(f"\n{C.BOLD}{'═' * 70}{C.RESET}")
    print(f"{C.BOLD}  3. GRPO RANKING — Group Relative Policy Optimization{C.RESET}")
    print(f"{C.BOLD}{'═' * 70}{C.RESET}\n")

    prompt = "Explain the second law of thermodynamics in simple terms."
    qualities = ["excellent", "good", "mediocre", "poor", "terrible"]
    responses = [
        "Energy always spreads out. Hot flows to cold, never backwards without work.",
        "The entropy of an isolated system tends to increase over time.",
        "Things get more disordered over time, which is why ice melts.",
        "It's about heat and energy and stuff. Everything breaks down eventually.",
        "The second law says you can't make energy from nothing, I think.",
    ]
    traces = [make_trace(q, i * 7) for i, q in enumerate(qualities)]

    grpo = CognitiveGRPO()
    group = grpo.rank_traces(prompt, responses, traces)

    print(f"  {C.CYAN}Prompt:{C.RESET} {prompt}\n")
    print(f"  {C.BOLD}{'Rank':>4s}  {'Advantage':>9s}  {'Reward':>8s}  Response{C.RESET}")
    print(f"  {'─' * 4}  {'─' * 9}  {'─' * 8}  {'─' * 40}")

    medals = ["🥇", "🥈", "🥉", "  ", "  "]
    for s in group.samples:
        adv_color = C.GREEN if s.advantage > 0 else C.RED
        print(
            f"  {medals[s.rank]} {s.rank}  {adv_color}{s.advantage:+.3f}{C.RESET}     "
            f"{colored_reward(s.reward.total)}  {s.response[:50]}"
        )

    print(f"\n  {C.DIM}Reward spread: {group.reward_spread:.4f} "
          f"(higher = stronger training signal){C.RESET}\n")


def demo_dpo_pairs():
    print(f"\n{C.BOLD}{'═' * 70}{C.RESET}")
    print(f"{C.BOLD}  4. DPO PREFERENCE PAIRS{C.RESET}")
    print(f"{C.BOLD}{'═' * 70}{C.RESET}\n")

    grpo = CognitiveGRPO()
    prompts = [
        "What is a black hole?",
        "How does photosynthesis work?",
        "Why is the sky blue?",
    ]

    groups = []
    for i, prompt in enumerate(prompts):
        qualities = ["excellent", "good", "mediocre", "poor"]
        responses = [f"[{q}] answer to: {prompt}" for q in qualities]
        traces = [make_trace(q, i * 100 + j) for j, q in enumerate(qualities)]
        groups.append(grpo.rank_traces(prompt, responses, traces))

    # Best-worst pairs
    gen = PreferencePairGenerator(GeneratorConfig(pair_strategy="best_worst"))
    pairs = gen.from_groups(groups)

    print(f"  Strategy: {C.CYAN}best_worst{C.RESET}  |  {len(pairs)} pairs from {len(groups)} groups\n")
    for p in pairs:
        print(f"  {C.GREEN}✓ chosen:{C.RESET}   {p.chosen[:55]}")
        print(f"  {C.RED}✗ rejected:{C.RESET} {p.rejected[:55]}")
        print(f"  {C.DIM}  margin = {p.reward_margin:.4f}{C.RESET}")
        print()


def demo_kto_samples():
    print(f"\n{C.BOLD}{'═' * 70}{C.RESET}")
    print(f"{C.BOLD}  5. KTO CLASSIFICATION — Desirable vs Undesirable{C.RESET}")
    print(f"{C.BOLD}{'═' * 70}{C.RESET}\n")

    grpo = CognitiveGRPO()
    groups = []
    for i in range(5):
        qualities = ["excellent", "good", "mediocre", "poor", "terrible"]
        responses = [f"Response {i}-{q}" for q in qualities]
        traces = [make_trace(q, i * 50 + j) for j, q in enumerate(qualities)]
        groups.append(grpo.rank_traces(f"Prompt {i}", responses, traces))

    gen = PreferencePairGenerator()
    kto = gen.to_kto(groups)

    desirable = [s for s in kto if s.label]
    undesirable = [s for s in kto if not s.label]

    print(f"  Total: {len(kto)} samples  |  "
          f"{C.GREEN}Desirable: {len(desirable)}{C.RESET}  |  "
          f"{C.RED}Undesirable: {len(undesirable)}{C.RESET}\n")

    print(f"  {C.GREEN}── Desirable (reward > 0.3) ──{C.RESET}")
    for s in desirable[:4]:
        print(f"    {C.GREEN}✓{C.RESET} {colored_reward(s.reward)}  {s.completion}")

    print(f"\n  {C.RED}── Undesirable (reward < -0.1) ──{C.RESET}")
    for s in undesirable[:4]:
        print(f"    {C.RED}✗{C.RESET} {colored_reward(s.reward)}  {s.completion}")

    print()


def demo_pipeline_summary():
    print(f"\n{C.BOLD}{'═' * 70}{C.RESET}")
    print(f"{C.BOLD}  PIPELINE SUMMARY{C.RESET}")
    print(f"{C.BOLD}{'═' * 70}{C.RESET}\n")

    print(f"  {C.CYAN}Traditional RLHF:{C.RESET}")
    print(f"    Human Preference → Reward Model (LLM) → PPO/DPO")
    print(f"    {C.DIM}Expensive, subjective, non-decomposable{C.RESET}\n")

    print(f"  {C.GREEN}METIS Cognitive Rewards:{C.RESET}")
    print(f"    CognitiveTrace → 5-Component Reward → GRPO/DPO/KTO")
    print(f"    {C.DIM}Free, objective, information-theoretic, decomposable{C.RESET}\n")

    print(f"  {C.BOLD}Reward Components:{C.RESET}")
    print(f"    R_coh    = Entropy stability (coherent reasoning)")
    print(f"    R_cal    = Confidence × surprise alignment (anti-hallucination)")
    print(f"    R_phase  = Cognitive phase health (no confusion)")
    print(f"    R_epist  = Epistemic honesty (hedge when unsure)")
    print(f"    R_eff    = Cognitive efficiency (fast when appropriate)\n")

    print(f"  {C.BOLD}Export formats:{C.RESET}  DPO JSONL  |  KTO JSONL  |  GRPO JSONL  |  Reward Model data\n")


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"""{C.GREEN}
███╗   ███╗███████╗████████╗██╗███████╗
████╗ ████║██╔════╝╚══██╔══╝██║██╔════╝
██╔████╔██║█████╗     ██║   ██║███████╗
██║╚██╔╝██║██╔══╝     ██║   ██║╚════██║
██║ ╚═╝ ██║███████╗   ██║   ██║███████║
╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝╚══════╝
{C.RESET}
 {C.BOLD}[SYSTEM::METIS]{C.RESET} {C.CYAN}Cognitive Reward Demo{C.RESET}
 {C.DIM}Information-Theoretic Rewards for GRPO / DPO / KTO{C.RESET}

 > REWARD_COMPUTER.......[{C.GREEN}ONLINE{C.RESET}]
 > GRPO_RANKER...........[{C.GREEN}ACTIVE{C.RESET}]
 > DPO_GENERATOR.........[{C.GREEN}READY{C.RESET}]
 > KTO_CLASSIFIER........[{C.GREEN}READY{C.RESET}]

 root@agi:~$ {C.GREEN}Initializing Demo...{C.RESET}
""")

    demo_reward_components()
    demo_reward_bars()
    demo_grpo_ranking()
    demo_dpo_pairs()
    demo_kto_samples()
    demo_pipeline_summary()

    print(f"  {C.BOLD}{C.CYAN}Done.{C.RESET}\n")
