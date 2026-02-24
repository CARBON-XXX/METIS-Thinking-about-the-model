"""
METIS CoT Manager
Dynamic Chain-of-Thought strategy manager

Uses a CUSUM (Cumulative Sum) control chart on cognitive difficulty
to detect when the model needs to "stop and think".

The difficulty score integrates z-score, DEEP decision, and semantic diversity
into a single signal. CUSUM accumulates this signal over time, triggering
a <thinking> block when sustained difficulty is detected.

This replaces four crude counting heuristics (consecutive DEEP, high-z count,
boundary event count, DEEP ratio) with a single principled statistic that
naturally captures both duration and magnitude of cognitive difficulty.

Strategy matrix (for diagnostic classification after trigger):
  - STANDARD:      Generic high entropy → "Let me think carefully"
  - CLARIFICATION:  Conceptual ambiguity → "Let me verify the definitions"
  - DECOMPOSITION:  Logical complexity → "Let me break it down step by step"
  - REFLECTION:     Self-contradiction → "Let me re-check"
"""
from typing import List, Optional
import collections

from ..core.types import CognitiveSignal, Decision, CoTStrategy, BoundaryAction

# ── Rust native acceleration (optional) ──
try:
    from metis_native import CotCusumNative as _NativeCotCusum
    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False

_STRATEGY_FROM_INT = [CoTStrategy.STANDARD, CoTStrategy.REFLECTION, CoTStrategy.DECOMPOSITION, CoTStrategy.CLARIFICATION]
_DECISION_TO_INT = {Decision.FAST: 0, Decision.NORMAL: 1, Decision.DEEP: 2}

# =============================================================
# CUSUM parameters for cognitive difficulty detection
# =============================================================
#
# Formula: S(t) = max(0, S(t-1) + difficulty(t) - k)
#   difficulty(t) = (max(0, z) + deep_bonus) * sd
#
# Lower k and h than boundary guard → CoT triggers BEFORE HEDGE,
# giving the model a chance to reason through difficulty.

COT_CUSUM_K = 0.3           # Allowance (lower than boundary = more sensitive)
COT_CUSUM_H = 4.0           # Trigger threshold
COT_CUSUM_DECAY = 0.9       # Decay when z < 0 (slower than boundary)
COT_DEEP_BONUS = 0.3        # Bonus contribution for DEEP decisions

# Predictive trigger: entropy momentum (acceleration) early-warning
# When entropy is accelerating upward (positive momentum), trigger CoT
# even before CUSUM reaches full threshold.
COT_MOMENTUM_H = 2.0        # Momentum accumulator trigger threshold
COT_CUSUM_EARLY = COT_CUSUM_H * 0.5  # CUSUM must be at least 50% for momentum trigger

# =============================================================
# Strategy selection thresholds (diagnostic, NOT trigger-related)
# =============================================================

# Oscillation detection: Decision switches in recent window -> REFLECTION
OSCILLATION_WINDOW = 8
OSCILLATION_THRESHOLD = 6

# Complexity detection: consecutive DEEP -> DECOMPOSITION strategy
DECOMPOSITION_DEEP_STREAK = 5

# Conceptual ambiguity: high semantic diversity + low confidence -> CLARIFICATION
CLARIFICATION_DIVERSITY_THRESHOLD = 0.6
CLARIFICATION_CONFIDENCE_THRESHOLD = 0.3

# =============================================================
# Injection limits
# =============================================================

# CoT injection cooldown: at least N steps between injections
COT_COOLDOWN_STEPS = 40

# Max CoT injections per session (hard cap)
MAX_COT_INJECTIONS_PER_SESSION = 3


class CoTManager:
    """
    Dynamic Chain-of-Thought trigger manager — CUSUM-based.

    Uses a cognitive difficulty CUSUM to detect when the model needs to
    "stop and think". This replaces four crude counting heuristics with
    a single principled statistic.

    Two trigger paths:
    1. CUSUM trigger: S(t) >= threshold (classic cumulative detection)
    2. Momentum trigger: entropy acceleration sustained + CUSUM >= 50%
       (predictive — intervenes BEFORE full difficulty builds up)

    Template-free: CoTManager decides WHEN and WHY to think.
    The model generates its own reasoning inside the <thinking> block.
    """

    def __init__(
        self,
        cooldown_steps: int = COT_COOLDOWN_STEPS,
        max_injections: int = MAX_COT_INJECTIONS_PER_SESSION,
    ):
        self._cooldown_steps = cooldown_steps
        self._max_injections = max_injections

        # CUSUM state
        self._difficulty_cusum: float = 0.0

        # Momentum-based predictive trigger state
        self._momentum_acc: float = 0.0     # Accumulated positive momentum
        self._momentum_steps: int = 0       # Consecutive positive-momentum steps

        # Strategy selection state (kept for diagnostic classification)
        self._decision_history: collections.deque = collections.deque(
            maxlen=OSCILLATION_WINDOW
        )
        self._consecutive_deep: int = 0

        # Injection tracking
        self._steps_since_last_cot: int = cooldown_steps  # Allow first injection
        self._total_injections: int = 0
        self._last_strategy: CoTStrategy = CoTStrategy.NONE

        # Rust native accelerator (if available)
        self._native = None
        if _HAS_NATIVE:
            self._native = _NativeCotCusum(
                cooldown=cooldown_steps,
                max_inj=max_injections,
                cusum_k=COT_CUSUM_K,
                cusum_h=COT_CUSUM_H,
                cusum_decay=COT_CUSUM_DECAY,
                deep_bonus=COT_DEEP_BONUS,
                momentum_h=COT_MOMENTUM_H,
                osc_window=OSCILLATION_WINDOW,
                osc_threshold=OSCILLATION_THRESHOLD,
            )

    def observe(self, signal: CognitiveSignal) -> None:
        """
        Called each step to update internal state.
        Must be called before should_inject / select_strategy.
        """
        # Rust fast path
        if self._native is not None:
            dec_int = _DECISION_TO_INT.get(signal.decision, 1)
            self._native.observe(
                signal.z_score, signal.semantic_diversity,
                dec_int, signal.entropy_momentum,
            )
            # Keep Python-side last_strategy tracking
            self._steps_since_last_cot = self._native.steps_since_last_cot
            self._total_injections = self._native.total_injections
            return

        # Python fallback
        # Strategy selection tracking
        self._decision_history.append(signal.decision)
        if signal.decision == Decision.DEEP:
            self._consecutive_deep += 1
        else:
            self._consecutive_deep = 0
        self._steps_since_last_cot += 1

        # ── Difficulty CUSUM update ──
        z = signal.z_score
        sd = signal.semantic_diversity
        deep_bonus = COT_DEEP_BONUS if signal.decision == Decision.DEEP else 0.0

        if z > 0 or deep_bonus > 0:
            # Positive z or DEEP decision contributes to difficulty
            z_contrib = max(0.0, z)
            increment = (z_contrib + deep_bonus) * sd - COT_CUSUM_K
            self._difficulty_cusum = max(0.0, self._difficulty_cusum + increment)
        elif z < 0:
            # Confident token: decay accumulated difficulty
            self._difficulty_cusum *= COT_CUSUM_DECAY

        # ── Momentum accumulator (predictive early-warning) ──
        # Positive entropy_momentum = entropy is accelerating upward
        # Sustained acceleration predicts imminent difficulty spike
        mom = signal.entropy_momentum
        if mom > 0:
            self._momentum_acc += mom
            self._momentum_steps += 1
        else:
            # Decay on non-positive momentum (don't hard-reset)
            self._momentum_acc *= 0.8
            self._momentum_steps = 0

    def should_inject(self) -> bool:
        """
        Whether to trigger a <thinking> block.

        Single trigger: difficulty CUSUM >= threshold.
        Constrained by cooldown and injection count limits.
        """
        if self._native is not None:
            return self._native.should_inject()

        if self._total_injections >= self._max_injections:
            return False

        if self._steps_since_last_cot < self._cooldown_steps:
            return False

        # Path 1: Classic CUSUM trigger (full difficulty detected)
        if self._difficulty_cusum >= COT_CUSUM_H:
            return True

        # Path 2: Predictive momentum trigger (entropy accelerating)
        # Requires: CUSUM already at 50%+ AND sustained momentum accumulation
        # This catches rising-difficulty situations BEFORE CUSUM reaches threshold
        if (
            self._difficulty_cusum >= COT_CUSUM_EARLY
            and self._momentum_acc >= COT_MOMENTUM_H
            and self._momentum_steps >= 3
        ):
            return True

        return False

    def select_strategy(self, signal: CognitiveSignal) -> CoTStrategy:
        """
        Select the most appropriate CoT strategy based on current signal
        characteristics AND cognitive phase.

        Priority (high to low):
        1. REFLECTION  — oscillation OR confusion phase (model stuck/flip-flopping)
        2. DECOMPOSITION — sustained depth OR exploration phase (need structure)
        3. CLARIFICATION — high diversity + low confidence (conceptual ambiguity)
        4. STANDARD — generic high entropy
        """
        if self._native is not None:
            phase = signal.cognitive_phase if signal.cognitive_phase else ""
            idx = self._native.select_strategy(
                signal.semantic_diversity, signal.confidence, phase
            )
            return _STRATEGY_FROM_INT[idx]

        phase = signal.cognitive_phase

        # 1. Oscillation or CONFUSION phase: model is stuck, needs self-reflection
        if self._detect_oscillation() or phase == "confusion":
            return CoTStrategy.REFLECTION

        # 2. Sustained depth or EXPLORATION phase: problem needs decomposition
        if self._consecutive_deep >= DECOMPOSITION_DEEP_STREAK or phase == "exploration":
            return CoTStrategy.DECOMPOSITION

        # 3. Conceptual ambiguity: high semantic diversity + low confidence
        if (
            signal.semantic_diversity > CLARIFICATION_DIVERSITY_THRESHOLD
            and signal.confidence < CLARIFICATION_CONFIDENCE_THRESHOLD
        ):
            return CoTStrategy.CLARIFICATION

        # 4. Default
        return CoTStrategy.STANDARD

    def record_injection(self, strategy: CoTStrategy) -> None:
        """Record a CoT injection and reset CUSUM"""
        self._total_injections += 1
        self._steps_since_last_cot = 0
        self._difficulty_cusum = 0.0  # Reset CUSUM after injection
        self._momentum_acc = 0.0
        self._momentum_steps = 0
        self._last_strategy = strategy
        if self._native is not None:
            self._native.record_injection()

    def reset(self) -> None:
        """Reset session state (called at start of new session)"""
        self._decision_history.clear()
        self._consecutive_deep = 0
        self._difficulty_cusum = 0.0
        self._momentum_acc = 0.0
        self._momentum_steps = 0
        self._steps_since_last_cot = self._cooldown_steps  # Allow first injection
        self._total_injections = 0
        self._last_strategy = CoTStrategy.NONE
        if self._native is not None:
            self._native.reset()

    def _detect_oscillation(self) -> bool:
        """
        Detect decision oscillation: within the last OSCILLATION_WINDOW steps,
        adjacent Decision switches exceeding the threshold.

        E.g.: FAST -> DEEP -> FAST -> DEEP -> ...
        Indicates model oscillating between known and unknown, needs REFLECTION.
        """
        if len(self._decision_history) < OSCILLATION_WINDOW:
            return False

        switches = 0
        prev = None
        for d in self._decision_history:
            if prev is not None and d != prev:
                switches += 1
            prev = d

        return switches >= OSCILLATION_THRESHOLD

    @property
    def stats(self) -> dict:
        if self._native is not None:
            return {
                "total_injections": self._native.total_injections,
                "difficulty_cusum": round(self._native.difficulty_cusum_val, 2),
                "momentum_acc": round(self._native.momentum_acc_val, 2),
                "momentum_steps": 0,
                "consecutive_deep": self._native.consecutive_deep,
                "steps_since_last_cot": self._native.steps_since_last_cot,
                "last_strategy": self._last_strategy.value,
                "remaining_budget": self._native.remaining_budget,
            }
        return {
            "total_injections": self._total_injections,
            "difficulty_cusum": round(self._difficulty_cusum, 2),
            "momentum_acc": round(self._momentum_acc, 2),
            "momentum_steps": self._momentum_steps,
            "consecutive_deep": self._consecutive_deep,
            "steps_since_last_cot": self._steps_since_last_cot,
            "last_strategy": self._last_strategy.value,
            "remaining_budget": self._max_injections - self._total_injections,
        }
