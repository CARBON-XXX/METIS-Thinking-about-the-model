"""
METIS CoT Manager
Dynamic Chain-of-Thought strategy manager

Selects the most appropriate CoT injection strategy based on cognitive signal characteristics.
Instead of uniformly saying "let me think", it precisely guides the model
into different reasoning modes based on the source of uncertainty.

Strategy matrix:
  - STANDARD:      Generic high entropy → "Let me think carefully"
  - CLARIFICATION:  Conceptual ambiguity → "Let me verify the definitions"
  - DECOMPOSITION:  Logical complexity → "Let me break it down step by step"
  - REFLECTION:     Self-contradiction → "Let me re-check"
"""
from typing import List, Optional
import collections

from ..core.types import CognitiveSignal, Decision, CoTStrategy

# =============================================================
# Strategy selection thresholds
# =============================================================

# Oscillation detection: Decision switches exceeding threshold in last N steps -> REFLECTION
# Raised from 4 to 6: Chinese text naturally alternates F/N/D on discourse tokens,
# which is linguistic diversity, not epistemic oscillation.
OSCILLATION_WINDOW = 8
OSCILLATION_THRESHOLD = 6

# Complexity detection: consecutive DEEP exceeding threshold -> DECOMPOSITION
DECOMPOSITION_DEEP_STREAK = 5

# Standard CoT trigger: consecutive DEEP exceeding threshold
# Raised from 3 to 5: 3 consecutive DEEPs are common on CJK connective tokens
# without genuine uncertainty. 5 consecutive DEEPs is a strong signal.
STANDARD_DEEP_STREAK = 5

# Fallback trigger: cumulative high z-score (backup path when DEEP decision is too conservative)
# Count of z-score > Z_TRIGGER in last N steps exceeding threshold -> trigger
# Raised count from 5 to 8 and z-threshold from 1.0 to 1.5:
#   Chinese discourse markers (在/有很多/比如/等) routinely hit z=1.0-1.4
#   without epistemic uncertainty. z>1.5 is a stronger signal of genuine confusion.
HIGH_Z_WINDOW = 12
HIGH_Z_COUNT_THRESHOLD = 8   # >= 8 high-z steps in 12 (was 5)
HIGH_Z_TRIGGER = 1.5          # z-score threshold for "high" (was 1.0 hardcoded)

# High semantic diversity (diffuse probability distribution) + low confidence -> CLARIFICATION
CLARIFICATION_DIVERSITY_THRESHOLD = 0.6
CLARIFICATION_CONFIDENCE_THRESHOLD = 0.3

# CoT injection cooldown: at least N steps between injections
# Prevents repeated injection causing context explosion and latency blowup
# Raised from 15 to 40: 15 Chinese tokens ≈ half a clause, far too short.
# 40 tokens gives the model a full thought before re-evaluating.
COT_COOLDOWN_STEPS = 40

# Max CoT injections per session (hard cap to prevent excessive reasoning latency)
MAX_COT_INJECTIONS_PER_SESSION = 3


class CoTManager:
    """
    Dynamic Chain-of-Thought trigger manager.

    Template-free design: CoTManager only decides WHEN and WHY to trigger
    thinking, NOT what to think. The actual reasoning is generated freely
    by the model inside a <thinking> block.

    Responsibilities:
    1. Determine whether to open a <thinking> block based on cognitive signals
    2. Classify the trigger reason (strategy) for logging/analysis
    3. Manage injection cooldown to prevent excessive thinking blocks
    """

    def __init__(
        self,
        cooldown_steps: int = COT_COOLDOWN_STEPS,
        max_injections: int = MAX_COT_INJECTIONS_PER_SESSION,
    ):
        self._cooldown_steps = cooldown_steps
        self._max_injections = max_injections

        # Session state
        self._decision_history: collections.deque = collections.deque(
            maxlen=OSCILLATION_WINDOW
        )
        self._z_history: collections.deque = collections.deque(
            maxlen=HIGH_Z_WINDOW
        )
        self._consecutive_deep: int = 0
        self._steps_since_last_cot: int = cooldown_steps  # Allow first injection
        self._total_injections: int = 0
        self._last_strategy: CoTStrategy = CoTStrategy.NONE

    def observe(self, signal: CognitiveSignal) -> None:
        """
        Called each step to update internal state.
        Must be called before should_inject / select_strategy.
        """
        self._decision_history.append(signal.decision)
        self._z_history.append(signal.z_score)
        self._steps_since_last_cot += 1

        if signal.decision == Decision.DEEP:
            self._consecutive_deep += 1
        else:
            self._consecutive_deep = 0

    def should_inject(self) -> bool:
        """
        Whether to inject CoT.
        
        Two trigger paths (OR):
        1. Primary: consecutive DEEP >= 3
        2. Fallback: >= 5 steps with z-score > 1.0 in last 12 steps
           (addresses Bonferroni being too conservative causing few DEEP decisions)
        
        Both paths are constrained by cooldown and injection count limits.
        """
        # Hard cap
        if self._total_injections >= self._max_injections:
            return False

        # Cooldown
        if self._steps_since_last_cot < self._cooldown_steps:
            return False

        # Primary path: consecutive DEEP
        if self._consecutive_deep >= STANDARD_DEEP_STREAK:
            return True

        # Fallback path: cumulative high z-score
        if len(self._z_history) >= HIGH_Z_WINDOW:
            high_z_count = sum(1 for z in self._z_history if z > HIGH_Z_TRIGGER)
            if high_z_count >= HIGH_Z_COUNT_THRESHOLD:
                return True

        return False

    def select_strategy(self, signal: CognitiveSignal) -> CoTStrategy:
        """
        Select the most appropriate CoT strategy based on current signal characteristics.

        Priority (high to low):
        1. REFLECTION  — decision oscillation (model flip-flopping between answers)
        2. DECOMPOSITION — sustained deep reasoning (high problem complexity)
        3. CLARIFICATION — high semantic diversity + low confidence (conceptual ambiguity)
        4. STANDARD — generic high entropy
        """
        # 1. Oscillation detection: frequent Decision switches in recent N steps
        if self._detect_oscillation():
            return CoTStrategy.REFLECTION

        # 2. Sustained depth: many consecutive DEEP steps, problem is complex
        if self._consecutive_deep >= DECOMPOSITION_DEEP_STREAK:
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
        """Record a CoT injection"""
        self._total_injections += 1
        self._steps_since_last_cot = 0
        self._last_strategy = strategy

    def reset(self) -> None:
        """Reset session state (called at start of new session)"""
        self._decision_history.clear()
        self._z_history.clear()
        self._consecutive_deep = 0
        self._steps_since_last_cot = self._cooldown_steps  # Allow first injection
        self._total_injections = 0
        self._last_strategy = CoTStrategy.NONE

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
        return {
            "total_injections": self._total_injections,
            "consecutive_deep": self._consecutive_deep,
            "steps_since_last_cot": self._steps_since_last_cot,
            "last_strategy": self._last_strategy.value,
            "remaining_budget": self._max_injections - self._total_injections,
        }
