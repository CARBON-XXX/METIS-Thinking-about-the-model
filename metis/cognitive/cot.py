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


# =============================================================
# Strategy -> Dynamic prompt construction (bilingual)
# =============================================================

# English strategy frames
# IMPORTANT: Templates are ordered by INTENSITY (mild → moderate → strong).
# Selection is driven by z-score magnitude, NOT random.
_COT_FRAMES_EN = {
    CoTStrategy.STANDARD: [
        # Mild: gentle pause
        "\nLet me think about '{ctx}' a bit more carefully.",
        # Moderate: explicit uncertainty
        "\nActually, I'm not fully confident about '{ctx}'. Let me reconsider.",
        # Strong: urgent re-examination
        "\nBefore continuing, I should double-check my reasoning about '{ctx}'.",
    ],
    CoTStrategy.CLARIFICATION: [
        # Mild: curiosity
        "\nLet me make sure I'm clear on what '{ctx}' means here.",
        # Strong: confusion acknowledged
        "\nI realize I might be conflating concepts around '{ctx}'. Let me be precise.",
    ],
    CoTStrategy.DECOMPOSITION: [
        # Mild: structured approach
        "\nThis involves '{ctx}' which is complex. Let me break it down step by step.",
        # Strong: necessity
        "\nTo handle '{ctx}' correctly, I need to decompose this into parts.",
    ],
    CoTStrategy.REFLECTION: [
        # Mild: gentle re-check
        "\nLet me re-check my reasoning about '{ctx}'.",
        # Moderate: doubt
        "\nWait — I said '{ctx}', but does that actually follow? Let me verify.",
        # Strong: something wrong
        "\nSomething about '{ctx}' doesn't feel right. Let me verify my logic.",
    ],
}

_COT_FALLBACK_EN = {
    CoTStrategy.STANDARD: "\nLet me think about this more carefully before continuing.",
    CoTStrategy.CLARIFICATION: "\nI need to clarify the key concepts before proceeding.",
    CoTStrategy.DECOMPOSITION: "\nThis is complex. Let me break it into smaller parts.",
    CoTStrategy.REFLECTION: "\nWait, I should re-examine my reasoning so far.",
}

# Chinese strategy frames
# IMPORTANT: Ordered by INTENSITY (mild → moderate → strong).
_COT_FRAMES_ZH = {
    CoTStrategy.STANDARD: [
        # Mild
        "\n关于'{ctx}'，让我再想想。",
        # Moderate
        "\n实际上，我对'{ctx}'不太确定，让我重新考虑一下。",
        # Strong
        "\n在继续之前，我应该再检查一下关于'{ctx}'的推理。",
    ],
    CoTStrategy.CLARIFICATION: [
        # Mild
        "\n让我确认一下'{ctx}'在这里的含义。",
        # Strong
        "\n我可能混淆了'{ctx}'相关的概念，让我精确地分析。",
    ],
    CoTStrategy.DECOMPOSITION: [
        # Mild
        "\n这涉及到'{ctx}'，让我分步骤来分析。",
        # Strong
        "\n为了正确处理'{ctx}'，我需要把它分解成几个部分。",
    ],
    CoTStrategy.REFLECTION: [
        # Mild: gentle re-check
        "\n让我重新检查一下关于'{ctx}'的推理。",
        # Moderate: something may be off
        "\n等等——我说了'{ctx}'，但这真的成立吗？让我验证。",
        # Strong: clear doubt
        "\n关于'{ctx}'，我感觉有些不对。让我验证一下我的逻辑。",
    ],
}

_COT_FALLBACK_ZH = {
    CoTStrategy.STANDARD: "\n让我在继续之前更仔细地思考这个问题。",
    CoTStrategy.CLARIFICATION: "\n我需要先理清关键概念再继续。",
    CoTStrategy.DECOMPOSITION: "\n这个问题比较复杂，让我分步骤来分析。",
    CoTStrategy.REFLECTION: "\n等等，我应该重新审视一下目前的推理。",
}


def _detect_cjk(text: str) -> bool:
    """Detect if text contains significant CJK characters (Chinese/Japanese/Korean)."""
    if not text:
        return False
    cjk_count = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    return cjk_count > len(text) * 0.1  # >10% CJK chars -> treat as CJK


class CoTManager:
    """
    Dynamic Chain-of-Thought strategy manager.

    Responsibilities:
    1. Determine whether to inject CoT based on cognitive signal sequence
    2. Select the most appropriate CoT strategy
    3. Manage injection cooldown to prevent excessive injection causing latency blowup
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

    def get_prompt(
        self, strategy: CoTStrategy, context: str = "",
        z_score: float = 0.0,
    ) -> str:
        """Dynamically construct strategy prompt based on current generation context.
        
        Auto-detects language from context to select matching templates (EN/ZH).
        
        Template selection is SIGNAL-DRIVEN, not random:
        Templates within each strategy are ordered by intensity (mild -> strong).
        The z-score magnitude determines which intensity level to use:
            z < 1.5  -> mild (index 0)
            z < 2.5  -> moderate (middle index)
            z >= 2.5 -> strong (last index)
        This ensures the injected text's tone matches the actual uncertainty level.
        """
        # Select language-matched templates
        use_zh = _detect_cjk(context)
        frames_dict = _COT_FRAMES_ZH if use_zh else _COT_FRAMES_EN
        fallback_dict = _COT_FALLBACK_ZH if use_zh else _COT_FALLBACK_EN

        # Extract key context snippet (core of the last sentence)
        ctx = self._extract_context(context)
        
        if ctx:
            frames = frames_dict.get(strategy, frames_dict[CoTStrategy.STANDARD])
            template = self._select_by_intensity(frames, z_score)
            return template.format(ctx=ctx)
        else:
            # No context (first injection, no content generated yet)
            return fallback_dict.get(strategy, fallback_dict[CoTStrategy.STANDARD])

    @staticmethod
    def _select_by_intensity(frames: list, z_score: float) -> str:
        """Select template by signal intensity instead of random.
        
        Maps z-score magnitude to template index:
        Templates are pre-ordered [mild, ..., strong].
        Higher z-score -> higher intensity template.
        
        Uses the z-score's position in a natural scale:
            [0, 1.5) -> first template (mild)
            [1.5, 2.5) -> middle template
            [2.5, inf) -> last template (strong)
        """
        n = len(frames)
        if n <= 1:
            return frames[0]
        # Map z-score to [0, n-1] index
        # z < 1.5 -> 0, z in [1.5, 2.5) -> middle, z >= 2.5 -> n-1
        z_abs = abs(z_score)
        if z_abs < 1.5:
            idx = 0
        elif z_abs < 2.5:
            idx = (n - 1) // 2
        else:
            idx = n - 1
        return frames[idx]

    @staticmethod
    def _extract_context(text: str, max_chars: int = 40) -> str:
        """Extract the last meaningful phrase from generated text as context"""
        if not text or len(text.strip()) < 5:
            return ""
        # Take last non-empty segment
        text = text.strip()
        # Split by sentence boundary, take last complete segment
        for sep in ['\n', '. ', ', ', '。', '，']:
            parts = text.rsplit(sep, 1)
            if len(parts) > 1 and len(parts[-1].strip()) >= 5:
                text = parts[-1].strip()
                break
        # Truncate and clean
        if len(text) > max_chars:
            text = text[-max_chars:]
            # Try to start from word boundary
            space_idx = text.find(' ')
            if space_idx > 0 and space_idx < 15:
                text = text[space_idx + 1:]
        return text.strip('.,;:!? \n')

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
