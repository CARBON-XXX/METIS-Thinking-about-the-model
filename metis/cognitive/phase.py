"""
METIS Cognitive Phase Detector

Higher-level abstraction above token-level signals: classifies the model's
generation into discrete cognitive phases based on windowed statistics.

Phases (from the model's perspective):
  FLUENT     — Autopilot: generating well-known patterns (very low H, very high conf)
  RECALL     — Knowledge retrieval: recalling stored facts (low H, high conf, FAST)
  REASONING  — Active computation: working through a problem (moderate H, DEEP mix)
  EXPLORATION— Search: uncertain, exploring alternatives (high H, high diversity)
  CONFUSION  — Stuck: unable to find a good path (high H, low diversity, rising mom)

All thresholds are self-calibrating from the session's own signal distribution.
No hardcoded floors — adapts to any model, language, or task difficulty.
"""
from collections import deque
from enum import Enum
from typing import Optional

from ..core.types import CognitiveSignal, Decision


class CognitivePhase(Enum):
    """Discrete cognitive phase of the generation process"""
    FLUENT = "fluent"           # Autopilot: very confident, very low entropy
    RECALL = "recall"           # Knowledge retrieval: confident, low entropy
    REASONING = "reasoning"     # Active computation: moderate uncertainty
    EXPLORATION = "exploration" # Searching: high uncertainty, high diversity
    CONFUSION = "confusion"     # Stuck: high uncertainty, low diversity, rising


class CognitivePhaseDetector:
    """
    Real-time cognitive phase classifier.

    Uses a sliding window of recent CognitiveSignals to classify the current
    generation phase. Self-calibrating: "high" and "low" are defined relative
    to the session's own running statistics (mean ± std).

    Design: O(1) per step (maintains running sums, not full window scan).
    """

    WINDOW = 4  # Classification window size (was 8 — too large for short gen)

    def __init__(self):
        self._buf: deque = deque(maxlen=self.WINDOW)
        # Running session statistics for self-calibration
        self._conf_sum = 0.0
        self._n = 0
        self._phase = CognitivePhase.RECALL
        self._phase_steps = 0  # Steps in current phase

    def observe(self, signal: CognitiveSignal) -> CognitivePhase:
        """
        Observe a new signal and return the current cognitive phase.

        Called once per token, after all signal fields are populated.
        """
        self._buf.append(signal)
        self._n += 1
        self._conf_sum += signal.confidence

        if len(self._buf) < self.WINDOW:
            return self._phase  # Not enough data yet

        # ── Windowed statistics ──
        buf = self._buf
        w = len(buf)
        w_entropy = sum(s.semantic_entropy for s in buf) / w
        w_conf = sum(s.confidence for s in buf) / w
        w_diversity = sum(s.semantic_diversity for s in buf) / w
        w_deep_ratio = sum(1 for s in buf if s.decision == Decision.DEEP) / w
        w_fast_ratio = sum(1 for s in buf if s.decision == Decision.FAST) / w
        w_momentum = sum(s.entropy_momentum for s in buf) / w

        # Use EMA-smoothed z_score from controller (immune to short-sequence
        # mean collapse that plagued the old sess_mean_h approach)
        w_z = sum(s.z_score for s in buf) / w

        # ── Session-level calibration (confidence only) ──
        sess_mean_c = self._conf_sum / self._n
        c_rel = w_conf - sess_mean_c  # Confidence relative to session mean

        # ── Phase classification (priority order) ──
        prev_phase = self._phase

        if w_z > 0.5 and w_diversity < 0.60 and w_momentum >= 0.0:
            # Elevated entropy + low diversity + non-decreasing entropy = stuck
            new_phase = CognitivePhase.CONFUSION
        elif w_z > 0.5 and w_diversity >= 0.60:
            # High entropy + high diversity = searching for answer
            new_phase = CognitivePhase.EXPLORATION
        elif w_deep_ratio > 0.3 or (w_z > 0.3 and w_z <= 0.8):
            # Moderate uncertainty with DEEP decisions = active reasoning
            new_phase = CognitivePhase.REASONING
        elif w_z < -0.5 and w_conf > 0.85 and w_fast_ratio > 0.7:
            # Very low entropy + very high confidence + mostly FAST = autopilot
            new_phase = CognitivePhase.FLUENT
        elif c_rel >= 0 and w_z <= 0.3:
            # Above-average confidence + below-average entropy = recall
            new_phase = CognitivePhase.RECALL
        else:
            new_phase = CognitivePhase.REASONING  # Default: active processing

        if new_phase == prev_phase:
            self._phase_steps += 1
        else:
            self._phase = new_phase
            self._phase_steps = 1

        return self._phase

    @property
    def current_phase(self) -> CognitivePhase:
        return self._phase

    @property
    def phase_duration(self) -> int:
        """Steps in the current phase"""
        return self._phase_steps

    def reset(self) -> None:
        self._buf.clear()
        self._conf_sum = 0.0
        self._n = 0
        self._phase = CognitivePhase.RECALL
        self._phase_steps = 0
