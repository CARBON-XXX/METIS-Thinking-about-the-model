"""
METIS MetacognitiveCore
Introspection analyzer for cognitive processes

Design documented in docs/METIS_METACOGNITION.md:
    Not an external tool, but an intrinsic part of the cognitive process.

MetacognitiveCore consumes CognitiveTrace, outputs MetaJudgment:
    1. Introspect: analyze cognitive trace, assess confidence/load/risk
    2. Regulate: suggest behavioral adjustments based on metacognitive judgment

Core metric computation (all based on real observations, no simulated values):
    - epistemic_confidence: based on confidence distribution + KNOWN/LIKELY ratio
    - cognitive_load: based on DEEP decision ratio + z-score distribution
    - hallucination_risk: based on contradictory signal detection (high confidence + high z-score)
    - stability: based on entropy_trend change frequency
"""
from __future__ import annotations

import math
from typing import Optional, List, Dict
from collections import Counter

from ..core.types import (
    CognitiveTrace,
    CognitiveEvent,
    MetaJudgment,
    Decision,
    EpistemicState,
    BoundaryAction,
    SemanticEntropyResult,
)


class MetacognitiveCore:
    """
    METIS MetacognitiveCore.

    Consumes CognitiveTrace (session-level cognitive trajectory),
    outputs MetaJudgment (metacognitive judgment).

    All computations are based on real observed data, no simulated values.
    """

    def __init__(
        self,
        # Hallucination risk detection: high confidence + high z-score thresholds
        hallucination_confidence_floor: float = 0.6,
        hallucination_z_floor: float = 1.5,
        # Cognitive load threshold
        high_load_deep_ratio: float = 0.2,
        # Stability detection
        volatility_threshold: float = 0.5,
    ):
        self._halluc_conf_floor = hallucination_confidence_floor
        self._halluc_z_floor = hallucination_z_floor
        self._high_load_deep_ratio = high_load_deep_ratio
        self._volatility_threshold = volatility_threshold

    def introspect(
        self,
        trace: CognitiveTrace,
        se_result: Optional[SemanticEntropyResult] = None,
    ) -> MetaJudgment:
        """
        Introspect: analyze cognitive trace, output metacognitive judgment.

        This is the core method of MetacognitiveCore.
        Analyzes the complete cognitive data of an inference session, assessing:
        - Model's confidence in its own answer
        - Cognitive resource consumption
        - Hallucination risk
        - Overall stability

        Args:
            trace: Session-level cognitive trace
            se_result: Optional Kuhn et al. semantic entropy result

        Returns:
            MetaJudgment: Metacognitive judgment
        """
        events = trace.events
        if not events:
            return MetaJudgment(reasoning="Empty trace, cannot introspect")

        n = len(events)

        # -- 1. Aggregate statistics --
        self._aggregate_trace(trace)

        # -- 2. Epistemic confidence --
        epistemic_confidence = self._compute_epistemic_confidence(trace, se_result)

        # -- 3. Cognitive load --
        cognitive_load = self._compute_cognitive_load(trace)

        # -- 4. Hallucination risk --
        hallucination_risk = self._compute_hallucination_risk(trace, se_result)

        # -- 5. Stability analysis --
        stability = self._compute_stability(trace)

        # -- 6. Boundary status --
        boundary_status = self._assess_boundary(trace)

        # -- 7. Dominant epistemic state --
        dominant_state = self._dominant_epistemic_state(trace)

        # -- 8. Suggested action --
        suggested_action, reasoning = self._recommend_action(
            epistemic_confidence, cognitive_load,
            hallucination_risk, stability, boundary_status,
            se_result,
        )

        return MetaJudgment(
            epistemic_confidence=epistemic_confidence,
            cognitive_load=cognitive_load,
            hallucination_risk=hallucination_risk,
            boundary_status=boundary_status,
            dominant_state=dominant_state,
            stability=stability,
            suggested_action=suggested_action,
            reasoning=reasoning,
        )

    def regulate(self, judgment: MetaJudgment) -> Dict[str, bool]:
        """
        Regulate: decide behavioral adjustments based on metacognitive judgment.

        Returns:
            {
                "should_verify": bool,    # Whether to trigger System 2 verification
                "should_hedge": bool,     # Whether to annotate uncertainty
                "should_abort": bool,     # Whether to abort
                "should_increase_samples": bool,  # Whether to increase SE sample count
            }
        """
        return {
            "should_verify": judgment.suggested_action == "verify",
            "should_hedge": judgment.suggested_action == "hedge",
            "should_abort": judgment.suggested_action == "abort",
            "should_increase_samples": (
                judgment.hallucination_risk > 0.5
                or judgment.epistemic_confidence < 0.3
            ),
        }

    # =============================================================
    # Internal computation methods
    # =============================================================

    def _aggregate_trace(self, trace: CognitiveTrace) -> None:
        """Aggregate trace statistics (in-place modification)"""
        events = trace.events
        n = len(events)
        if n == 0:
            return

        trace.fast_count = sum(1 for e in events if e.decision == Decision.FAST)
        trace.deep_count = sum(1 for e in events if e.decision == Decision.DEEP)
        trace.hedge_count = sum(
            1 for e in events if e.boundary_action == BoundaryAction.HEDGE
        )
        trace.seek_count = sum(
            1 for e in events if e.boundary_action == BoundaryAction.SEEK
        )
        trace.refuse_count = sum(
            1 for e in events if e.boundary_action == BoundaryAction.REFUSE
        )
        trace.peak_z_score = max(e.z_score for e in events)
        trace.mean_entropy = sum(e.semantic_entropy for e in events) / n
        trace.mean_confidence = sum(e.confidence for e in events) / n

        # Overall trend: compare second half vs first half mean entropy
        mid = n // 2
        if mid > 0:
            first_half_entropy = sum(e.semantic_entropy for e in events[:mid]) / mid
            second_half_entropy = sum(e.semantic_entropy for e in events[mid:]) / (n - mid)
            delta = second_half_entropy - first_half_entropy
            if delta > 0.3:
                trace.entropy_trend_summary = "rising"
            elif delta < -0.3:
                trace.entropy_trend_summary = "falling"
            else:
                trace.entropy_trend_summary = "stable"

    def _compute_epistemic_confidence(
        self, trace: CognitiveTrace,
        se_result: Optional[SemanticEntropyResult],
    ) -> float:
        """
        Combined epistemic confidence [0, 1].

        Based on three real signals:
        1. Token-level confidence distribution (weight 0.4)
        2. Epistemic state distribution: KNOWN+LIKELY ratio (weight 0.3)
        3. Kuhn et al. SE result (if available) (weight 0.3)
        """
        events = trace.events
        n = len(events)

        # Signal 1: mean token confidence
        sig1 = trace.mean_confidence

        # Signal 2: KNOWN + LIKELY ratio
        known_likely = sum(
            1 for e in events
            if e.epistemic_state in (EpistemicState.KNOWN, EpistemicState.LIKELY)
        )
        sig2 = known_likely / max(n, 1)

        # Signal 3: SE-based (if available)
        if se_result is not None and se_result.n_samples > 0:
            # Higher majority_cluster_prob -> more confident
            sig3 = se_result.majority_cluster_prob
            return 0.35 * sig1 + 0.25 * sig2 + 0.4 * sig3
        else:
            return 0.6 * sig1 + 0.4 * sig2

    def _compute_cognitive_load(self, trace: CognitiveTrace) -> float:
        """
        Cognitive load [0, 1].

        Based on:
        1. DEEP decision ratio
        2. z-score variance (higher -> more entropy fluctuation -> higher load)
        """
        n = max(trace.total_tokens, 1)
        deep_ratio = trace.deep_count / n

        # z-score variance
        z_scores = [e.z_score for e in trace.events]
        if len(z_scores) > 1:
            z_mean = sum(z_scores) / len(z_scores)
            z_var = sum((z - z_mean) ** 2 for z in z_scores) / (len(z_scores) - 1)
            z_std = math.sqrt(z_var)
        else:
            z_std = 0.0

        # Normalize: deep_ratio range [0,1], z_std typically 0~3
        load = 0.6 * min(deep_ratio / max(self._high_load_deep_ratio, 0.01), 1.0) \
             + 0.4 * min(z_std / 2.0, 1.0)
        return min(load, 1.0)

    def _compute_hallucination_risk(
        self, trace: CognitiveTrace,
        se_result: Optional[SemanticEntropyResult],
    ) -> float:
        """
        Hallucination risk [0, 1].

        Core signal: contradiction detection
        - High confidence + high z-score = model confident but answer unstable -> hallucination
        - SE multi-cluster + high majority_prob = model thinks it's sure but actual disagreement -> hallucination
        """
        events = trace.events
        n = max(len(events), 1)

        # Signal 1: contradictory token ratio
        # (confidence > floor AND z-score > floor -> contradiction)
        contradictory = sum(
            1 for e in events
            if e.confidence >= self._halluc_conf_floor
            and e.z_score >= self._halluc_z_floor
        )
        sig1 = contradictory / n

        # Signal 2: SE-based (if available)
        if se_result is not None and se_result.n_samples > 0:
            # Multi-cluster + uncertain -> high hallucination risk
            if se_result.is_uncertain:
                # More clusters -> higher risk
                cluster_risk = min(se_result.n_clusters / se_result.n_samples, 1.0)
                sig2 = 0.5 + 0.5 * cluster_risk
            else:
                sig2 = 0.0
            return 0.5 * sig1 + 0.5 * sig2
        else:
            return sig1

    def _compute_stability(self, trace: CognitiveTrace) -> str:
        """
        Stability assessment.

        Based on entropy_trend change frequency:
        - stable: consistent trend
        - volatile: frequently changing trend (oscillating)
        - degrading: persistently rising trend
        """
        events = trace.events
        if len(events) < 5:
            return "stable"

        trends = [e.entropy_trend for e in events]
        trend_changes = sum(
            1 for i in range(1, len(trends)) if trends[i] != trends[i-1]
        )
        change_rate = trend_changes / max(len(trends) - 1, 1)

        if change_rate > self._volatility_threshold:
            return "volatile"

        # Check for sustained degradation
        recent = events[-min(10, len(events)):]
        rising_count = sum(1 for e in recent if e.entropy_trend == "rising")
        if rising_count > len(recent) * 0.6:
            return "degrading"

        return "stable"

    def _assess_boundary(self, trace: CognitiveTrace) -> str:
        """Boundary status assessment"""
        if trace.refuse_count > 0:
            return "breached"
        if trace.seek_count > 0 or trace.hedge_count > trace.total_tokens * 0.3:
            return "warning"
        return "stable"

    def _dominant_epistemic_state(self, trace: CognitiveTrace) -> EpistemicState:
        """Dominant epistemic state (mode)"""
        if not trace.events:
            return EpistemicState.LIKELY
        counter = Counter(e.epistemic_state for e in trace.events)
        return counter.most_common(1)[0][0]

    def _recommend_action(
        self,
        confidence: float,
        load: float,
        halluc_risk: float,
        stability: str,
        boundary_status: str,
        se_result: Optional[SemanticEntropyResult],
    ) -> tuple:
        """
        Recommend action based on all signals.

        Decision tree (priority high to low):
        1. Boundary breached -> abort
        2. High hallucination risk -> verify (with SE) or hedge
        3. Cognitive instability -> verify
        4. Low confidence -> hedge
        5. Normal -> continue
        """
        parts = []

        if boundary_status == "breached":
            parts.append("Cognitive boundary breached (REFUSE triggered)")
            return "abort", "; ".join(parts)

        if halluc_risk > 0.5:
            parts.append(f"Hallucination risk {halluc_risk:.0%}")
            if se_result is None:
                parts.append("Suggest System 2 verification")
                return "verify", "; ".join(parts)
            else:
                parts.append("High risk despite SE verification, hedging")
                return "hedge", "; ".join(parts)

        if stability == "degrading":
            parts.append("Cognitive state degrading")
            return "verify", "; ".join(parts)

        if stability == "volatile":
            parts.append("Cognitive state volatile")
            if confidence < 0.4:
                return "verify", "; ".join(parts)
            return "hedge", "; ".join(parts)

        if confidence < 0.3:
            parts.append(f"Low confidence {confidence:.0%}")
            return "hedge", "; ".join(parts)

        parts.append(f"Confidence {confidence:.0%}, Load {load:.0%}, Stable")
        return "continue", "; ".join(parts)
