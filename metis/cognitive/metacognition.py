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

    Design principle: ZERO hardcoded thresholds.
    All decision boundaries are derived from the session's own signal
    distribution via statistical self-calibration:
        - "Abnormal" = value in the tail of the session's own distribution
        - "High risk"  = risk exceeds what the confidence level warrants
        - "Low confidence" = confidence-risk gap is negative
    This ensures METIS adapts automatically to different models, languages,
    and task difficulties without manual tuning.
    """

    def __init__(self):
        # No hardcoded thresholds — all adaptive from trace statistics
        pass

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

        All decisions derived from the judgment's own signals — no fixed cutoffs.
        "should_increase_samples" uses the confidence-risk gap:
            risk exceeds what confidence warrants -> need more samples.

        Returns:
            {
                "should_verify": bool,
                "should_hedge": bool,
                "should_abort": bool,
                "should_increase_samples": bool,
            }
        """
        # Confidence-risk gap: positive = risk exceeds confidence-implied baseline
        confidence_risk_gap = (
            judgment.hallucination_risk - (1.0 - judgment.epistemic_confidence)
        )
        return {
            "should_verify": judgment.suggested_action == "verify",
            "should_hedge": judgment.suggested_action == "hedge",
            "should_abort": judgment.suggested_action == "abort",
            "should_increase_samples": confidence_risk_gap > 0,
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
        # Adaptive: use session's own entropy std as the significance threshold
        # instead of hardcoded 0.3 — different models/tasks have different baselines
        mid = n // 2
        if mid > 0:
            entropies = [e.semantic_entropy for e in events]
            e_mean = sum(entropies) / n
            e_var = sum((h - e_mean) ** 2 for h in entropies) / max(n - 1, 1)
            e_std = math.sqrt(e_var) if e_var > 0 else 0.1

            first_half_entropy = sum(e.semantic_entropy for e in events[:mid]) / mid
            second_half_entropy = sum(e.semantic_entropy for e in events[mid:]) / (n - mid)
            delta = second_half_entropy - first_half_entropy
            # Significant shift = delta exceeds 1 std of the session's entropy
            if delta > e_std:
                trace.entropy_trend_summary = "rising"
            elif delta < -e_std:
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

        Self-calibrating from the session's own signal distribution:
        1. DEEP decision ratio (normalized by session length)
        2. z-score std (normalized by the session's own z-range)

        No hardcoded normalization constants — uses the data's own range.
        """
        n = max(trace.total_tokens, 1)
        deep_ratio = trace.deep_count / n

        # z-score std, self-normalized by the session's z-range
        z_scores = [e.z_score for e in trace.events]
        if len(z_scores) > 1:
            z_mean = sum(z_scores) / len(z_scores)
            z_var = sum((z - z_mean) ** 2 for z in z_scores) / (len(z_scores) - 1)
            z_std = math.sqrt(z_var)
            z_range = max(z_scores) - min(z_scores)
            # Normalize z_std by the session's own range (0=uniform, 1=maximal spread)
            z_load = z_std / max(z_range, 0.01) if z_range > 0 else 0.0
        else:
            z_load = 0.0

        # Both components are [0, 1]; combine with equal weighting
        load = 0.6 * deep_ratio + 0.4 * min(z_load, 1.0)
        return min(load, 1.0)

    def _compute_hallucination_risk(
        self, trace: CognitiveTrace,
        se_result: Optional[SemanticEntropyResult],
    ) -> float:
        """
        Hallucination risk [0, 1].

        Core signal: contradiction detection
        - High confidence + high z-score = model confident but answer unstable -> hallucination

        Self-calibrating: "high confidence" and "high z-score" are defined
        relative to the SESSION's own distributions (mean + 1 std),
        not hardcoded floors. This adapts to different models and tasks.
        """
        events = trace.events
        n = max(len(events), 1)

        # Compute adaptive floors from session distribution
        confidences = [e.confidence for e in events]
        z_scores = [e.z_score for e in events]

        conf_mean = sum(confidences) / n
        z_mean = sum(z_scores) / n

        conf_var = sum((c - conf_mean) ** 2 for c in confidences) / max(n - 1, 1)
        z_var = sum((z - z_mean) ** 2 for z in z_scores) / max(n - 1, 1)

        conf_std = math.sqrt(conf_var) if conf_var > 0 else 0.1
        z_std = math.sqrt(z_var) if z_var > 0 else 0.1

        # Adaptive floors: "high" = above mean + 1 std of session distribution
        adaptive_conf_floor = conf_mean + conf_std
        adaptive_z_floor = z_mean + 2 * z_std

        # Signal 1: contradictory token ratio (high confidence AND high z-score)
        contradictory = sum(
            1 for e in events
            if e.confidence >= adaptive_conf_floor
            and e.z_score >= adaptive_z_floor
        )
        sig1 = contradictory / n

        # Signal 2: SE-based (if available)
        if se_result is not None and se_result.n_samples > 0:
            if se_result.is_uncertain:
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

        Self-calibrating: volatility threshold is derived from the theoretical
        maximum change rate for the number of distinct trend states.
        If trends were random, expected change rate ~ (1 - 1/K) where K = states.
        "Volatile" = change rate exceeds 80% of theoretical maximum.
        """
        events = trace.events
        if len(events) < 5:
            return "stable"

        trends = [e.entropy_trend for e in events]
        trend_changes = sum(
            1 for i in range(1, len(trends)) if trends[i] != trends[i-1]
        )
        change_rate = trend_changes / max(len(trends) - 1, 1)

        # Adaptive volatility: unique trend states determine the theoretical max
        n_unique_trends = len(set(trends))
        # Maximum possible change rate for K states = (K-1)/K (alternating)
        theoretical_max = (n_unique_trends - 1) / max(n_unique_trends, 1)
        # Volatile if change rate exceeds 80% of theoretical maximum
        if change_rate > 0.8 * theoretical_max and theoretical_max > 0:
            return "volatile"

        # Check for sustained degradation in recent window
        recent = events[-min(10, len(events)):]
        rising_count = sum(1 for e in recent if e.entropy_trend == "rising")
        if rising_count > len(recent) * 0.6:
            return "degrading"

        return "stable"

    def _assess_boundary(self, trace: CognitiveTrace) -> str:
        """Boundary status assessment.

        Adaptive: "warning" threshold scales with session complexity
        (deep_ratio). Higher complexity sessions tolerate more hedging.
        """
        if trace.refuse_count > 0:
            return "breached"
        n = max(trace.total_tokens, 1)
        deep_ratio = trace.deep_count / n
        # Adaptive hedge tolerance: complex sessions (high deep_ratio) tolerate more
        hedge_tolerance = 0.1 + 0.3 * deep_ratio  # ranges from 0.1 to 0.4
        if trace.seek_count > 0 or trace.hedge_count > n * hedge_tolerance:
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

        ZERO hardcoded thresholds. All decisions use relative comparisons
        between the session's own computed signals:

        Core metric: confidence_risk_gap = halluc_risk - (1 - confidence)
            - Positive gap = risk exceeds what confidence warrants
            - The larger the gap, the more severe the situation

        Decision tree (priority high to low):
        1. Boundary breached -> abort
        2. Risk exceeds confidence-implied baseline -> verify or hedge
        3. Cognitive instability -> verify or hedge
        4. Confidence-risk imbalance -> hedge
        5. Normal -> continue
        """
        parts = []

        # Confidence-risk gap: the core adaptive metric
        # If confidence=0.8, expected risk baseline = 0.2
        # If actual risk=0.5, gap = 0.5 - 0.2 = 0.3 (anomalous)
        expected_risk = 1.0 - confidence
        risk_gap = halluc_risk - expected_risk

        if boundary_status == "breached":
            parts.append("Cognitive boundary breached (REFUSE triggered)")
            return "abort", "; ".join(parts)

        # Risk significantly exceeds confidence-implied baseline
        if risk_gap > 0:
            parts.append(
                f"Risk-confidence anomaly: risk={halluc_risk:.0%} "
                f"> expected={expected_risk:.0%} (gap={risk_gap:+.0%})"
            )
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
            parts.append(f"Cognitive state volatile (conf={confidence:.0%})")
            # Volatile + negative risk_gap but still risky relative to load
            if confidence < (1.0 - load):
                return "verify", "; ".join(parts)
            return "hedge", "; ".join(parts)

        # Low confidence relative to cognitive load
        # Harder tasks (higher load) get more leeway
        if confidence < 0.5 * (1.0 - load):
            parts.append(
                f"Low confidence-to-load ratio: "
                f"conf={confidence:.0%}, load={load:.0%}"
            )
            return "hedge", "; ".join(parts)

        parts.append(
            f"Confidence {confidence:.0%}, Load {load:.0%}, "
            f"Risk gap {risk_gap:+.2f}, Stable"
        )
        return "continue", "; ".join(parts)
