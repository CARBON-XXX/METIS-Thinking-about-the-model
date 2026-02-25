"""
METIS Cognitive Reward Computer

Transforms CognitiveTrace into scalar reward signals for RLHF/GRPO/DPO training.

Unlike LLM-as-judge reward models, these rewards are:
1. Information-theoretic — based on entropy, surprise, calibration
2. Objective — no subjective human/LLM preference involved
3. Decomposable — each reward component has clear mathematical meaning
4. Cheap — no extra LLM inference needed, computed from existing trace

Reward Components:
═══════════════════════════════════════════════════════════════════
R_total = w₁·R_coh + w₂·R_cal + w₃·R_phase + w₄·R_epist + w₅·R_eff + w₆·R_think

R_coh   : Coherence       — entropy stability (low variance = smooth reasoning)
R_cal   : Calibration     — confidence-surprise alignment (penalize overconfidence)
R_phase : Phase Quality   — penalize confusion, reward natural reasoning arcs
R_epist : Epistemic Honor — appropriate uncertainty expression
R_eff   : Efficiency      — don't overthink simple things
R_think : Thinking Quality — penalize thinking-answer overlap (recitation detection)
═══════════════════════════════════════════════════════════════════

All component rewards are normalized to [-1, 1].
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..core.types import (
    CognitiveTrace,
    CognitiveEvent,
    Decision,
    EpistemicState,
    BoundaryAction,
)

# ── Rust native reward accelerator (10-50x faster) ──
try:
    from metis_native import RewardComputerNative as _NativeRewardComputer
    _HAS_NATIVE_REWARDS = True
except ImportError:
    _HAS_NATIVE_REWARDS = False

# Enum → int mappings for Rust interface
_DECISION_TO_INT = {Decision.FAST: 0, Decision.NORMAL: 1, Decision.DEEP: 2}
_PHASE_TO_INT = {"fluent": 0, "recall": 1, "reasoning": 2, "exploration": 3, "confusion": 4}
_ESTATE_TO_INT = {
    EpistemicState.KNOWN: 0, EpistemicState.LIKELY: 1,
    EpistemicState.UNCERTAIN: 2, EpistemicState.UNKNOWN: 3,
}
_BACTION_TO_INT = {
    BoundaryAction.GENERATE: 0, BoundaryAction.HEDGE: 1,
    BoundaryAction.SEEK: 2, BoundaryAction.REFUSE: 3,
}


# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────

@dataclass
class RewardConfig:
    """Weights and hyperparameters for cognitive reward computation."""
    # Component weights (must sum to 1.0)
    w_coherence: float = 0.20
    w_calibration: float = 0.34
    w_phase: float = 0.23
    w_epistemic: float = 0.15
    w_efficiency: float = 0.08

    # Coherence v2: windowed CV + entropy floor
    coherence_cv_scale: float = 0.5       # Reduced: 2.0 was over-penalizing DPO models
    coherence_window: int = 16            # Sliding window for local CV
    entropy_floor: float = 0.3            # Min mean entropy before penalty
    entropy_floor_penalty: float = 2.0    # Penalty scale for entropy collapse

    # Calibration: surprise baseline for "overconfident" detection
    calibration_surprise_baseline: float = 3.0  # bits

    # Phase v2: continuous arc quality scoring (no binary confusion gate)
    phase_confusion_penalty: float = 2.0
    phase_monotone_penalty: float = 0.5   # Penalty for all-same phase (no reasoning)
    phase_arc_bonus: float = 0.3          # Bonus for natural reasoning arcs
    phase_oscillation_penalty: float = 1.0

    # Epistemic v2: surprise-conditional scoring (not just label-based)
    epistemic_surprise_weight: float = 0.6  # Weight for surprise-confidence divergence
    epistemic_label_weight: float = 0.4    # Weight for epistemic state labels
    epistemic_unknown_penalty: float = 3.0

    # Efficiency: target FAST ratio (domain-dependent)
    efficiency_target_fast: float = 0.3

    # Thinking quality: penalize thinking-answer overlap (recitation detection)
    w_thinking_quality: float = 0.0        # 0.0 = disabled by default (no thinking in training)
    thinking_overlap_threshold: float = 0.3  # Jaccard overlap above this → penalty
    thinking_ngram_size: int = 2            # Character n-gram size for overlap

    # Length penalty: slight preference for concise responses
    length_penalty_threshold: int = 512     # tokens
    length_penalty_scale: float = 0.001     # per extra token


# ─────────────────────────────────────────────────────────
# Reward Breakdown
# ─────────────────────────────────────────────────────────

@dataclass
class RewardBreakdown:
    """Decomposed reward with per-component scores and diagnostics."""
    # Total reward (weighted sum)
    total: float = 0.0

    # Component scores (each in [-1, 1])
    coherence: float = 0.0
    calibration: float = 0.0
    phase_quality: float = 0.0
    epistemic_honesty: float = 0.0
    efficiency: float = 0.0
    thinking_quality: float = 0.0

    # Length penalty (subtracted from total)
    length_penalty: float = 0.0

    # Diagnostics
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        return {
            "total": round(self.total, 4),
            "coherence": round(self.coherence, 4),
            "calibration": round(self.calibration, 4),
            "phase_quality": round(self.phase_quality, 4),
            "epistemic_honesty": round(self.epistemic_honesty, 4),
            "efficiency": round(self.efficiency, 4),
            "thinking_quality": round(self.thinking_quality, 4),
            "length_penalty": round(self.length_penalty, 4),
            **{k: round(v, 4) for k, v in self.diagnostics.items()},
        }


# ─────────────────────────────────────────────────────────
# Core Reward Computer
# ─────────────────────────────────────────────────────────

class CognitiveRewardComputer:
    """
    Compute cognitive reward from a CognitiveTrace.

    Usage:
        computer = CognitiveRewardComputer()
        trace = metis.get_trace()  # After inference
        reward = computer.compute(trace)
        print(reward.total, reward.to_dict())
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self._config = config or RewardConfig()
        self._native = _NativeRewardComputer() if _HAS_NATIVE_REWARDS else None

    def compute(self, trace: CognitiveTrace) -> RewardBreakdown:
        """
        Compute full reward breakdown from a CognitiveTrace.

        Args:
            trace: Complete cognitive trace from an inference session

        Returns:
            RewardBreakdown with total score and per-component breakdown
        """
        events = trace.events
        if len(events) < 2:
            return RewardBreakdown(diagnostics={"error": "too_few_events"})

        # ── Rust fast path: ~10-50x faster for large traces ──
        if self._native is not None:
            return self._compute_native(events, trace)

        # ── Python fallback ──
        # Degeneration guard: if entropy has near-zero variance AND mean
        # confidence is extremely high, output is likely degenerate
        # (e.g., "TheTheThe..." repeated tokens)
        entropies = [e.semantic_entropy for e in events]
        mean_h = sum(entropies) / len(entropies)
        var_h = sum((h - mean_h) ** 2 for h in entropies) / len(entropies)
        decisions = set(e.decision for e in events)
        if var_h < 0.001 and len(decisions) <= 1:
            return RewardBreakdown(
                total=-1.0, coherence=-1.0,
                diagnostics={"degenerate": 1.0, "entropy_var": var_h},
            )

        cfg = self._config
        breakdown = RewardBreakdown()

        # Compute each component
        breakdown.coherence = self._reward_coherence(events)
        breakdown.calibration = self._reward_calibration(events)
        breakdown.phase_quality = self._reward_phase_quality(events)
        breakdown.epistemic_honesty = self._reward_epistemic_honesty(events)
        breakdown.efficiency = self._reward_efficiency(events, trace)

        # Completeness bonus: reward detailed, high-quality responses
        # Prevents "silence is gold" gaming where model outputs EOS early
        n = len(events)
        completeness_bonus = 0.0
        if n > 30 and breakdown.coherence > 0 and breakdown.calibration > 0:
            completeness_bonus = min(0.25, (n - 30) * 0.005)
            breakdown.diagnostics["completeness_bonus"] = round(completeness_bonus, 4)

        # Length penalty (extreme length only)
        if n > cfg.length_penalty_threshold:
            breakdown.length_penalty = (n - cfg.length_penalty_threshold) * cfg.length_penalty_scale

        # Thinking quality (only if text available on trace)
        if trace.thinking_text and trace.answer_text:
            breakdown.thinking_quality = self._reward_thinking_quality(
                trace.thinking_text, trace.answer_text
            )

        # Weighted total
        breakdown.total = (
            cfg.w_coherence * breakdown.coherence
            + cfg.w_calibration * breakdown.calibration
            + cfg.w_phase * breakdown.phase_quality
            + cfg.w_epistemic * breakdown.epistemic_honesty
            + cfg.w_efficiency * breakdown.efficiency
            + cfg.w_thinking_quality * breakdown.thinking_quality
            + completeness_bonus
            - breakdown.length_penalty
        )

        return breakdown

    def _compute_native(self, events: List[CognitiveEvent], trace: CognitiveTrace) -> RewardBreakdown:
        """Rust-accelerated reward computation."""
        # Extract arrays for Rust
        entropies = [e.semantic_entropy for e in events]
        token_entropies = [e.token_entropy for e in events]
        confidences = [e.confidence for e in events]
        surprises = [e.token_surprise for e in events]
        z_scores = [e.z_score for e in events]
        decisions = [_DECISION_TO_INT.get(e.decision, 1) for e in events]
        phases = [_PHASE_TO_INT.get(
            getattr(e.cognitive_phase, 'value', e.cognitive_phase), 1
        ) for e in events]
        epistemic_states = [_ESTATE_TO_INT.get(e.epistemic_state, 1) for e in events]
        boundary_actions = [_BACTION_TO_INT.get(e.boundary_action, 0) for e in events]

        r = self._native.compute(
            entropies, token_entropies, confidences, surprises, z_scores,
            decisions, phases, epistemic_states, boundary_actions,
        )

        breakdown = RewardBreakdown(
            total=r["total"],
            coherence=r["coherence"],
            calibration=r["calibration"],
            phase_quality=r["phase_quality"],
            epistemic_honesty=r["epistemic_honesty"],
            efficiency=r["efficiency"],
            length_penalty=r.get("length_penalty", 0.0),
        )
        if r.get("completeness_bonus", 0.0) > 0:
            breakdown.diagnostics["completeness_bonus"] = round(r["completeness_bonus"], 4)
        if r.get("degenerate", 0.0) > 0:
            breakdown.diagnostics["degenerate"] = 1.0
            if "entropy_var" in r:
                breakdown.diagnostics["entropy_var"] = r["entropy_var"]

        # Thinking quality: computed in Python (Rust has no text access)
        cfg = self._config
        if cfg.w_thinking_quality > 0 and trace.thinking_text and trace.answer_text:
            breakdown.thinking_quality = self._reward_thinking_quality(
                trace.thinking_text, trace.answer_text
            )
            breakdown.total += cfg.w_thinking_quality * breakdown.thinking_quality

        return breakdown

    # ═══════════════════════════════════════════════════════
    # R₁: Coherence — entropy stability
    # ═══════════════════════════════════════════════════════

    def _reward_coherence(self, events: List[CognitiveEvent]) -> float:
        """
        Coherence reward v2: windowed local smoothness + entropy floor guard.

        Two sub-signals:

        1. Windowed CV [−1.0, 0.6]:
           - Compute CV in sliding windows of size W, then average
           - Rewards LOCAL smoothness (adjacent tokens coherent)
           - Unlike global CV, does NOT reward degenerate flat distributions
           - A natural response with smooth local transitions scores well
             even if global entropy varies across topics

        2. Entropy floor guard [−0.4, 0]:
           - If mean entropy < floor → penalty (distribution is collapsing)
           - This creates a soft constraint against mode collapse
           - Prevents coherence reward from competing with calibration/efficiency

        R_coh = windowed_score + floor_penalty ∈ [-1.0, 0.6]
        """
        entropies = [e.semantic_entropy for e in events]
        n = len(entropies)
        cfg = self._config

        mean_h = sum(entropies) / n

        # ── Sub-signal 1: Windowed CV ──
        window = min(cfg.coherence_window, n)
        if window < 2:
            windowed_score = 0.0
        else:
            local_cvs = []
            for start in range(0, n - window + 1, window // 2):  # 50% overlap
                chunk = entropies[start : start + window]
                w_mean = sum(chunk) / len(chunk)
                if w_mean < 1e-6:
                    local_cvs.append(0.0)
                    continue
                w_var = sum((h - w_mean) ** 2 for h in chunk) / len(chunk)
                local_cvs.append(math.sqrt(w_var) / w_mean)

            if local_cvs:
                avg_cv = sum(local_cvs) / len(local_cvs)
            else:
                avg_cv = 0.0

            # Softplus smoothing: log1p(cv) compresses high CV → diminishing penalty
            # Linear was too harsh: avg_cv=1.5 gave -0.15, now gives +0.14
            windowed_score = 0.6 - cfg.coherence_cv_scale * math.log1p(avg_cv)
            windowed_score = max(-1.0, min(0.6, windowed_score))

        # ── Sub-signal 2: Entropy floor guard ──
        floor_penalty = 0.0
        if mean_h < cfg.entropy_floor:
            # Penalty grows as entropy drops below floor
            deficit = (cfg.entropy_floor - mean_h) / max(cfg.entropy_floor, 1e-6)
            floor_penalty = cfg.entropy_floor_penalty * deficit
            floor_penalty = min(floor_penalty, 0.4)  # Cap at -0.4

        reward = windowed_score - floor_penalty
        return max(-1.0, min(1.0, reward))

    # ═══════════════════════════════════════════════════════
    # R₂: Calibration — confidence-surprise alignment
    # ═══════════════════════════════════════════════════════

    def _reward_calibration(self, events: List[CognitiveEvent]) -> float:
        """
        Calibration reward: penalize overconfident hallucination.

        Core idea: if model is confident (high p(token)) but surprise is high
        (actually chose a low-probability token from the full distribution),
        something is wrong — the model is overconfident about uncertain content.

        Metric: -mean(confidence * max(0, surprise - baseline))
        - Only penalizes when surprise exceeds baseline (normal surprise is fine)
        - Weighted by confidence (only bad when model is also confident)

        This is a novel reward signal that captures the gap between
        token-level confidence and sequence-level prediction error.
        """
        baseline = self._config.calibration_surprise_baseline
        n = len(events)

        # Compute miscalibration: confidence * excess_surprise
        miscal_sum = 0.0
        miscal_count = 0
        for e in events:
            excess = e.token_surprise - baseline
            if excess > 0:
                miscal_sum += e.confidence * excess
                miscal_count += 1

        if miscal_count == 0:
            return 1.0  # No miscalibration events

        # Normalize by total events (not just miscalibrated ones)
        # This way, a few miscalibrated tokens in a long response aren't catastrophic
        mean_miscal = miscal_sum / n

        # Map to [-1, 1]: 0 miscalibration → +1, high miscalibration → -1
        # Empirically, mean_miscal > 0.5 is very bad
        reward = 1.0 - 4.0 * mean_miscal
        return max(-1.0, min(1.0, reward))

    # ═══════════════════════════════════════════════════════
    # R₃: Phase Quality — cognitive phase health
    # ═══════════════════════════════════════════════════════

    def _reward_phase_quality(self, events: List[CognitiveEvent]) -> float:
        """
        Phase quality reward v2: continuous arc scoring (no dead nodes).

        Three sub-signals (each contributes to a continuous score):

        1. Phase diversity score [0, 0.4]:
           - Monotone generation (all "fluent") is penalized
           - Uses normalized phase entropy: H(phases) / log(n_possible_phases)
           - This breaks the 1.0 saturation problem

        2. Reasoning arc completeness [0, 0.3]:
           - Detects natural arcs: exploration → reasoning → resolution
           - Measured by entropy gradient trajectory (descending trend after peak)
           - Rewards sequences where entropy rises (exploration), then falls (resolution)

        3. Confusion penalty [-0.7, 0]:
           - Still penalizes confusion phases, but as one of three signals
           - Also penalizes rapid oscillation (>40% transition rate)

        Total R_phase ∈ [-1.0, 0.7] — never saturates at 1.0 unless
        perfect arc + diversity + zero confusion (extremely rare).
        """
        n = len(events)
        cfg = self._config

        # ── Sub-signal 1: Phase diversity (H(phase) / log(K)) ──
        phase_counts: Dict[str, int] = {}
        for e in events:
            p = e.cognitive_phase
            phase_counts[p] = phase_counts.get(p, 0) + 1

        n_unique = len(phase_counts)
        if n_unique <= 1:
            # Monotone: single phase throughout → penalty
            diversity_score = -cfg.phase_monotone_penalty
        else:
            # Shannon entropy of phase distribution
            phase_entropy = 0.0
            for count in phase_counts.values():
                p_i = count / n
                if p_i > 0:
                    phase_entropy -= p_i * math.log(p_i + 1e-12)
            # Normalize by log(K) where K = number of possible phases
            # Use observed unique phases as denominator (conservative)
            max_entropy = math.log(max(n_unique, 2))
            normalized = phase_entropy / max_entropy  # ∈ [0, 1]
            diversity_score = 0.4 * normalized  # ∈ [0, 0.4]

        # ── Sub-signal 2: Reasoning arc completeness ──
        # Detect entropy trajectory: does it rise (exploration) then fall (resolution)?
        if n >= 8:
            # Split trace into thirds
            third = n // 3
            entropies = [e.semantic_entropy for e in events]
            h_early = sum(entropies[:third]) / max(third, 1)
            h_mid = sum(entropies[third:2*third]) / max(third, 1)
            h_late = sum(entropies[2*third:]) / max(n - 2*third, 1)

            # Ideal arc: h_mid > h_early (exploration) AND h_late < h_mid (resolution)
            has_exploration = h_mid > h_early * 1.05  # 5% rise = exploration
            has_resolution = h_late < h_mid * 0.95    # 5% drop = resolution
            arc_score = cfg.phase_arc_bonus * (
                (0.5 if has_exploration else 0.0)
                + (0.5 if has_resolution else 0.0)
            )  # ∈ [0, phase_arc_bonus]
        else:
            arc_score = 0.0  # Too short to detect arc

        # ── Sub-signal 3: Confusion penalty with Cognitive Recovery Bonus ──
        confusion_count = sum(1 for e in events if getattr(e.cognitive_phase, "value", e.cognitive_phase) == "confusion")
        confusion_ratio = confusion_count / n

        transitions = 0
        for i in range(1, n):
            if events[i].cognitive_phase != events[i - 1].cognitive_phase:
                transitions += 1
        oscillation_rate = transitions / max(n - 1, 1)

        # Detect cognitive recovery: confusion with DEEP → entropy drop → reasoning/recall
        recovered = False
        if confusion_count > 0:
            # Check if any confusion segment contains DEEP and is followed by resolution
            in_confusion = False
            had_deep_in_confusion = False
            for i, e in enumerate(events):
                phase_val = getattr(e.cognitive_phase, "value", e.cognitive_phase)
                if phase_val == "confusion":
                    in_confusion = True
                    if e.decision == Decision.DEEP:
                        had_deep_in_confusion = True
                elif in_confusion:
                    # Exited confusion — check if it resolved
                    if had_deep_in_confusion and phase_val in ("reasoning", "recall", "fluent"):
                        recovered = True
                        break
                    in_confusion = False
                    had_deep_in_confusion = False

        penalty = 0.0
        recovery_bonus = 0.0
        if recovered:
            # Successful cognitive recovery: confusion is investment, not failure
            recovery_bonus = min(0.3, confusion_ratio * 2.0)
        else:
            # Unresolved confusion: penalize (but less harshly than before)
            penalty += cfg.phase_confusion_penalty * confusion_ratio * 0.5
        penalty += cfg.phase_oscillation_penalty * max(0, oscillation_rate - 0.4)

        # ── Combine ──
        reward = diversity_score + arc_score + recovery_bonus - penalty
        return max(-1.0, min(1.0, reward))

    # ═══════════════════════════════════════════════════════
    # R₄: Epistemic Honesty — appropriate uncertainty handling
    # ═══════════════════════════════════════════════════════

    def _reward_epistemic_honesty(self, events: List[CognitiveEvent]) -> float:
        """
        Epistemic honesty reward v2: surprise-conditional continuous scoring.

        Problem with v1: label-based scoring (EpistemicState counts) is a
        confound — both METIS and Random DPO improve equally because the
        improvement comes from DPO format, not cognitive signals.

        v2 uses two weighted sub-signals:

        1. Surprise-confidence divergence (60% weight):
           - Continuous metric: |confidence - (1 - normalized_surprise)|
           - Measures gap between model's expressed confidence and
             information-theoretic surprise on each token
           - Low divergence = well-calibrated epistemic behavior
           - THIS is METIS-specific: depends on actual entropy/surprise signals
           - DPO format alone cannot improve this without METIS trace data

        2. Label-based honesty (40% weight):
           - Kept for interpretability but downweighted
           - Same logic as v1 (hedge when uncertain, confident when likely)

        R_epist = w_s * surprise_score + w_l * label_score
        """
        n = len(events)
        cfg = self._config
        w_s = cfg.epistemic_surprise_weight  # 0.6
        w_l = cfg.epistemic_label_weight     # 0.4

        # ── Sub-signal 1: Surprise-confidence divergence ──
        # For each token: ideal confidence ≈ 1 - normalized_surprise
        # normalized_surprise = min(token_surprise / baseline, 1.0)
        baseline = cfg.calibration_surprise_baseline
        divergence_sum = 0.0
        for e in events:
            norm_surprise = min(e.token_surprise / max(baseline, 1e-6), 1.0)
            ideal_confidence = 1.0 - norm_surprise
            divergence_sum += abs(e.confidence - ideal_confidence)

        mean_divergence = divergence_sum / max(n, 1)
        # Map: 0 divergence → +1, high divergence → −1
        surprise_score = 1.0 - 3.0 * mean_divergence
        surprise_score = max(-1.0, min(1.0, surprise_score))

        # ── Sub-signal 2: Label-based honesty (same as v1) ──
        honest_count = 0
        dishonest_penalty = 0.0

        for e in events:
            is_uncertain = e.epistemic_state in (
                EpistemicState.UNCERTAIN,
                EpistemicState.UNKNOWN,
            )
            is_confident_output = e.confidence > 0.7

            if is_uncertain and is_confident_output:
                weight = cfg.epistemic_unknown_penalty if e.epistemic_state == EpistemicState.UNKNOWN else 1.0
                dishonest_penalty += weight
            elif is_uncertain and e.boundary_action == BoundaryAction.SEEK:
                honest_count += 2.0  # Best honesty: actively seeking clarification under uncertainty
            elif is_uncertain and e.decision == Decision.DEEP:
                honest_count += 1.5  # Good honesty: actively reasoning under uncertainty
            elif is_uncertain and e.boundary_action in (
                BoundaryAction.HEDGE,
                BoundaryAction.REFUSE,
            ):
                honest_count += 1.0  # Acceptable: admitting ignorance without attempting resolution
            elif not is_uncertain and is_confident_output:
                honest_count += 1

        honest_ratio = honest_count / max(n, 1)
        dishonest_ratio = dishonest_penalty / max(n, 1)
        label_score = honest_ratio - dishonest_ratio
        label_score = max(-1.0, min(1.0, label_score))

        # ── Combine ──
        reward = w_s * surprise_score + w_l * label_score
        return max(-1.0, min(1.0, reward))

    # ═══════════════════════════════════════════════════════
    # R₅: Efficiency — cognitive resource allocation
    # ═══════════════════════════════════════════════════════

    def _reward_efficiency(
        self, events: List[CognitiveEvent], trace: CognitiveTrace
    ) -> float:
        """
        Efficiency reward: appropriate cognitive resource allocation.

        Penalize:
        - Too much DEEP reasoning on simple content (overthinking)
        - DEEP reasoning that doesn't resolve (futile deliberation)

        Reward:
        - High FAST ratio when no boundary alarms follow (quick + correct)
        - DEEP reasoning that leads to resolution (useful deliberation)

        This teaches the model to be "thinking fast and slow" appropriately —
        System 1 for easy stuff, System 2 only when genuinely needed.
        """
        n = len(events)
        if n == 0:
            return 0.0

        deep_count = sum(1 for e in events if e.decision == Decision.DEEP)

        # ── Decision appropriateness: does the decision match uncertainty? ──
        # FAST  + z ≤ 0.0  → appropriate (confident → fast output)
        # DEEP  + z > 0.3  → appropriate (uncertain → deliberate thinking)
        # NORMAL            → always appropriate (default processing)
        # FAST  + z > 0.5  → inappropriate (hasty under uncertainty)
        # DEEP  + z ≤ -0.2 → inappropriate (overthinking when confident)
        appropriate = 0
        for e in events:
            if e.decision == Decision.FAST:
                if e.z_score <= 0.0:
                    appropriate += 1        # confident + fast = ideal
                elif e.z_score <= 0.5:
                    appropriate += 0.5      # borderline, partial credit
                # z > 0.5 + FAST → 0 credit (hasty)
            elif e.decision == Decision.DEEP:
                if e.z_score > 0.3:
                    appropriate += 1        # uncertain + thinking = ideal
                elif e.z_score > -0.2:
                    appropriate += 0.5      # borderline, partial credit
                # z ≤ -0.2 + DEEP → 0 credit (overthinking)
            else:  # Decision.NORMAL
                if e.z_score > 0.4:
                    appropriate += 0.0      # cognitive laziness: should be DEEP
                elif e.z_score > 0.0:
                    appropriate += 0.35     # borderline: mild penalty
                else:
                    appropriate += 0.7      # low-entropy default: acceptable

        appropriateness = appropriate / n   # [0, 1]

        # ── DEEP resolution bonus: reward productive System 2 ──
        resolved_deep = 0
        for i in range(len(events) - 1):
            if events[i].decision == Decision.DEEP:
                lookahead = events[i + 1 : min(i + 4, len(events))]
                if lookahead and any(
                    e.token_entropy < events[i].token_entropy * 0.7
                    for e in lookahead
                ):
                    resolved_deep += 1
        resolution_rate = resolved_deep / max(deep_count, 1)
        resolution_bonus = resolution_rate * 0.3

        # ── Length factor: prevent short-sequence gaming ──
        length_factor = min(1.0, n / 40.0)

        reward = (appropriateness + resolution_bonus) * length_factor
        return max(-1.0, min(1.0, reward))

    # ═══════════════════════════════════════════════════════
    # R₆: Thinking Quality — thinking-answer overlap penalty
    # ═══════════════════════════════════════════════════════

    def _reward_thinking_quality(
        self, thinking_text: str, answer_text: str
    ) -> float:
        """
        Thinking quality reward: penalize recitation in thinking blocks.

        Core idea: if <thinking> content has high n-gram overlap with the
        final answer, the model is just reciting/dumping knowledge in the
        thinking block instead of genuinely analyzing. Real analysis should
        produce DIFFERENT text from the answer — meta-reasoning about the
        problem, not a rough draft of the answer itself.

        Algorithm:
          1. Extract character-level n-grams from both texts
          2. Compute Jaccard similarity: |A ∩ B| / |A ∪ B|
          3. High overlap (>threshold) → penalty (recitation)
          4. Low overlap (<threshold/2) → bonus (genuine analysis)

        Character-level n-grams work for both Chinese (no word boundaries)
        and English. Bigrams are the default (cfg.thinking_ngram_size=2).

        Returns:
            float in [-1.0, 1.0]
            - Positive: thinking is genuinely different from answer (good)
            - Negative: thinking overlaps heavily with answer (recitation)
            - 0.0: no thinking text available
        """
        cfg = self._config

        # Clean inputs: strip whitespace, tags, scaffold remnants
        t = thinking_text.strip()
        a = answer_text.strip()

        if not t or not a:
            return 0.0

        # Remove <thinking>/<highlight> and other tags from thinking
        t = re.sub(r'<[^>]+>', '', t).strip()
        # Remove scaffold remnants from thinking
        t = re.sub(r'"[^"]{0,120}"\s*vs\s*"[^"]{0,80}"', '', t).strip()

        if len(t) < 4 or len(a) < 4:
            return 0.0  # Too short to meaningfully compare

        # Extract character-level n-grams
        n = cfg.thinking_ngram_size
        t_ngrams = set()
        for i in range(len(t) - n + 1):
            t_ngrams.add(t[i:i + n])

        a_ngrams = set()
        for i in range(len(a) - n + 1):
            a_ngrams.add(a[i:i + n])

        if not t_ngrams or not a_ngrams:
            return 0.0

        # Jaccard similarity: |intersection| / |union|
        intersection = len(t_ngrams & a_ngrams)
        union = len(t_ngrams | a_ngrams)
        jaccard = intersection / max(union, 1)

        threshold = cfg.thinking_overlap_threshold  # default 0.3

        if jaccard > threshold:
            # High overlap → recitation penalty
            # Scale: 0.3 → 0.0, 0.6 → -0.5, 0.9 → -1.0
            excess = (jaccard - threshold) / (1.0 - threshold)
            reward = -min(1.0, excess * 1.5)
        else:
            # Low overlap → genuine thinking bonus
            # Scale: 0.3 → 0.0, 0.15 → +0.5, 0.0 → +1.0
            deficit = (threshold - jaccard) / max(threshold, 1e-6)
            reward = min(1.0, deficit)

        return max(-1.0, min(1.0, reward))
