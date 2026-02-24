"""
METIS Adaptive Controller

Fusion of signal processing and decision theory:
1. Adaptive Forgetting Factor (AFF)
2. Siegmund's Corrected CUSUM - change-point detection
3. Cornish-Fisher Expansion - non-Gaussian quantiles
4. Empirical Bayes - posterior probability
5. Bonferroni Correction - multiple testing correction
6. Decision-Theoretic Thresholds - cost-sensitive thresholds
"""
import math
import threading
import collections
from typing import Optional, List, Dict, Tuple

from .types import Decision, ControllerConfig
from .statistics import SlidingWindowStats

# ── Rust native acceleration (optional) ──
try:
    from metis_native import AdaptiveControllerNative as _NativeController
    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False

_DECISION_FROM_INT = [Decision.FAST, Decision.NORMAL, Decision.DEEP]


class AdaptiveController:
    """
    METIS Adaptive Cognitive Decision Controller.
    
    Core responsibility: transform continuous entropy signals into discrete cognitive decisions.
    
    Academic implementation:
    ═══════════════════════════════════════════════
    1. AFF: λ_t = λ_base / (1 + α·|prediction_error|/σ)
    2. CUSUM: S⁺ = max(0, S⁺ + z - k), alarm if S⁺ > h
       h via Siegmund: h ≈ ln(2k²·ARL₀ + 1) / 2k
    3. Thresholds via Cornish-Fisher:
       z_adj = z + (z²-1)S/6 + (z³-3z)K/24
    4. O1 requires ≥2 of 4 Bonferroni-corrected criteria
    ═══════════════════════════════════════════════
    """

    # ── Constants ──
    AFF_EMA_DECAY = 0.95                # Prediction error EMA decay
    AFF_ERROR_SCALE = 0.1               # Relative error scaling factor
    
    EB_PRIOR_DECAY = 0.995              # Empirical Bayes prior decay
    EB_HIGH_LIKELIHOOD = 0.8            # Likelihood under high-entropy state
    EB_LOW_LIKELIHOOD = 0.2             # Likelihood under low-entropy state
    EB_POSTERIOR_MOMENTUM = 0.9         # Posterior probability momentum
    
    Z_SCORE_STD_FLOOR = 0.15            # Minimum std for z-score computation (prevents over-sensitivity at low entropy)
    SAFE_ENTROPY_THRESHOLD = 0.6        # Absolute safe entropy threshold (ignore z-score below this)
    # H=0.6 approx equals p=0.85. If model is >85% sure, we shouldn't flag it as uncertain
    # regardless of how stable the previous context was.

    CONSECUTIVE_HISTORY_LEN = 50        # Consecutive high-entropy run history length
    BONFERRONI_Z_THRESHOLD = 2.25       # Bonferroni-corrected z threshold (alpha ~ 0.0125)
    
    CB_HISTORY_LEN = 20                 # Circuit breaker history length
    CB_FAST_RATIO = 0.8                 # FAST ratio threshold to trigger circuit breaker
    CB_DEEP_RATIO = 0.4                 # DEEP ratio threshold to trigger circuit breaker

    def __init__(self, config: ControllerConfig = None):
        self._config = config or ControllerConfig()
        c = self._config

        # ── Sliding Window Stats ──
        self._entropy_stats = SlidingWindowStats(c.window_size)
        self._confidence_stats = SlidingWindowStats(c.window_size)

        # ── Adaptive Forgetting Factor ──
        self._base_lambda = c.forgetting_factor
        self._current_lambda = self._base_lambda
        self._entropy_ema = c.cold_start_entropy_mean
        self._entropy_emv = c.cold_start_entropy_std ** 2
        self._prediction_error_ema = 0.0

        # ── Siegmund CUSUM ──
        self._cusum_k = c.cusum_k
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        k = self._cusum_k
        self._cusum_h = math.log(2 * k**2 * c.target_arl0 + 1) / (2 * k)
        self._change_detected = False

        # ── Decision Theory ──
        self._cost_ratio = c.cost_ratio
        self._posterior_threshold = 1.0 / (1.0 + self._cost_ratio)

        # ── Empirical Bayes ──
        self._o1_posterior = 0.1
        self._alpha_prior = 1.0
        self._beta_prior = 9.0
        self._high_entropy_rate = 0.1

        # ── State ──
        self._step_count = 0
        self._is_calibrated = False
        self._lock = threading.Lock()
        self._last_z_score = 0.0  # Cache latest EMA z-score

        # ── Predictive Signals ──
        self._prev_entropy = c.cold_start_entropy_mean
        self._entropy_gradient = 0.0     # d(entropy)/dt
        self._entropy_momentum = 0.0     # EMA of gradient (acceleration)
        self.MOMENTUM_DECAY = 0.9        # Momentum EMA decay

        # ── Consecutive Run Detection ──
        self._consecutive_high = 0
        self._consecutive_threshold = 2
        self._high_entropy_runs: List[int] = []

        # ── Thresholds (cold start) ──
        self._fast_threshold = 1.5      # Below this -> FAST
        self._deep_threshold = 2.0      # Above this -> DEEP (was 5.25 — unreachable)
        self._confidence_threshold = 0.5

        # ── Circuit Breaker ──
        self._circuit_breaker = False
        self._decision_history = collections.deque(maxlen=50)

        # ── Rust native accelerator (if available) ──
        self._native = None
        if _HAS_NATIVE:
            self._native = _NativeController(
                window_size=c.window_size,
                forgetting_factor=c.forgetting_factor,
                cusum_k=c.cusum_k,
                target_arl0=c.target_arl0,
                cost_ratio=c.cost_ratio,
                min_samples=c.min_samples,
                cold_start_mean=c.cold_start_entropy_mean,
                cold_start_std=c.cold_start_entropy_std,
            )

    # ═══════════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════════

    def update(self, entropy: float, confidence: float = None) -> None:
        """
        Input new entropy observation, update all internal statistics.

        Args:
            entropy: Current token's semantic entropy
            confidence: Current token's confidence (optional)
        """
        # Rust fast path
        if self._native is not None:
            self._native.update(entropy, confidence)
            return

        with self._lock:
            self._step_count += 1

            # 1. Window stats
            self._entropy_stats.update(entropy)
            if confidence is not None:
                self._confidence_stats.update(confidence)

            # 2. Adaptive Forgetting Factor
            pred_error = abs(entropy - self._entropy_ema)
            self._prediction_error_ema = (
                self.AFF_EMA_DECAY * self._prediction_error_ema 
                + (1 - self.AFF_EMA_DECAY) * pred_error
            )
            std = math.sqrt(self._entropy_emv) if self._entropy_emv > 0 else 1.0
            rel_error = self._prediction_error_ema / max(std, 0.1)
            self._current_lambda = self._base_lambda / (1.0 + self.AFF_ERROR_SCALE * rel_error)

            alpha = 1.0 - self._current_lambda
            delta = entropy - self._entropy_ema
            self._entropy_ema += alpha * delta
            self._entropy_emv = (1 - alpha) * (self._entropy_emv + alpha * delta ** 2)

            # 3. Entropy gradient & momentum (predictive signals)
            self._entropy_gradient = entropy - self._prev_entropy
            self._entropy_momentum = (
                self.MOMENTUM_DECAY * self._entropy_momentum
                + (1 - self.MOMENTUM_DECAY) * self._entropy_gradient
            )
            self._prev_entropy = entropy

            # 4. Siegmund CUSUM (z-score based on EMA estimates)
            std_curr = max(math.sqrt(self._entropy_emv), self.Z_SCORE_STD_FLOOR)
            z_score = (entropy - self._entropy_ema) / std_curr
            self._last_z_score = z_score  # Cache, shared with get_z_score()
            self._cusum_pos = max(0, self._cusum_pos + z_score - self._cusum_k)
            self._cusum_neg = max(0, self._cusum_neg - z_score - self._cusum_k)
            self._change_detected = (self._cusum_pos > self._cusum_h) or (self._cusum_neg > self._cusum_h)

            # 5. Empirical Bayes
            is_high = z_score > 1.0
            self._alpha_prior = (
                self.EB_PRIOR_DECAY * self._alpha_prior 
                + ((1 - self.EB_PRIOR_DECAY) if is_high else 0)
            )
            self._beta_prior = (
                self.EB_PRIOR_DECAY * self._beta_prior 
                + ((1 - self.EB_PRIOR_DECAY) if not is_high else 0)
            )
            self._high_entropy_rate = self._alpha_prior / (self._alpha_prior + self._beta_prior)

            likelihood_o1 = self.EB_HIGH_LIKELIHOOD if is_high else self.EB_LOW_LIKELIHOOD
            likelihood_norm = self.EB_LOW_LIKELIHOOD if is_high else self.EB_HIGH_LIKELIHOOD
            evidence = likelihood_o1 * self._o1_posterior + likelihood_norm * (1 - self._o1_posterior)
            self._o1_posterior = (likelihood_o1 * self._o1_posterior) / evidence
            self._o1_posterior = (
                self.EB_POSTERIOR_MOMENTUM * self._o1_posterior 
                + (1 - self.EB_POSTERIOR_MOMENTUM) * self._high_entropy_rate
            )

            # 6. Consecutive logic
            if is_high:
                self._consecutive_high += 1
            else:
                if self._consecutive_high > 0:
                    self._high_entropy_runs.append(self._consecutive_high)
                    if len(self._high_entropy_runs) > self.CONSECUTIVE_HISTORY_LEN:
                        self._high_entropy_runs.pop(0)
                    self._learn_consecutive()
                self._consecutive_high = 0

            # 7. Recalibrate thresholds
            if self._step_count >= self._config.min_samples:
                self._recalculate_thresholds()
                self._is_calibrated = True

    def decide(self, entropy: float, confidence: float = None) -> Decision:
        """
        Cognitive decision: map continuous entropy signal to discrete decision.

        Uses Bonferroni-corrected multiple hypothesis testing.
        
        Returns:
            Decision.FAST  -> System 1: automatic output
            Decision.NORMAL -> Standard reasoning
            Decision.DEEP  -> System 2: deliberate reasoning
        """
        # Rust fast path
        if self._native is not None:
            return _DECISION_FROM_INT[self._native.decide(entropy, confidence)]

        with self._lock:
            self._update_circuit_breaker_check()

            if self._circuit_breaker:
                return Decision.NORMAL
            
            # Cold start period: use initial thresholds for simple decision (no multiple testing)
            if not self._is_calibrated:
                if entropy <= self._fast_threshold:
                    if confidence is None or confidence >= 0.5:
                        self._record_decision(Decision.FAST)
                        return Decision.FAST
                # Allow DEEP in cold start — System 2 must not be suppressed
                elif entropy >= self._deep_threshold:
                    self._record_decision(Decision.DEEP)
                    return Decision.DEEP
                return Decision.NORMAL

            # ── FAST Decision (System 1) ──
            if entropy <= self._fast_threshold:
                if confidence is not None and confidence < self._confidence_threshold:
                    return Decision.NORMAL  # Low entropy but low confidence -> don't trust
                self._record_decision(Decision.FAST)
                return Decision.FAST

            # ── DEEP Decision (System 2) ──
            # Bonferroni: m=4 tests, α=0.05 → α_adj=0.0125 → z≈2.25
            criteria_met = 0
            estats = self._entropy_stats.get_stats()
            z = (entropy - estats["mean"]) / max(estats["std"], self.Z_SCORE_STD_FLOOR)

            if z > self.BONFERRONI_Z_THRESHOLD:                     # A. Statistical outlier
                criteria_met += 1
            if self._consecutive_high >= self._consecutive_threshold:  # B. Consecutive high
                criteria_met += 1
            if self._o1_posterior > self._posterior_threshold:       # C. Bayesian posterior
                criteria_met += 1
            if self._change_detected:                               # D. CUSUM alarm
                criteria_met += 1

            if criteria_met >= 2:
                self._record_decision(Decision.DEEP)
                return Decision.DEEP

            self._record_decision(Decision.NORMAL)
            return Decision.NORMAL

    def get_dynamic_z_thresholds(self) -> Tuple[float, float]:
        """
        Dynamically compute boundary thresholds based on entropy distribution shape (skewness/kurtosis).
        
        Returns:
            (z_uncertain, z_unknown)
        """
        if self._native is not None:
            return self._native.get_dynamic_z_thresholds()

        with self._lock:
            estats = self._entropy_stats.get_stats()
            # If too few samples, fall back to default normal distribution thresholds
            if estats["n"] < 30:
                return (1.0, 2.0)
            
            skew = max(-2.0, min(2.0, estats["skew"]))
            kurt = max(-2.0, min(2.0, estats["kurt"]))
            
            # Target: ~85th percentile for UNCERTAIN (std normal z=1.04)
            # Target: ~98th percentile for UNKNOWN (std normal z=2.05)
            # Use Cornish-Fisher expansion to correct for non-Gaussianity
            
            def cf_expansion(z_target):
                # z_adj = z + (z^2 - 1)S/6 + (z^3 - 3z)K/24 - (2z^3 - 5z)S^2/36
                # Simplified (2nd order):
                return z_target + (z_target**2 - 1) * skew / 6 + (z_target**3 - 3*z_target) * kurt / 24

            z_unc = max(0.5, cf_expansion(1.0))
            z_unk = max(1.5, cf_expansion(2.0))
            
            # Ensure monotonicity
            if z_unk <= z_unc:
                z_unk = z_unc + 1.0
                
            return (z_unc, z_unk)

    def get_z_score(self, entropy: float) -> float:
        """
        Get current entropy's z-score (based on EMA estimates).
        
        Unified with CUSUM using the same z-score source:
        - If update() was called -> return cached EMA z-score
        - Otherwise -> compute on-the-fly using EMA mean/std
        
        Returns the SAME z-score used internally by CUSUM/Bayesian posterior.
        This ensures boundary guard decisions are consistent with internal signals.
        """
        if self._native is not None:
            return self._native.get_z_score(entropy)
        if self._step_count > 0:
            # Return value already computed in update() (consistent with CUSUM)
            return self._last_z_score
        # Cold start: compute on-the-fly, clamp to avoid numerical instability
        if entropy < self.SAFE_ENTROPY_THRESHOLD:
            return 0.0
        std = max(math.sqrt(self._entropy_emv), self.Z_SCORE_STD_FLOOR)
        return (entropy - self._entropy_ema) / std

    def reset_session(self) -> None:
        """Reset session state (preserve learned thresholds)"""
        if self._native is not None:
            self._native.reset_session()
            return
        self._consecutive_high = 0
        self._cusum_pos = 0
        self._cusum_neg = 0
        self._change_detected = False
        self._circuit_breaker = False
        self._decision_history.clear()

    @property
    def stats(self) -> Dict[str, float]:
        """Export complete statistics"""
        if self._native is not None:
            return self._native.stats
        estats = self._entropy_stats.get_stats()
        return {
            "entropy_mean": estats["mean"],
            "entropy_std": estats["std"],
            "entropy_skew": estats["skew"],
            "entropy_kurt": estats["kurt"],
            "fast_threshold": self._fast_threshold,
            "deep_threshold": self._deep_threshold,
            "lambda_aff": self._current_lambda,
            "o1_posterior": self._o1_posterior,
            "cusum_pos": self._cusum_pos,
            "cusum_neg": self._cusum_neg,
            "cusum_h": self._cusum_h,
            "change_detected": self._change_detected,
            "is_calibrated": self._is_calibrated,
            "step_count": self._step_count,
        }

    def get_predictive_signals(self) -> Tuple[float, float]:
        """
        Return entropy gradient and momentum for predictive cognitive signals.

        Returns:
            (entropy_gradient, entropy_momentum)
            - gradient: instantaneous d(entropy)/dt
            - momentum: EMA of gradient (captures acceleration/deceleration)
        """
        if self._native is not None:
            return self._native.get_predictive_signals()
        return self._entropy_gradient, self._entropy_momentum

    # ═══════════════════════════════════════════════════════════════
    # Internal
    # ═══════════════════════════════════════════════════════════════

    def _recalculate_thresholds(self) -> None:
        """Cornish-Fisher expansion for non-Gaussian quantiles"""
        estats = self._entropy_stats.get_stats()
        mean, std = estats["mean"], estats["std"]
        skew, kurt = estats["skew"], estats["kurt"]

        # Clamp skew/kurt to prevent CF expansion from exploding
        skew_c = max(-2.0, min(2.0, skew))
        kurt_c = max(-2.0, min(2.0, kurt))

        # FAST threshold: target z = -0.5 (~31st percentile)
        # Use moderate quantile so ~1/3 of low-entropy tokens trigger System 1
        z = -0.5
        z_adj = z + (z**2 - 1) * skew_c / 6 + (z**3 - 3*z) * kurt_c / 24
        fast = mean + z_adj * std

        # DEEP threshold: target z = 1.5 (~6.7th percentile)
        # Use moderate quantile so ~7% of high-entropy tokens trigger System 2
        z = 1.5
        z_adj = z + (z**2 - 1) * skew_c / 6 + (z**3 - 3*z) * kurt_c / 24
        deep = mean + z_adj * std

        # Safety clamps:
        # 1. fast_threshold cannot be negative (entropy >= 0)
        # 2. fast_threshold cannot exceed mean (otherwise most would be FAST)
        # 3. deep_threshold cannot be below mean (otherwise most would be DEEP)
        # 4. The two must not cross
        self._fast_threshold = max(0.0, min(fast, mean))
        self._deep_threshold = max(deep, mean + 0.5 * std)

        # Confidence
        cstats = self._confidence_stats.get_stats()
        if cstats["n"] > 0 and cstats["std"] > 0:
            self._confidence_threshold = cstats["mean"] - 0.5 * cstats["std"]

    def _learn_consecutive(self) -> None:
        if len(self._high_entropy_runs) < 5:
            return
        runs = sorted(self._high_entropy_runs)
        median = runs[len(runs) // 2]
        mad = sorted(abs(r - median) for r in runs)[len(runs) // 2]
        self._consecutive_threshold = max(2, median + mad)

    def _record_decision(self, decision: Decision) -> None:
        self._decision_history.append(decision)

    def _update_circuit_breaker_check(self) -> None:
        if len(self._decision_history) < self.CB_HISTORY_LEN:
            self._circuit_breaker = False
            return
        fast_count = sum(1 for d in self._decision_history if d == Decision.FAST)
        deep_count = sum(1 for d in self._decision_history if d == Decision.DEEP)
        total = len(self._decision_history)
        self._circuit_breaker = (
            (fast_count / total > self.CB_FAST_RATIO) or 
            (deep_count / total > self.CB_DEEP_RATIO)
        )
