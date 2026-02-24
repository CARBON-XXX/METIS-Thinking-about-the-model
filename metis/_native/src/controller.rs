//! METIS Adaptive Controller — Rust accelerated.
//!
//! Combines:
//!   1. Adaptive Forgetting Factor (AFF)
//!   2. Siegmund's Corrected CUSUM — change-point detection
//!   3. Cornish-Fisher Expansion — non-Gaussian quantiles
//!   4. Empirical Bayes — posterior probability
//!   5. Bonferroni Correction — multiple testing correction
//!   6. Decision-Theoretic Thresholds — cost-sensitive thresholds
//!
//! Decision codes: 0=FAST, 1=NORMAL, 2=DEEP

use pyo3::prelude::*;
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────
// Internal sliding window (not exposed to Python)
// Avoids PyO3 boundary crossing overhead on every stats call
// ─────────────────────────────────────────────────────────

struct InternalWindow {
    buf: Vec<f64>,
    head: usize,
    len: usize,
    cap: usize,
}

impl InternalWindow {
    fn new(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
            head: 0,
            len: 0,
            cap,
        }
    }

    fn push(&mut self, x: f64) {
        if self.len < self.cap {
            self.buf.push(x);
            self.len += 1;
            self.head = self.len % self.cap;
        } else {
            self.buf[self.head] = x;
            self.head = (self.head + 1) % self.cap;
        }
    }

    fn count(&self) -> usize {
        self.len
    }

    fn mean(&self) -> f64 {
        if self.len == 0 {
            return 0.0;
        }
        self.buf[..self.len].iter().sum::<f64>() / self.len as f64
    }

    fn std_bessel(&self) -> f64 {
        if self.len < 2 {
            return 0.1;
        }
        let n = self.len as f64;
        let m = self.mean();
        let m2: f64 = self.buf[..self.len].iter().map(|&x| (x - m) * (x - m)).sum();
        let var = m2 / (n - 1.0);
        if var > 1e-10 {
            var.sqrt()
        } else {
            0.01
        }
    }

    /// Returns (mean, std, skew, kurt, n) — single-pass two-pass algorithm.
    fn full_stats(&self) -> (f64, f64, f64, f64, usize) {
        let n = self.len;
        if n < 2 {
            return (0.0, 0.1, 0.0, 0.0, n);
        }
        let nf = n as f64;
        let data = &self.buf[..n];
        let mean = data.iter().sum::<f64>() / nf;

        let (mut m2, mut m3, mut m4) = (0.0f64, 0.0f64, 0.0f64);
        for &x in data {
            let d = x - mean;
            let d2 = d * d;
            m2 += d2;
            m3 += d2 * d;
            m4 += d2 * d2;
        }

        // Std (Bessel correction)
        let var = m2 / (nf - 1.0);
        let std_val = if var > 1e-10 { var.sqrt() } else { 0.01 };

        // Population std for moment standardization
        let std_pop = {
            let v = m2 / nf;
            if v > 0.0 {
                v.sqrt()
            } else {
                0.01
            }
        };

        // Skewness (Fisher-Pearson, bias-corrected)
        let skew = if n >= 3 && std_pop > 1e-6 {
            let g1 = (m3 / nf) / std_pop.powi(3);
            g1 * (nf * (nf - 1.0)).sqrt() / (nf - 2.0)
        } else {
            0.0
        };

        // Kurtosis (Fisher excess, bias-corrected)
        let kurt = if n >= 4 && std_pop > 1e-6 {
            let g2 = (m4 / nf) / std_pop.powi(4) - 3.0;
            ((nf + 1.0) * g2 + 6.0) * (nf - 1.0) / ((nf - 2.0) * (nf - 3.0))
        } else {
            0.0
        };

        (mean, std_val, skew, kurt, n)
    }

    fn reset(&mut self) {
        self.buf.clear();
        self.head = 0;
        self.len = 0;
    }
}

// ─────────────────────────────────────────────────────────
// Cornish-Fisher expansion (shared helper)
// ─────────────────────────────────────────────────────────

/// Cornish-Fisher expansion: non-Gaussian quantile adjustment.
///
/// z_adj = z + (z²-1)S/6 + (z³-3z)K/24
pub(crate) fn cornish_fisher(z: f64, skew: f64, kurt: f64) -> f64 {
    z + (z * z - 1.0) * skew / 6.0 + (z * z * z - 3.0 * z) * kurt / 24.0
}

// ─────────────────────────────────────────────────────────
// Constants matching Python AdaptiveController
// ─────────────────────────────────────────────────────────

const AFF_EMA_DECAY: f64 = 0.95;
const AFF_ERROR_SCALE: f64 = 0.1;
const EB_PRIOR_DECAY: f64 = 0.995;
const EB_HIGH_LIKELIHOOD: f64 = 0.8;
const EB_LOW_LIKELIHOOD: f64 = 0.2;
const EB_POSTERIOR_MOMENTUM: f64 = 0.9;
const Z_SCORE_STD_FLOOR: f64 = 0.15;
const SAFE_ENTROPY_THRESHOLD: f64 = 0.6;
const CONSECUTIVE_HISTORY_LEN: usize = 50;
const BONFERRONI_Z: f64 = 2.25;
const CB_HISTORY_LEN: usize = 20;
const CB_FAST_RATIO: f64 = 0.8;
const CB_DEEP_RATIO: f64 = 0.4;
const MOMENTUM_DECAY: f64 = 0.9;
const DECISION_BUF_LEN: usize = 50;

// ─────────────────────────────────────────────────────────
// AdaptiveControllerNative
// ─────────────────────────────────────────────────────────

#[pyclass(module = "metis_native")]
pub struct AdaptiveControllerNative {
    // Sliding windows (internal — zero PyO3 overhead)
    entropy_win: InternalWindow,
    confidence_win: InternalWindow,
    // Adaptive Forgetting Factor
    base_lambda: f64,
    current_lambda: f64,
    entropy_ema: f64,
    entropy_emv: f64,
    pred_error_ema: f64,
    // Siegmund CUSUM
    cusum_k: f64,
    cusum_pos: f64,
    cusum_neg: f64,
    cusum_h: f64,
    change_detected: bool,
    // Empirical Bayes
    o1_posterior: f64,
    alpha_prior: f64,
    beta_prior: f64,
    high_entropy_rate: f64,
    posterior_threshold: f64,
    // Predictive signals
    prev_entropy: f64,
    entropy_gradient: f64,
    entropy_momentum: f64,
    // Consecutive run detection
    consec_high: usize,
    consec_threshold: usize,
    high_runs: Vec<usize>,
    // Thresholds
    fast_threshold: f64,
    deep_threshold: f64,
    conf_threshold: f64,
    // Circuit breaker (circular buffer of decisions)
    dec_buf: Vec<u8>,
    dec_head: usize,
    dec_count: usize,
    circuit_breaker: bool,
    // Meta
    step_count: usize,
    is_calibrated: bool,
    min_samples: usize,
    last_z: f64,
}

#[pymethods]
impl AdaptiveControllerNative {
    #[new]
    #[pyo3(signature = (
        window_size = 500,
        forgetting_factor = 0.995,
        cusum_k = 0.5,
        target_arl0 = 200,
        cost_ratio = 5.0,
        min_samples = 10,
        cold_start_mean = 1.0,
        cold_start_std = 1.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        window_size: usize,
        forgetting_factor: f64,
        cusum_k: f64,
        target_arl0: usize,
        cost_ratio: f64,
        min_samples: usize,
        cold_start_mean: f64,
        cold_start_std: f64,
    ) -> Self {
        // Siegmund threshold: h ≈ ln(2k²·ARL₀ + 1) / (2k)
        let cusum_h =
            (2.0 * cusum_k * cusum_k * target_arl0 as f64 + 1.0).ln() / (2.0 * cusum_k);

        Self {
            entropy_win: InternalWindow::new(window_size),
            confidence_win: InternalWindow::new(window_size),
            base_lambda: forgetting_factor,
            current_lambda: forgetting_factor,
            entropy_ema: cold_start_mean,
            entropy_emv: cold_start_std * cold_start_std,
            pred_error_ema: 0.0,
            cusum_k,
            cusum_pos: 0.0,
            cusum_neg: 0.0,
            cusum_h,
            change_detected: false,
            o1_posterior: 0.1,
            alpha_prior: 1.0,
            beta_prior: 9.0,
            high_entropy_rate: 0.1,
            posterior_threshold: 1.0 / (1.0 + cost_ratio),
            prev_entropy: cold_start_mean,
            entropy_gradient: 0.0,
            entropy_momentum: 0.0,
            consec_high: 0,
            consec_threshold: 2,
            high_runs: Vec::new(),
            fast_threshold: 1.5,
            deep_threshold: 2.0,
            conf_threshold: 0.5,
            dec_buf: vec![0u8; DECISION_BUF_LEN],
            dec_head: 0,
            dec_count: 0,
            circuit_breaker: false,
            step_count: 0,
            is_calibrated: false,
            min_samples,
            last_z: 0.0,
        }
    }

    // ═══════════════════════════════════════════════════════
    // Public API
    // ═══════════════════════════════════════════════════════

    /// Update all internal statistics with a new entropy observation.
    #[pyo3(signature = (entropy, confidence = None))]
    fn update(&mut self, entropy: f64, confidence: Option<f64>) {
        self.step_count += 1;

        // 1. Window stats
        self.entropy_win.push(entropy);
        if let Some(c) = confidence {
            self.confidence_win.push(c);
        }

        // 2. Adaptive Forgetting Factor
        let pred_error = (entropy - self.entropy_ema).abs();
        self.pred_error_ema =
            AFF_EMA_DECAY * self.pred_error_ema + (1.0 - AFF_EMA_DECAY) * pred_error;
        let std = if self.entropy_emv > 0.0 {
            self.entropy_emv.sqrt()
        } else {
            1.0
        };
        let rel_error = self.pred_error_ema / std.max(0.1);
        self.current_lambda = self.base_lambda / (1.0 + AFF_ERROR_SCALE * rel_error);

        let alpha = 1.0 - self.current_lambda;
        let delta = entropy - self.entropy_ema;
        self.entropy_ema += alpha * delta;
        self.entropy_emv = (1.0 - alpha) * (self.entropy_emv + alpha * delta * delta);

        // 3. Entropy gradient & momentum
        self.entropy_gradient = entropy - self.prev_entropy;
        self.entropy_momentum =
            MOMENTUM_DECAY * self.entropy_momentum + (1.0 - MOMENTUM_DECAY) * self.entropy_gradient;
        self.prev_entropy = entropy;

        // 4. Siegmund CUSUM (z-score from EMA estimates)
        let std_curr = self.entropy_emv.sqrt().max(Z_SCORE_STD_FLOOR);
        let z = (entropy - self.entropy_ema) / std_curr;
        self.last_z = z;
        self.cusum_pos = (self.cusum_pos + z - self.cusum_k).max(0.0);
        self.cusum_neg = (self.cusum_neg - z - self.cusum_k).max(0.0);
        self.change_detected = self.cusum_pos > self.cusum_h || self.cusum_neg > self.cusum_h;

        // 5. Empirical Bayes
        let is_high = z > 1.0;
        self.alpha_prior = EB_PRIOR_DECAY * self.alpha_prior
            + if is_high { 1.0 - EB_PRIOR_DECAY } else { 0.0 };
        self.beta_prior = EB_PRIOR_DECAY * self.beta_prior
            + if !is_high { 1.0 - EB_PRIOR_DECAY } else { 0.0 };
        self.high_entropy_rate = self.alpha_prior / (self.alpha_prior + self.beta_prior);

        let lk_o1 = if is_high {
            EB_HIGH_LIKELIHOOD
        } else {
            EB_LOW_LIKELIHOOD
        };
        let lk_norm = if is_high {
            EB_LOW_LIKELIHOOD
        } else {
            EB_HIGH_LIKELIHOOD
        };
        let evidence = lk_o1 * self.o1_posterior + lk_norm * (1.0 - self.o1_posterior);
        self.o1_posterior = (lk_o1 * self.o1_posterior) / evidence;
        self.o1_posterior = EB_POSTERIOR_MOMENTUM * self.o1_posterior
            + (1.0 - EB_POSTERIOR_MOMENTUM) * self.high_entropy_rate;

        // 6. Consecutive run detection
        if is_high {
            self.consec_high += 1;
        } else {
            if self.consec_high > 0 {
                self.high_runs.push(self.consec_high);
                if self.high_runs.len() > CONSECUTIVE_HISTORY_LEN {
                    self.high_runs.remove(0);
                }
                self.learn_consecutive();
            }
            self.consec_high = 0;
        }

        // 7. Recalibrate thresholds via Cornish-Fisher
        if self.step_count >= self.min_samples {
            self.recalculate_thresholds();
            self.is_calibrated = true;
        }
    }

    /// Cognitive decision: map entropy signal to 0=FAST, 1=NORMAL, 2=DEEP.
    ///
    /// Uses Bonferroni-corrected multiple hypothesis testing (m=4, α_adj≈0.0125).
    #[pyo3(signature = (entropy, confidence = None))]
    fn decide(&mut self, entropy: f64, confidence: Option<f64>) -> u8 {
        self.update_circuit_breaker();
        if self.circuit_breaker {
            return 1; // NORMAL — circuit breaker engaged
        }

        // Cold start: simple threshold decision
        if !self.is_calibrated {
            if entropy <= self.fast_threshold {
                if confidence.is_none() || confidence.unwrap_or(0.5) >= 0.5 {
                    self.record_decision(0);
                    return 0; // FAST
                }
            } else if entropy >= self.deep_threshold {
                self.record_decision(2);
                return 2; // DEEP — System 2 must not be suppressed
            }
            return 1; // NORMAL
        }

        let conf = confidence.unwrap_or(0.5);

        // ── FAST Decision (System 1) ──
        if entropy <= self.fast_threshold {
            if confidence.is_some() && conf < self.conf_threshold {
                return 1; // Low entropy but low confidence → don't trust
            }
            self.record_decision(0);
            return 0; // FAST
        }

        // ── DEEP Decision (System 2): Bonferroni 4-way test ──
        let (emean, estd, _, _, _) = self.entropy_win.full_stats();
        let z = (entropy - emean) / estd.max(Z_SCORE_STD_FLOOR);

        let mut criteria: u8 = 0;
        if z > BONFERRONI_Z {
            criteria += 1; // A. Statistical outlier
        }
        if self.consec_high >= self.consec_threshold {
            criteria += 1; // B. Consecutive high
        }
        if self.o1_posterior > self.posterior_threshold {
            criteria += 1; // C. Bayesian posterior
        }
        if self.change_detected {
            criteria += 1; // D. CUSUM alarm
        }

        if criteria >= 2 {
            self.record_decision(2);
            return 2; // DEEP
        }

        self.record_decision(1);
        1 // NORMAL
    }

    /// Dynamic z thresholds via Cornish-Fisher. Returns (z_uncertain, z_unknown).
    fn get_dynamic_z_thresholds(&self) -> (f64, f64) {
        let (_, _, skew, kurt, n) = self.entropy_win.full_stats();
        if n < 30 {
            return (1.0, 2.0);
        }
        let sk = skew.max(-2.0).min(2.0);
        let ku = kurt.max(-2.0).min(2.0);
        let z_unc = cornish_fisher(1.0, sk, ku).max(0.5);
        let z_unk = cornish_fisher(2.0, sk, ku).max(1.5);
        if z_unk <= z_unc {
            (z_unc, z_unc + 1.0)
        } else {
            (z_unc, z_unk)
        }
    }

    /// Get EMA-based z-score (consistent with CUSUM internal z-score).
    fn get_z_score(&self, entropy: f64) -> f64 {
        if self.step_count > 0 {
            return self.last_z;
        }
        if entropy < SAFE_ENTROPY_THRESHOLD {
            return 0.0;
        }
        let std = self.entropy_emv.sqrt().max(Z_SCORE_STD_FLOOR);
        (entropy - self.entropy_ema) / std
    }

    /// Return (entropy_gradient, entropy_momentum).
    fn get_predictive_signals(&self) -> (f64, f64) {
        (self.entropy_gradient, self.entropy_momentum)
    }

    /// Reset session state (preserve learned thresholds).
    fn reset_session(&mut self) {
        self.consec_high = 0;
        self.cusum_pos = 0.0;
        self.cusum_neg = 0.0;
        self.change_detected = false;
        self.circuit_breaker = false;
        self.dec_count = 0;
        self.dec_head = 0;
    }

    /// Export complete statistics as a dict.
    #[getter]
    fn stats(&self) -> HashMap<String, f64> {
        let (emean, estd, eskew, ekurt, _) = self.entropy_win.full_stats();
        let mut m = HashMap::new();
        m.insert("entropy_mean".into(), emean);
        m.insert("entropy_std".into(), estd);
        m.insert("entropy_skew".into(), eskew);
        m.insert("entropy_kurt".into(), ekurt);
        m.insert("fast_threshold".into(), self.fast_threshold);
        m.insert("deep_threshold".into(), self.deep_threshold);
        m.insert("lambda_aff".into(), self.current_lambda);
        m.insert("o1_posterior".into(), self.o1_posterior);
        m.insert("cusum_pos".into(), self.cusum_pos);
        m.insert("cusum_neg".into(), self.cusum_neg);
        m.insert("cusum_h".into(), self.cusum_h);
        m.insert(
            "change_detected".into(),
            if self.change_detected { 1.0 } else { 0.0 },
        );
        m.insert(
            "is_calibrated".into(),
            if self.is_calibrated { 1.0 } else { 0.0 },
        );
        m.insert("step_count".into(), self.step_count as f64);
        m
    }
}

// ─────────────────────────────────────────────────────────
// Internal methods (not exposed to Python)
// ─────────────────────────────────────────────────────────

impl AdaptiveControllerNative {
    /// Cornish-Fisher threshold recalibration.
    fn recalculate_thresholds(&mut self) {
        let (mean, std, skew, kurt, _) = self.entropy_win.full_stats();
        let sk = skew.max(-2.0).min(2.0);
        let ku = kurt.max(-2.0).min(2.0);

        // FAST: z = -0.5 (~31st percentile)
        let z_adj_fast = cornish_fisher(-0.5, sk, ku);
        let fast = mean + z_adj_fast * std;

        // DEEP: z = 1.5 (~93rd percentile)
        let z_adj_deep = cornish_fisher(1.5, sk, ku);
        let deep = mean + z_adj_deep * std;

        // Safety clamps
        self.fast_threshold = fast.max(0.0).min(mean);
        self.deep_threshold = deep.max(mean + 0.5 * std);

        // Confidence threshold
        let (cmean, cstd, _, _, cn) = self.confidence_win.full_stats();
        if cn > 0 && cstd > 0.0 {
            self.conf_threshold = cmean - 0.5 * cstd;
        }
    }

    /// Learn consecutive threshold from run history (MAD-based).
    fn learn_consecutive(&mut self) {
        if self.high_runs.len() < 5 {
            return;
        }
        let mut sorted = self.high_runs.clone();
        sorted.sort();
        let median = sorted[sorted.len() / 2];
        let mut abs_devs: Vec<usize> = sorted
            .iter()
            .map(|&r| if r > median { r - median } else { median - r })
            .collect();
        abs_devs.sort();
        let mad = abs_devs[abs_devs.len() / 2];
        self.consec_threshold = (median + mad).max(2);
    }

    /// Record decision in circular buffer.
    fn record_decision(&mut self, decision: u8) {
        let idx = self.dec_head % DECISION_BUF_LEN;
        self.dec_buf[idx] = decision;
        self.dec_head += 1;
        if self.dec_count < DECISION_BUF_LEN {
            self.dec_count += 1;
        }
    }

    /// Update circuit breaker: detect pathological decision patterns.
    fn update_circuit_breaker(&mut self) {
        if self.dec_count < CB_HISTORY_LEN {
            self.circuit_breaker = false;
            return;
        }
        let start = self.dec_head.saturating_sub(self.dec_count);
        let mut fast_c = 0usize;
        let mut deep_c = 0usize;
        for i in start..self.dec_head {
            match self.dec_buf[i % DECISION_BUF_LEN] {
                0 => fast_c += 1,
                2 => deep_c += 1,
                _ => {}
            }
        }
        let total = self.dec_count as f64;
        self.circuit_breaker =
            (fast_c as f64 / total > CB_FAST_RATIO) || (deep_c as f64 / total > CB_DEEP_RATIO);
    }
}
