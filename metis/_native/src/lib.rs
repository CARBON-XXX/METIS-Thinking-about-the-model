//! METIS Native Accelerators (Phase 21)
//!
//! Rust implementations of CPU-bound hot paths in the METIS cognitive pipeline.
//! These replace pure-Python equivalents for ~6-75x speedup on tight loops.
//!
//! Exposed modules:
//!   - detect_repetition_hybrid(tokens, max_window) -> (len, score)
//!   - SlidingWindowStats: Hybrid Welford O(1) mean/variance with periodic
//!     O(N) recalibration of skewness/kurtosis every 100 steps.
//!   - AdaptiveControllerNative: Rust port of AdaptiveController with
//!     Siegmund-corrected CUSUM, Cornish-Fisher quantiles, Empirical Bayes.
//!   - CotCusumNative: CUSUM-based CoT trigger detection.
//!   - CognitiveRewardNative: Reward computation hot path.
//!   - BoundaryGuardNative: Epistemic boundary classification.

use pyo3::prelude::*;
use std::collections::HashSet;

mod boundary;
mod controller;
mod cot;
mod rewards;

// ─────────────────────────────────────────────────────────
// 1. Repetition Detection (Jaccard + Positional Fuzzy)
// ─────────────────────────────────────────────────────────

/// Hybrid repetition detection matching Python's _detect_repetition_hybrid.
///
/// - Long windows (>=32): Jaccard set similarity >= 0.7
/// - Short windows (4..=15): Positional fuzzy match >= 0.9
///
/// Returns (repetition_length, score). (0, 0.0) if no repetition found.
#[pyfunction]
fn detect_repetition_hybrid(tokens: Vec<u32>, max_window: usize) -> (usize, f64) {
    let n = tokens.len();

    // 1. Long Semantic Loops — Jaccard similarity
    //    Step size 4, window range [32, max_window]
    let mut w = 32usize;
    while w <= max_window {
        if n >= 2 * w {
            let start_a = n - 2 * w;
            let end_a = n - w;
            let start_b = n - w;

            let set_a: HashSet<u32> = tokens[start_a..end_a].iter().copied().collect();
            let set_b: HashSet<u32> = tokens[start_b..n].iter().copied().collect();

            if !set_a.is_empty() && !set_b.is_empty() {
                let intersection = set_a.intersection(&set_b).count();
                let union = set_a.union(&set_b).count();
                let jaccard = intersection as f64 / union as f64;

                if jaccard >= 0.7 {
                    return (w, jaccard);
                }
            }
        }
        w += 4;
    }

    // 2. Short Exact/Near-Exact Loops — Positional match
    //    Dense scan downwards from min(max_window, 15) to 4
    let short_max = if max_window < 15 { max_window } else { 15 };
    let mut w = short_max;
    while w > 3 {
        if n >= 2 * w {
            let mut matches = 0u32;
            for i in 0..w {
                if tokens[n - 2 * w + i] == tokens[n - w + i] {
                    matches += 1;
                }
            }
            let score = matches as f64 / w as f64;
            if score >= 0.9 {
                return (w, score);
            }
        }
        w -= 1;
    }

    (0, 0.0)
}

// ─────────────────────────────────────────────────────────
// 2. Sliding Window Statistics — Hybrid O(1) / O(N)
// ─────────────────────────────────────────────────────────
//
// Phase 18 Hybrid Architecture:
//   Mean & Variance: strict O(1) incremental (Welford-style running sum).
//   Skew & Kurtosis: periodic O(N) recalibration every K steps.
//
// Rationale: Welford is numerically rock-solid for 1st/2nd moments.
// But sliding-window subtraction of 3rd/4th moment accumulators
// suffers catastrophic cancellation over time. Periodic O(N) over
// a few hundred f64 values costs nanoseconds (fits in L1 cache,
// auto-vectorized by LLVM), indistinguishable from O(1) in the
// macro tensor-computation context.

const CALIBRATION_INTERVAL: usize = 100;

/// High-performance sliding window statistics.
///
/// O(1) mean/std via running accumulators.
/// Periodic O(N) skew/kurt recalibration every 100 steps.
#[pyclass]
struct SlidingWindowStats {
    buffer: Vec<f64>,
    head: usize,
    count: usize,
    capacity: usize,
    // ── O(1) incremental accumulators ──
    running_sum: f64,
    running_sum_sq: f64, // Σ(x_i - mean)^2 maintained via Welford update
    // ── Periodic O(N) cache for higher moments ──
    cached_skew: f64,
    cached_kurt: f64,
    steps_since_cal: usize,
}

#[pymethods]
impl SlidingWindowStats {
    #[new]
    #[pyo3(signature = (window_size = 500))]
    fn new(window_size: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(window_size),
            head: 0,
            count: 0,
            capacity: window_size,
            running_sum: 0.0,
            running_sum_sq: 0.0,
            cached_skew: 0.0,
            cached_kurt: 0.0,
            steps_since_cal: 0,
        }
    }

    /// Number of elements currently in the window.
    #[getter]
    fn n(&self) -> usize {
        self.count
    }

    /// Push a new value into the sliding window.
    /// O(1) amortized: updates running_sum and running_sum_sq incrementally.
    fn update(&mut self, x: f64) {
        if self.count < self.capacity {
            // ── Growing phase: buffer not yet full ──
            let old_mean = if self.count > 0 {
                self.running_sum / self.count as f64
            } else {
                0.0
            };
            self.buffer.push(x);
            self.count += 1;
            self.running_sum += x;
            let new_mean = self.running_sum / self.count as f64;
            // Welford: sum_sq += (x - old_mean) * (x - new_mean)
            self.running_sum_sq += (x - old_mean) * (x - new_mean);
            self.head = self.count % self.capacity;
        } else {
            // ── Sliding phase: evict oldest, insert new ──
            let old_val = self.buffer[self.head];
            let old_mean = self.running_sum / self.count as f64;

            // Remove old_val contribution
            self.running_sum -= old_val;
            let interim_mean = self.running_sum / self.count as f64;
            // Reverse Welford for removal: subtract (old_val - old_mean)*(old_val - interim_mean)
            self.running_sum_sq -= (old_val - old_mean) * (old_val - interim_mean);

            // Add new value
            self.running_sum += x;
            let new_mean = self.running_sum / self.count as f64;
            // Forward Welford for insertion: add (x - interim_mean)*(x - new_mean)
            self.running_sum_sq += (x - interim_mean) * (x - new_mean);

            // Numerical safety: variance accumulator must never go negative
            if self.running_sum_sq < 0.0 {
                self.running_sum_sq = 0.0;
            }

            self.buffer[self.head] = x;
            self.head = (self.head + 1) % self.capacity;
        }

        // ── Periodic recalibration of higher moments ──
        self.steps_since_cal += 1;
        if self.steps_since_cal >= CALIBRATION_INTERVAL {
            self.recalibrate();
        }
    }

    /// Current mean — O(1).
    #[getter]
    fn mean(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.running_sum / self.count as f64
    }

    /// Current std with Bessel correction — O(1).
    #[getter]
    fn std(&self) -> f64 {
        if self.count < 2 {
            return 0.1;
        }
        let var = self.running_sum_sq / (self.count as f64 - 1.0);
        if var > 1e-10 { var.sqrt() } else { 0.01 }
    }

    /// Compute all statistics: mean, std (O(1)), skew, kurt (cached).
    ///
    /// Returns a dict matching Python SlidingWindowStats.get_stats().
    fn get_stats(&self) -> PyResult<std::collections::HashMap<String, f64>> {
        let n = self.count;
        let mut result = std::collections::HashMap::new();

        if n < 2 {
            result.insert("mean".into(), 0.0);
            result.insert("std".into(), 0.1);
            result.insert("skew".into(), 0.0);
            result.insert("kurt".into(), 0.0);
            result.insert("n".into(), n as f64);
            return Ok(result);
        }

        let mean_val = self.running_sum / n as f64;
        let var = self.running_sum_sq / (n as f64 - 1.0);
        let std_val = if var > 1e-10 { var.sqrt() } else { 0.01 };

        result.insert("mean".into(), mean_val);
        result.insert("std".into(), std_val);
        result.insert("skew".into(), self.cached_skew);
        result.insert("kurt".into(), self.cached_kurt);
        result.insert("n".into(), n as f64);
        Ok(result)
    }

    /// Reset all state.
    fn reset(&mut self) {
        self.buffer.clear();
        self.head = 0;
        self.count = 0;
        self.running_sum = 0.0;
        self.running_sum_sq = 0.0;
        self.cached_skew = 0.0;
        self.cached_kurt = 0.0;
        self.steps_since_cal = 0;
    }
}

impl SlidingWindowStats {
    /// O(N) recalibration of skewness and kurtosis from buffer.
    /// Called every CALIBRATION_INTERVAL steps. Also corrects any
    /// floating-point drift in running_sum and running_sum_sq.
    fn recalibrate(&mut self) {
        self.steps_since_cal = 0;
        let n = self.count;
        if n < 3 {
            self.cached_skew = 0.0;
            self.cached_kurt = 0.0;
            return;
        }

        let nf = n as f64;
        let data = &self.buffer[..n];

        // Authoritative mean from buffer (corrects drift)
        let mean: f64 = data.iter().sum::<f64>() / nf;
        self.running_sum = mean * nf;

        let (mut m2, mut m3, mut m4) = (0.0f64, 0.0f64, 0.0f64);
        for &x in data {
            let d = x - mean;
            let d2 = d * d;
            m2 += d2;
            m3 += d2 * d;
            m4 += d2 * d2;
        }

        // Correct variance accumulator drift
        self.running_sum_sq = m2;

        // Population std for moment standardization
        let std_pop = {
            let v = m2 / nf;
            if v > 0.0 { v.sqrt() } else { 0.01 }
        };

        // Skewness (Fisher-Pearson, bias-corrected)
        self.cached_skew = if n >= 3 && std_pop > 1e-6 {
            let g1 = (m3 / nf) / std_pop.powi(3);
            g1 * (nf * (nf - 1.0)).sqrt() / (nf - 2.0)
        } else {
            0.0
        };

        // Kurtosis (Fisher excess, bias-corrected)
        self.cached_kurt = if n >= 4 && std_pop > 1e-6 {
            let g2 = (m4 / nf) / std_pop.powi(4) - 3.0;
            ((nf + 1.0) * g2 + 6.0) * (nf - 1.0) / ((nf - 2.0) * (nf - 3.0))
        } else {
            0.0
        };
    }
}

// ─────────────────────────────────────────────────────────
// Standalone functions
// ─────────────────────────────────────────────────────────

/// Cornish-Fisher expansion: non-Gaussian quantile adjustment.
///
/// z_adj = z + (z²-1)S/6 + (z³-3z)K/24
#[pyfunction]
fn cornish_fisher_quantile(z: f64, skew: f64, kurt: f64) -> f64 {
    controller::cornish_fisher(z, skew, kurt)
}

// ─────────────────────────────────────────────────────────
// Module registration
// ─────────────────────────────────────────────────────────

#[pymodule]
fn metis_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Existing
    m.add_function(wrap_pyfunction!(detect_repetition_hybrid, m)?)?;
    m.add_class::<SlidingWindowStats>()?;
    // Phase 1: new accelerators
    m.add_function(wrap_pyfunction!(cornish_fisher_quantile, m)?)?;
    m.add_class::<boundary::BoundaryGuardNative>()?;
    m.add_class::<cot::CotCusumNative>()?;
    m.add_class::<controller::AdaptiveControllerNative>()?;
    m.add_class::<rewards::RewardComputerNative>()?;
    Ok(())
}

// ─────────────────────────────────────────────────────────
// Unit Tests — cargo test
// ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Welford O(1) Statistics ──

    #[test]
    fn test_welford_mean_basic() {
        let mut w = SlidingWindowStats::new(100);
        for i in 1..=10 {
            w.update(i as f64);
        }
        let mean = w.running_sum / w.count as f64;
        assert!((mean - 5.5).abs() < 1e-10, "mean should be 5.5, got {mean}");
    }

    #[test]
    fn test_welford_variance_known() {
        // Variance of [1,2,3,4,5] with Bessel: 2.5
        let mut w = SlidingWindowStats::new(100);
        for i in 1..=5 {
            w.update(i as f64);
        }
        let var = w.running_sum_sq / (w.count as f64 - 1.0);
        assert!((var - 2.5).abs() < 1e-10, "var should be 2.5, got {var}");
    }

    #[test]
    fn test_welford_sliding_eviction() {
        // Window size 5, push 10 values → only last 5 remain
        let mut w = SlidingWindowStats::new(5);
        for i in 1..=10 {
            w.update(i as f64);
        }
        assert_eq!(w.count, 5);
        let mean = w.running_sum / w.count as f64;
        // Last 5: [6,7,8,9,10] → mean = 8.0
        assert!((mean - 8.0).abs() < 1e-6, "sliding mean should be 8.0, got {mean}");
    }

    #[test]
    fn test_welford_variance_nonnegative() {
        // Adversarial: constant input should never produce negative variance
        let mut w = SlidingWindowStats::new(10);
        for _ in 0..100 {
            w.update(42.0);
        }
        assert!(w.running_sum_sq >= 0.0, "variance accumulator must be >= 0");
    }

    #[test]
    fn test_welford_recalibration_drift_correction() {
        // After many updates, recalibration should correct drift
        let mut w = SlidingWindowStats::new(50);
        for i in 0..500 {
            w.update((i as f64 * 0.1).sin());
        }
        // After recalibration, mean should match buffer truth
        let n = w.count as f64;
        let buf_mean: f64 = w.buffer.iter().sum::<f64>() / n;
        let welford_mean = w.running_sum / n;
        assert!(
            (welford_mean - buf_mean).abs() < 1e-10,
            "post-recal drift: welford={welford_mean}, truth={buf_mean}"
        );
    }

    #[test]
    fn test_welford_empty_and_single() {
        let w = SlidingWindowStats::new(10);
        assert_eq!(w.count, 0);
        // mean of empty = 0
        assert_eq!(w.running_sum, 0.0);

        let mut w2 = SlidingWindowStats::new(10);
        w2.update(7.0);
        assert_eq!(w2.count, 1);
        let mean = w2.running_sum / w2.count as f64;
        assert!((mean - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_welford_reset() {
        let mut w = SlidingWindowStats::new(10);
        for i in 0..20 {
            w.update(i as f64);
        }
        w.reset();
        assert_eq!(w.count, 0);
        assert_eq!(w.running_sum, 0.0);
        assert_eq!(w.running_sum_sq, 0.0);
        assert_eq!(w.cached_skew, 0.0);
        assert_eq!(w.cached_kurt, 0.0);
    }

    // ── Repetition Detection ──

    #[test]
    fn test_repetition_exact_loop() {
        // Min short window = 4, so use [1,2,3,4,1,2,3,4]
        let tokens = vec![1, 2, 3, 4, 1, 2, 3, 4];
        let (len, score) = detect_repetition_hybrid(tokens, 64);
        assert!(len >= 4, "should detect repetition, got len={len}");
        assert!((score - 1.0).abs() < 1e-10, "exact loop score should be 1.0, got {score}");
    }

    #[test]
    fn test_repetition_no_loop() {
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let (len, _score) = detect_repetition_hybrid(tokens, 64);
        assert_eq!(len, 0, "no repetition in unique sequence");
    }

    #[test]
    fn test_repetition_jaccard_long() {
        // Build a long repeated block ≥32 tokens
        let block: Vec<u32> = (0..40).collect();
        let mut tokens = block.clone();
        tokens.extend_from_slice(&block);
        let (len, score) = detect_repetition_hybrid(tokens, 64);
        assert!(len >= 32, "should detect long Jaccard repetition, got len={len}");
        assert!(score >= 0.7, "Jaccard score should be >= 0.7, got {score}");
    }

    #[test]
    fn test_repetition_too_short() {
        // Only 4 tokens, need 2*w >= 8 for w=4
        let tokens = vec![1, 2, 3, 4];
        let (len, _) = detect_repetition_hybrid(tokens, 64);
        assert_eq!(len, 0, "too short for any detection");
    }

    // ── Cornish-Fisher Quantile ──

    #[test]
    fn test_cornish_fisher_gaussian() {
        // For skew=0, kurt=0 (Gaussian): z_adj = z
        let z = 1.96;
        let adj = controller::cornish_fisher(z, 0.0, 0.0);
        assert!((adj - z).abs() < 1e-10, "Gaussian case: adj should equal z");
    }

    #[test]
    fn test_cornish_fisher_skewed() {
        // With positive skew, z_adj > z for z > 1
        let z = 1.96;
        let adj = controller::cornish_fisher(z, 0.5, 0.0);
        assert!(adj > z, "positive skew should increase the quantile");
    }

    #[test]
    fn test_cornish_fisher_formula() {
        // z_adj = z + (z²-1)*S/6 + (z³-3z)*K/24
        let z = 2.0;
        let s = 0.3;
        let k = 0.1;
        let expected = z + (z * z - 1.0) * s / 6.0 + (z * z * z - 3.0 * z) * k / 24.0;
        let result = controller::cornish_fisher(z, s, k);
        assert!(
            (result - expected).abs() < 1e-10,
            "CF formula mismatch: expected={expected}, got={result}"
        );
    }
}
