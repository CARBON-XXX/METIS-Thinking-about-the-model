//! METIS Native Accelerators
//!
//! Rust implementations of CPU-bound hot paths in the METIS cognitive pipeline.
//! These replace pure-Python equivalents for ~10-50x speedup on tight loops.
//!
//! Exposed functions:
//!   - detect_repetition_hybrid(tokens, max_window) -> (len, score)
//!   - SlidingWindowStats: single-pass online moments (mean, std, skew, kurt)

use pyo3::prelude::*;
use std::collections::HashSet;

mod boundary;
mod controller;
mod cot;

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
// 2. Sliding Window Statistics (single-pass moments)
// ─────────────────────────────────────────────────────────

/// High-performance sliding window statistics.
///
/// Maintains a circular buffer and computes mean, std (Bessel),
/// skewness (Fisher-Pearson unbiased), kurtosis (Fisher excess unbiased)
/// in a single pass over the buffer.
#[pyclass]
struct SlidingWindowStats {
    buffer: Vec<f64>,
    head: usize,      // Next write position
    count: usize,     // Current number of elements (<= capacity)
    capacity: usize,
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
        }
    }

    /// Number of elements currently in the window.
    #[getter]
    fn n(&self) -> usize {
        self.count
    }

    /// Push a new value into the sliding window.
    fn update(&mut self, x: f64) {
        if self.count < self.capacity {
            self.buffer.push(x);
            self.count += 1;
            self.head = self.count % self.capacity;
        } else {
            self.buffer[self.head] = x;
            self.head = (self.head + 1) % self.capacity;
        }
    }

    /// Current mean (O(N) but fast in Rust).
    #[getter]
    fn mean(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let sum: f64 = self.buffer[..self.count].iter().sum();
        sum / self.count as f64
    }

    /// Current std with Bessel correction.
    #[getter]
    fn std(&self) -> f64 {
        if self.count < 2 {
            return 0.1;
        }
        let n = self.count as f64;
        let mean = self.mean();
        let m2: f64 = self.buffer[..self.count]
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum();
        let var = m2 / (n - 1.0);
        if var > 1e-10 { var.sqrt() } else { 0.01 }
    }

    /// Compute all statistics in a single pass: mean, std, skew, kurt, n.
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

        let nf = n as f64;
        let data = &self.buffer[..n];

        // Single pass: accumulate sum, sum of squared deviations,
        // sum of cubed deviations, sum of quartic deviations.
        // Two-pass for numerical stability (first pass = mean).
        let mean: f64 = data.iter().sum::<f64>() / nf;

        let mut m2: f64 = 0.0;
        let mut m3: f64 = 0.0;
        let mut m4: f64 = 0.0;
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
            if v > 0.0 { v.sqrt() } else { 0.01 }
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

        result.insert("mean".into(), mean);
        result.insert("std".into(), std_val);
        result.insert("skew".into(), skew);
        result.insert("kurt".into(), kurt);
        result.insert("n".into(), nf);
        Ok(result)
    }

    /// Reset all state.
    fn reset(&mut self) {
        self.buffer.clear();
        self.head = 0;
        self.count = 0;
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
    Ok(())
}
