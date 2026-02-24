//! METIS CoT Difficulty CUSUM + Momentum trigger — Rust accelerated.
//!
//! Matches Python `CoTManager.observe()` + `should_inject()` logic exactly.
//! Decision codes: 0=FAST, 1=NORMAL, 2=DEEP
//! Strategy codes: 0=STANDARD, 1=REFLECTION, 2=DECOMPOSITION, 3=CLARIFICATION

use pyo3::prelude::*;

#[pyclass(module = "metis_native")]
pub struct CotCusumNative {
    cusum_k: f64,
    cusum_h: f64,
    cusum_decay: f64,
    deep_bonus: f64,
    momentum_h: f64,
    cusum_early: f64,
    cooldown: usize,
    max_inj: usize,
    // State
    diff_cusum: f64,
    mom_acc: f64,
    mom_steps: usize,
    since_last: usize,
    total_inj: usize,
    consec_deep: usize,
    // Oscillation circular buffer
    dec_buf: Vec<u8>,
    dec_head: usize,
    dec_count: usize,
    osc_window: usize,
    osc_threshold: usize,
}

#[pymethods]
impl CotCusumNative {
    #[new]
    #[pyo3(signature = (
        cooldown = 40,
        max_inj = 3,
        cusum_k = 0.3,
        cusum_h = 4.0,
        cusum_decay = 0.9,
        deep_bonus = 0.3,
        momentum_h = 2.0,
        osc_window = 8,
        osc_threshold = 6,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        cooldown: usize,
        max_inj: usize,
        cusum_k: f64,
        cusum_h: f64,
        cusum_decay: f64,
        deep_bonus: f64,
        momentum_h: f64,
        osc_window: usize,
        osc_threshold: usize,
    ) -> Self {
        Self {
            cusum_k,
            cusum_h,
            cusum_decay,
            deep_bonus,
            momentum_h,
            cusum_early: cusum_h * 0.5,
            cooldown,
            max_inj,
            diff_cusum: 0.0,
            mom_acc: 0.0,
            mom_steps: 0,
            since_last: cooldown, // Allow first injection immediately
            total_inj: 0,
            consec_deep: 0,
            dec_buf: vec![0u8; osc_window],
            dec_head: 0,
            dec_count: 0,
            osc_window,
            osc_threshold,
        }
    }

    /// Observe a new cognitive signal.
    ///
    /// `decision`: 0=FAST, 1=NORMAL, 2=DEEP
    fn observe(&mut self, z: f64, sd: f64, decision: u8, entropy_momentum: f64) {
        // Decision history (circular buffer)
        let idx = self.dec_head % self.osc_window;
        self.dec_buf[idx] = decision;
        self.dec_head += 1;
        if self.dec_count < self.osc_window {
            self.dec_count += 1;
        }

        if decision == 2 {
            self.consec_deep += 1;
        } else {
            self.consec_deep = 0;
        }
        self.since_last += 1;

        // ── Difficulty CUSUM ──
        // difficulty(t) = (max(0, z) + deep_bonus) * sd - k
        let db = if decision == 2 {
            self.deep_bonus
        } else {
            0.0
        };
        if z > 0.0 || db > 0.0 {
            let inc = (z.max(0.0) + db) * sd - self.cusum_k;
            self.diff_cusum = (self.diff_cusum + inc).max(0.0);
        } else if z < 0.0 {
            // Confident token: geometric decay
            self.diff_cusum *= self.cusum_decay;
        }

        // ── Momentum accumulator (predictive early-warning) ──
        if entropy_momentum > 0.0 {
            self.mom_acc += entropy_momentum;
            self.mom_steps += 1;
        } else {
            self.mom_acc *= 0.8;
            self.mom_steps = 0;
        }
    }

    /// Whether to trigger a `<thinking>` block.
    fn should_inject(&self) -> bool {
        if self.total_inj >= self.max_inj || self.since_last < self.cooldown {
            return false;
        }
        // Path 1: Classic CUSUM trigger
        if self.diff_cusum >= self.cusum_h {
            return true;
        }
        // Path 2: Predictive momentum trigger (entropy accelerating + CUSUM >= 50%)
        self.diff_cusum >= self.cusum_early
            && self.mom_acc >= self.momentum_h
            && self.mom_steps >= 3
    }

    /// Select CoT strategy.
    ///
    /// Returns: 0=STANDARD, 1=REFLECTION, 2=DECOMPOSITION, 3=CLARIFICATION
    fn select_strategy(&self, sd: f64, confidence: f64, phase: &str) -> u8 {
        if self.detect_oscillation() || phase == "confusion" {
            return 1; // REFLECTION
        }
        if self.consec_deep >= 5 || phase == "exploration" {
            return 2; // DECOMPOSITION
        }
        if sd > 0.6 && confidence < 0.3 {
            return 3; // CLARIFICATION
        }
        0 // STANDARD
    }

    /// Record a CoT injection and reset accumulators.
    fn record_injection(&mut self) {
        self.total_inj += 1;
        self.since_last = 0;
        self.diff_cusum = 0.0;
        self.mom_acc = 0.0;
        self.mom_steps = 0;
    }

    fn reset(&mut self) {
        self.diff_cusum = 0.0;
        self.mom_acc = 0.0;
        self.mom_steps = 0;
        self.since_last = self.cooldown;
        self.total_inj = 0;
        self.consec_deep = 0;
        self.dec_count = 0;
        self.dec_head = 0;
    }

    #[getter]
    fn difficulty_cusum_val(&self) -> f64 {
        self.diff_cusum
    }

    #[getter]
    fn momentum_acc_val(&self) -> f64 {
        self.mom_acc
    }

    #[getter]
    fn remaining_budget(&self) -> usize {
        self.max_inj.saturating_sub(self.total_inj)
    }

    #[getter]
    fn total_injections(&self) -> usize {
        self.total_inj
    }

    #[getter]
    fn consecutive_deep(&self) -> usize {
        self.consec_deep
    }

    #[getter]
    fn steps_since_last_cot(&self) -> usize {
        self.since_last
    }
}

// Internal methods (not exposed to Python)
impl CotCusumNative {
    fn detect_oscillation(&self) -> bool {
        if self.dec_count < self.osc_window {
            return false;
        }
        let w = self.osc_window;
        let start = self.dec_head.saturating_sub(w);
        let mut switches = 0usize;
        let mut prev: Option<u8> = None;
        for i in start..self.dec_head {
            let d = self.dec_buf[i % w];
            if let Some(p) = prev {
                if d != p {
                    switches += 1;
                }
            }
            prev = Some(d);
        }
        switches >= self.osc_threshold
    }
}
