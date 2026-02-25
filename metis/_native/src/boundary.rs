//! METIS Epistemic Boundary Guard — Rust accelerated CUSUM.
//!
//! Matches Python `EpistemicBoundaryGuard.evaluate()` logic exactly.
//! Returns integer codes for state/action to avoid crossing complex Python enums.
//!
//!   state: 0=KNOWN, 1=LIKELY, 2=UNCERTAIN, 3=UNKNOWN
//!   action: 0=GENERATE, 1=HEDGE, 2=SEEK, 3=REFUSE

use pyo3::prelude::*;

#[pyclass(module = "metis_native")]
pub struct BoundaryGuardNative {
    uncertain_z: f64,
    unknown_z: f64,
    known_z: f64,
    min_warmup: usize,
    tok_count: usize,
    cusum: f64,
    last_surprise: f64,
    cusum_k: f64,
    hedge_h: f64,
    refuse_h: f64,
    decay: f64,
    surprise_base: f64,
    surprise_w: f64,
    conf_refuse: f64,
    conf_seek: f64,
    conf_known: f64,
    action_counts: [u64; 4],
}

#[pymethods]
impl BoundaryGuardNative {
    #[new]
    #[pyo3(signature = (
        uncertain_z = 1.0,
        unknown_z = 1.2,
        known_z = -0.5,
        min_warmup = 4,
        cusum_k = 0.5,
        hedge_h = 4.0,
        refuse_h = 8.0,
        decay = 0.85,
        surprise_base = 2.5,
        surprise_w = 0.25,
        conf_refuse = 0.3,
        conf_seek = 0.7,
        conf_known = 0.7,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        uncertain_z: f64,
        unknown_z: f64,
        known_z: f64,
        min_warmup: usize,
        cusum_k: f64,
        hedge_h: f64,
        refuse_h: f64,
        decay: f64,
        surprise_base: f64,
        surprise_w: f64,
        conf_refuse: f64,
        conf_seek: f64,
        conf_known: f64,
    ) -> Self {
        Self {
            uncertain_z,
            unknown_z,
            known_z,
            min_warmup,
            tok_count: 0,
            cusum: 0.0,
            last_surprise: 0.0,
            cusum_k,
            hedge_h,
            refuse_h,
            decay,
            surprise_base,
            surprise_w,
            conf_refuse,
            conf_seek,
            conf_known,
            action_counts: [0; 4],
        }
    }

    /// Evaluate epistemic boundary.
    ///
    /// Returns `(state_u8, action_u8, explanation)`.
    #[pyo3(signature = (z, confidence, sd, z_unc = None, z_unk = None))]
    fn evaluate(
        &mut self,
        z: f64,
        confidence: f64,
        sd: f64,
        z_unc: Option<f64>,
        z_unk: Option<f64>,
    ) -> (u8, u8, String) {
        self.tok_count += 1;
        let z_unc = z_unc.unwrap_or(self.uncertain_z);
        let z_unk = z_unk.unwrap_or(self.unknown_z);

        // Cold-start: not enough tokens yet
        if self.tok_count <= self.min_warmup {
            self.action_counts[0] += 1;
            return (1, 0, String::new()); // LIKELY, GENERATE
        }

        // ── Dynamic Allowance (K) ──
        let mut current_k = self.cusum_k;
        if self.tok_count <= 20 {
            current_k = self.cusum_k + 0.5;
        }

        // ── CUSUM update ──
        // Positive z above allowance: accumulate weighted by semantic diversity
        if z > current_k {
            self.cusum += (z - current_k) * sd;
        } else if z < 0.0 {
            // Confident token: geometric decay
            self.cusum *= self.decay;
        }

        // Surprise feedback: external surprise signal boosts CUSUM
        if self.last_surprise > self.surprise_base {
            self.cusum += (self.last_surprise - self.surprise_base) * self.surprise_w;
        }

        // ── Epistemic state classification ──
        let state: u8 = if z > z_unk {
            3 // UNKNOWN
        } else if z > z_unc {
            2 // UNCERTAIN
        } else if z < self.known_z && confidence > self.conf_known {
            0 // KNOWN
        } else {
            1 // LIKELY
        };

        // ── Boundary actions (priority: REFUSE > SEEK > HEDGE > GENERATE) ──
        
        // Broader window + relaxed diversity to catch multi-meaning exploration
        let is_intent_exploration = self.tok_count <= 80 && sd >= 0.7;

        // REFUSE: extreme sustained uncertainty + very low confidence
        if self.cusum >= self.refuse_h && confidence < self.conf_refuse {
            let v = self.cusum;
            self.cusum = 0.0;
            self.action_counts[3] += 1;
            return (
                3,
                3,
                format!("Sustained extreme uncertainty (cusum={v:.1})"),
            );
        }

        // SEEK: extreme sustained uncertainty + moderate confidence
        if self.cusum >= self.refuse_h && confidence < self.conf_seek {
            let v = self.cusum;
            self.cusum = 0.0;
            self.action_counts[2] += 1;
            return (
                3,
                2,
                format!("External verification needed (cusum={v:.1})"),
            );
        }

        // HEDGE: moderate accumulated uncertainty
        if self.cusum >= self.hedge_h {
            let v = self.cusum;
            self.cusum = 0.0;
            
            if is_intent_exploration {
                self.action_counts[2] += 1;
                return (
                    2,
                    2,
                    format!("Intent clarification needed (cusum={v:.1})"),
                );
            }
            
            self.action_counts[1] += 1;
            return (
                2,
                1,
                format!("Accumulated uncertainty (cusum={v:.1})"),
            );
        }

        // GENERATE: no action needed
        self.action_counts[0] += 1;
        (state, 0, String::new())
    }

    /// Feed external surprise signal for next evaluation.
    fn feed_surprise(&mut self, surprise: f64) {
        self.last_surprise = surprise;
    }

    /// Current CUSUM accumulator value.
    #[getter]
    fn uncertainty_score(&self) -> f64 {
        self.cusum
    }

    /// Action counts: [GENERATE, HEDGE, SEEK, REFUSE].
    fn get_action_counts(&self) -> [u64; 4] {
        self.action_counts
    }

    fn reset(&mut self) {
        self.cusum = 0.0;
        self.last_surprise = 0.0;
        self.tok_count = 0;
        self.action_counts = [0; 4];
    }
}
