//! METIS Cognitive Reward Accelerator
//!
//! Rust implementation of the 5 reward components from rewards.py.
//! Takes pre-extracted arrays from CognitiveTrace events and returns
//! individual reward scores + total.
//!
//! Each function mirrors the Python implementation exactly, but runs
//! ~10-50x faster due to tight loops over contiguous f64 arrays.

use pyo3::prelude::*;
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════
// Configuration (mirrors RewardConfig defaults)
// ═══════════════════════════════════════════════════════

const W_COHERENCE: f64 = 0.20;
const W_CALIBRATION: f64 = 0.34;
const W_PHASE: f64 = 0.23;
const W_EPISTEMIC: f64 = 0.15;
const W_EFFICIENCY: f64 = 0.08;

const COHERENCE_CV_SCALE: f64 = 0.5;
const COHERENCE_WINDOW: usize = 16;
const ENTROPY_FLOOR: f64 = 0.3;
const ENTROPY_FLOOR_PENALTY: f64 = 2.0;

const CALIBRATION_SURPRISE_BASELINE: f64 = 3.0;

const PHASE_CONFUSION_PENALTY: f64 = 2.0;
const PHASE_MONOTONE_PENALTY: f64 = 0.5;
const PHASE_ARC_BONUS: f64 = 0.3;
const PHASE_OSCILLATION_PENALTY: f64 = 1.0;

const EPISTEMIC_SURPRISE_WEIGHT: f64 = 0.6;
const EPISTEMIC_LABEL_WEIGHT: f64 = 0.4;
const EPISTEMIC_UNKNOWN_PENALTY: f64 = 3.0;

const LENGTH_PENALTY_THRESHOLD: usize = 512;
const LENGTH_PENALTY_SCALE: f64 = 0.001;

fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo { lo } else if v > hi { hi } else { v }
}

// ═══════════════════════════════════════════════════════
// R1: Coherence — windowed CV + entropy floor
// ═══════════════════════════════════════════════════════

fn reward_coherence(entropies: &[f64]) -> f64 {
    let n = entropies.len();
    if n == 0 { return 0.0; }

    let mean_h: f64 = entropies.iter().sum::<f64>() / n as f64;

    // Windowed CV
    let window = COHERENCE_WINDOW.min(n);
    let windowed_score = if window < 2 {
        0.0
    } else {
        let step = window / 2; // 50% overlap
        let mut local_cvs: Vec<f64> = Vec::new();
        let mut start = 0;
        while start + window <= n {
            let chunk = &entropies[start..start + window];
            let w_mean: f64 = chunk.iter().sum::<f64>() / chunk.len() as f64;
            if w_mean < 1e-6 {
                local_cvs.push(0.0);
            } else {
                let w_var: f64 = chunk.iter().map(|h| (h - w_mean).powi(2)).sum::<f64>()
                    / chunk.len() as f64;
                local_cvs.push(w_var.sqrt() / w_mean);
            }
            start += step;
        }
        if local_cvs.is_empty() {
            0.0
        } else {
            let avg_cv: f64 = local_cvs.iter().sum::<f64>() / local_cvs.len() as f64;
            clamp(0.6 - COHERENCE_CV_SCALE * avg_cv, -1.0, 0.6)
        }
    };

    // Entropy floor guard
    let floor_penalty = if mean_h < ENTROPY_FLOOR {
        let deficit = (ENTROPY_FLOOR - mean_h) / ENTROPY_FLOOR.max(1e-6);
        (ENTROPY_FLOOR_PENALTY * deficit).min(0.4)
    } else {
        0.0
    };

    clamp(windowed_score - floor_penalty, -1.0, 1.0)
}

// ═══════════════════════════════════════════════════════
// R2: Calibration — confidence × excess surprise
// ═══════════════════════════════════════════════════════

fn reward_calibration(confidences: &[f64], surprises: &[f64]) -> f64 {
    let n = confidences.len();
    if n == 0 { return 0.0; }

    let mut miscal_sum = 0.0f64;
    let mut miscal_count = 0usize;

    for i in 0..n {
        let excess = surprises[i] - CALIBRATION_SURPRISE_BASELINE;
        if excess > 0.0 {
            miscal_sum += confidences[i] * excess;
            miscal_count += 1;
        }
    }

    if miscal_count == 0 {
        return 1.0;
    }

    let mean_miscal = miscal_sum / n as f64;
    clamp(1.0 - 4.0 * mean_miscal, -1.0, 1.0)
}

// ═══════════════════════════════════════════════════════
// R3: Phase Quality — diversity + arc + confusion
// ═══════════════════════════════════════════════════════

/// phases: 0=fluent, 1=recall, 2=reasoning, 3=exploration, 4=confusion
/// decisions: 0=FAST, 1=NORMAL, 2=DEEP
fn reward_phase_quality(
    entropies: &[f64],
    phases: &[u8],
    decisions: &[u8],
) -> f64 {
    let n = phases.len();
    if n == 0 { return 0.0; }

    // Sub-signal 1: Phase diversity (Shannon entropy)
    let mut phase_counts = [0u32; 5];
    for &p in phases {
        if (p as usize) < 5 {
            phase_counts[p as usize] += 1;
        }
    }
    let n_unique = phase_counts.iter().filter(|&&c| c > 0).count();

    let diversity_score = if n_unique <= 1 {
        -PHASE_MONOTONE_PENALTY
    } else {
        let nf = n as f64;
        let mut phase_entropy = 0.0f64;
        for &count in &phase_counts {
            if count > 0 {
                let p_i = count as f64 / nf;
                phase_entropy -= p_i * (p_i + 1e-12).ln();
            }
        }
        let max_entropy = (n_unique.max(2) as f64).ln();
        let normalized = phase_entropy / max_entropy;
        0.4 * normalized
    };

    // Sub-signal 2: Reasoning arc
    let arc_score = if n >= 8 {
        let third = n / 3;
        let h_early: f64 = entropies[..third].iter().sum::<f64>() / third.max(1) as f64;
        let h_mid: f64 = entropies[third..2 * third].iter().sum::<f64>() / third.max(1) as f64;
        let remaining = n - 2 * third;
        let h_late: f64 = entropies[2 * third..].iter().sum::<f64>() / remaining.max(1) as f64;

        let has_exploration = h_mid > h_early * 1.05;
        let has_resolution = h_late < h_mid * 0.95;
        PHASE_ARC_BONUS * (
            if has_exploration { 0.5 } else { 0.0 }
            + if has_resolution { 0.5 } else { 0.0 }
        )
    } else {
        0.0
    };

    // Sub-signal 3: Confusion penalty + cognitive recovery
    let confusion_count = phases.iter().filter(|&&p| p == 4).count();
    let confusion_ratio = confusion_count as f64 / n as f64;

    let mut transitions = 0usize;
    for i in 1..n {
        if phases[i] != phases[i - 1] {
            transitions += 1;
        }
    }
    let oscillation_rate = transitions as f64 / (n - 1).max(1) as f64;

    // Cognitive recovery detection
    let mut recovered = false;
    if confusion_count > 0 {
        let mut in_confusion = false;
        let mut had_deep_in_confusion = false;
        for i in 0..n {
            if phases[i] == 4 {
                in_confusion = true;
                if decisions[i] == 2 { // DEEP
                    had_deep_in_confusion = true;
                }
            } else if in_confusion {
                if had_deep_in_confusion && (phases[i] == 2 || phases[i] == 1 || phases[i] == 0) {
                    recovered = true;
                    break;
                }
                in_confusion = false;
                had_deep_in_confusion = false;
            }
        }
    }

    let (penalty, recovery_bonus) = if recovered {
        (0.0, (confusion_ratio * 2.0).min(0.3))
    } else {
        let p = PHASE_CONFUSION_PENALTY * confusion_ratio * 0.5
            + PHASE_OSCILLATION_PENALTY * (oscillation_rate - 0.4).max(0.0);
        (p, 0.0)
    };

    clamp(diversity_score + arc_score + recovery_bonus - penalty, -1.0, 1.0)
}

// ═══════════════════════════════════════════════════════
// R4: Epistemic Honesty — surprise divergence + labels
// ═══════════════════════════════════════════════════════

/// epistemic_states: 0=KNOWN, 1=LIKELY, 2=UNCERTAIN, 3=UNKNOWN
/// boundary_actions: 0=GENERATE, 1=HEDGE, 2=SEEK, 3=REFUSE
fn reward_epistemic_honesty(
    confidences: &[f64],
    surprises: &[f64],
    epistemic_states: &[u8],
    decisions: &[u8],
    boundary_actions: &[u8],
) -> f64 {
    let n = confidences.len();
    if n == 0 { return 0.0; }

    // Sub-signal 1: Surprise-confidence divergence
    let mut divergence_sum = 0.0f64;
    for i in 0..n {
        let norm_surprise = (surprises[i] / CALIBRATION_SURPRISE_BASELINE.max(1e-6)).min(1.0);
        let ideal_confidence = 1.0 - norm_surprise;
        divergence_sum += (confidences[i] - ideal_confidence).abs();
    }
    let mean_divergence = divergence_sum / n as f64;
    let surprise_score = clamp(1.0 - 3.0 * mean_divergence, -1.0, 1.0);

    // Sub-signal 2: Label-based honesty
    let mut honest_count = 0.0f64;
    let mut dishonest_penalty = 0.0f64;

    for i in 0..n {
        let is_uncertain = epistemic_states[i] >= 2; // UNCERTAIN or UNKNOWN
        let is_confident_output = confidences[i] > 0.7;

        if is_uncertain && is_confident_output {
            let weight = if epistemic_states[i] == 3 { EPISTEMIC_UNKNOWN_PENALTY } else { 1.0 };
            dishonest_penalty += weight;
        } else if is_uncertain && decisions[i] == 2 { // DEEP
            honest_count += 1.5;
        } else if is_uncertain && (boundary_actions[i] == 1 || boundary_actions[i] == 3) { // HEDGE or REFUSE
            honest_count += 1.0;
        } else if !is_uncertain && is_confident_output {
            honest_count += 1.0;
        }
    }

    let honest_ratio = honest_count / n as f64;
    let dishonest_ratio = dishonest_penalty / n as f64;
    let label_score = clamp(honest_ratio - dishonest_ratio, -1.0, 1.0);

    clamp(EPISTEMIC_SURPRISE_WEIGHT * surprise_score + EPISTEMIC_LABEL_WEIGHT * label_score, -1.0, 1.0)
}

// ═══════════════════════════════════════════════════════
// R5: Efficiency — decision appropriateness
// ═══════════════════════════════════════════════════════

fn reward_efficiency(
    z_scores: &[f64],
    token_entropies: &[f64],
    decisions: &[u8],
) -> f64 {
    let n = z_scores.len();
    if n == 0 { return 0.0; }

    let deep_count = decisions.iter().filter(|&&d| d == 2).count();

    // Decision appropriateness
    let mut appropriate = 0.0f64;
    for i in 0..n {
        match decisions[i] {
            0 => { // FAST
                if z_scores[i] <= 0.0 {
                    appropriate += 1.0;
                } else if z_scores[i] <= 0.5 {
                    appropriate += 0.5;
                }
            },
            2 => { // DEEP
                if z_scores[i] > 0.3 {
                    appropriate += 1.0;
                } else if z_scores[i] > -0.2 {
                    appropriate += 0.5;
                }
            },
            _ => { // NORMAL
                if z_scores[i] > 0.4 {
                    // cognitive laziness
                } else if z_scores[i] > 0.0 {
                    appropriate += 0.35;
                } else {
                    appropriate += 0.7;
                }
            },
        }
    }
    let appropriateness = appropriate / n as f64;

    // DEEP resolution bonus
    let mut resolved_deep = 0usize;
    for i in 0..n.saturating_sub(1) {
        if decisions[i] == 2 {
            let end = (i + 4).min(n);
            for j in (i + 1)..end {
                if token_entropies[j] < token_entropies[i] * 0.7 {
                    resolved_deep += 1;
                    break;
                }
            }
        }
    }
    let resolution_rate = resolved_deep as f64 / deep_count.max(1) as f64;
    let resolution_bonus = resolution_rate * 0.3;

    // Length factor
    let length_factor = (n as f64 / 40.0).min(1.0);

    clamp((appropriateness + resolution_bonus) * length_factor, -1.0, 1.0)
}

// ═══════════════════════════════════════════════════════
// PyO3 exposed class
// ═══════════════════════════════════════════════════════

#[pyclass]
pub struct RewardComputerNative;

#[pymethods]
impl RewardComputerNative {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Compute all 5 reward components + total from pre-extracted arrays.
    ///
    /// Args:
    ///   entropies: semantic_entropy per event
    ///   token_entropies: token_entropy per event
    ///   confidences: confidence per event
    ///   surprises: token_surprise per event
    ///   z_scores: z_score per event
    ///   decisions: 0=FAST, 1=NORMAL, 2=DEEP
    ///   phases: 0=fluent, 1=recall, 2=reasoning, 3=exploration, 4=confusion
    ///   epistemic_states: 0=KNOWN, 1=LIKELY, 2=UNCERTAIN, 3=UNKNOWN
    ///   boundary_actions: 0=GENERATE, 1=HEDGE, 2=SEEK, 3=REFUSE
    ///
    /// Returns: dict with total, coherence, calibration, phase_quality,
    ///          epistemic_honesty, efficiency, length_penalty, completeness_bonus
    #[pyo3(signature = (entropies, token_entropies, confidences, surprises, z_scores, decisions, phases, epistemic_states, boundary_actions))]
    fn compute(
        &self,
        entropies: Vec<f64>,
        token_entropies: Vec<f64>,
        confidences: Vec<f64>,
        surprises: Vec<f64>,
        z_scores: Vec<f64>,
        decisions: Vec<u8>,
        phases: Vec<u8>,
        epistemic_states: Vec<u8>,
        boundary_actions: Vec<u8>,
    ) -> PyResult<HashMap<String, f64>> {
        let n = entropies.len();
        let mut result = HashMap::new();

        // Degeneration guard
        if n < 2 {
            result.insert("total".into(), 0.0);
            result.insert("coherence".into(), 0.0);
            result.insert("calibration".into(), 0.0);
            result.insert("phase_quality".into(), 0.0);
            result.insert("epistemic_honesty".into(), 0.0);
            result.insert("efficiency".into(), 0.0);
            result.insert("length_penalty".into(), 0.0);
            result.insert("completeness_bonus".into(), 0.0);
            result.insert("degenerate".into(), 0.0);
            return Ok(result);
        }

        let mean_h: f64 = entropies.iter().sum::<f64>() / n as f64;
        let var_h: f64 = entropies.iter().map(|h| (h - mean_h).powi(2)).sum::<f64>() / n as f64;
        let unique_decisions: std::collections::HashSet<u8> = decisions.iter().copied().collect();

        if var_h < 0.001 && unique_decisions.len() <= 1 {
            result.insert("total".into(), -1.0);
            result.insert("coherence".into(), -1.0);
            result.insert("calibration".into(), 0.0);
            result.insert("phase_quality".into(), 0.0);
            result.insert("epistemic_honesty".into(), 0.0);
            result.insert("efficiency".into(), 0.0);
            result.insert("length_penalty".into(), 0.0);
            result.insert("completeness_bonus".into(), 0.0);
            result.insert("degenerate".into(), 1.0);
            result.insert("entropy_var".into(), var_h);
            return Ok(result);
        }

        // Compute components
        let r_coh = reward_coherence(&entropies);
        let r_cal = reward_calibration(&confidences, &surprises);
        let r_phase = reward_phase_quality(&entropies, &phases, &decisions);
        let r_epist = reward_epistemic_honesty(
            &confidences, &surprises, &epistemic_states, &decisions, &boundary_actions,
        );
        let r_eff = reward_efficiency(&z_scores, &token_entropies, &decisions);

        // Completeness bonus
        let completeness_bonus = if n > 30 && r_coh > 0.0 && r_cal > 0.0 {
            ((n as f64 - 30.0) * 0.005).min(0.25)
        } else {
            0.0
        };

        // Length penalty
        let length_penalty = if n > LENGTH_PENALTY_THRESHOLD {
            (n - LENGTH_PENALTY_THRESHOLD) as f64 * LENGTH_PENALTY_SCALE
        } else {
            0.0
        };

        let total = W_COHERENCE * r_coh
            + W_CALIBRATION * r_cal
            + W_PHASE * r_phase
            + W_EPISTEMIC * r_epist
            + W_EFFICIENCY * r_eff
            + completeness_bonus
            - length_penalty;

        result.insert("total".into(), total);
        result.insert("coherence".into(), r_coh);
        result.insert("calibration".into(), r_cal);
        result.insert("phase_quality".into(), r_phase);
        result.insert("epistemic_honesty".into(), r_epist);
        result.insert("efficiency".into(), r_eff);
        result.insert("length_penalty".into(), length_penalty);
        result.insert("completeness_bonus".into(), completeness_bonus);
        result.insert("degenerate".into(), 0.0);

        Ok(result)
    }
}
