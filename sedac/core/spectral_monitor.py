"""
SEDAC V7.3 Spectral Monitor: Frequency-Domain Analysis for Early Exit

This module implements spectral analysis to detect:
1. Pseudo-Convergence: Tokens that "look stable" but will produce wrong outputs
2. Reasoning Oscillation: Tokens where model hesitates between concepts

Key Insight:
- Pseudo-converged tokens have HIGHER stability_mean but LOWER spectral_centroid
- This paradox is invisible to pure cosine similarity but detectable via FFT

Mathematical Foundation:
- Layer index → Time axis
- Stability sequence → Signal waveform
- FFT reveals frequency structure that distinguishes true vs pseudo convergence

Author: SEDAC Research
Date: 2026-01-23
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


# =============================================================================
# Enums and Configuration
# =============================================================================

class ConvergenceType(Enum):
    """Classification of token convergence behavior."""
    UNKNOWN = "unknown"           # Not enough data yet
    TRUE_CONVERGENCE = "true"     # Safe to exit
    PSEUDO_CONVERGENCE = "pseudo" # Looks stable but risky
    OSCILLATING = "oscillating"   # Model is hesitating
    UNSTABLE = "unstable"         # Clearly not converged


@dataclass
class SpectralConfig:
    """Configuration for spectral analysis."""
    
    # Minimum layers before spectral analysis is valid
    min_layers_for_analysis: int = 8
    
    # Minimum layers before ANY exit is allowed (forces spectral observation)
    # This prevents tokens from exiting before spectral features can be computed
    min_layers_before_exit: int = 12
    
    # Pseudo-convergence detection thresholds
    # Based on empirical analysis: pseudo tokens have higher stability but lower centroid
    pseudo_stability_threshold: float = 0.928  # If stability > this AND centroid < threshold
    pseudo_centroid_threshold: float = 0.210   # Flag as pseudo-convergence
    
    # Oscillation detection
    min_alternations_for_oscillation: int = 4  # At least 2 full cycles
    oscillation_amplitude_threshold: float = 0.03  # Min std to consider oscillation
    
    # Dip detection (sudden stability drops)
    dip_threshold: float = 0.05  # Minimum drop to count as a dip
    max_allowed_dips: int = 2    # Too many dips = unstable
    
    # Early layer analysis window
    early_layer_window: int = 10
    
    # Spectral feature weights for risk scoring
    weight_spectral_centroid: float = 0.35
    weight_stability_mean: float = 0.25
    weight_max_dip: float = 0.20
    weight_early_mean: float = 0.20
    
    @classmethod
    def conservative(cls) -> "SpectralConfig":
        """Conservative: Prioritize quality, flag more tokens as risky."""
        return cls(
            min_layers_for_analysis=8,
            min_layers_before_exit=14,  # Must observe 14 layers before exit
            pseudo_stability_threshold=0.920,
            pseudo_centroid_threshold=0.215,
            min_alternations_for_oscillation=3,
            max_allowed_dips=1,
        )
    
    @classmethod
    def balanced(cls) -> "SpectralConfig":
        """Balanced: Trade-off between speed and safety."""
        return cls(
            min_layers_for_analysis=8,
            min_layers_before_exit=12,  # Must observe 12 layers before exit
            pseudo_stability_threshold=0.928,
            pseudo_centroid_threshold=0.210,
            min_alternations_for_oscillation=4,
            max_allowed_dips=2,
        )
    
    @classmethod
    def aggressive(cls) -> "SpectralConfig":
        """Aggressive: Prioritize speed, but still require minimum observation."""
        return cls(
            min_layers_for_analysis=6,
            min_layers_before_exit=10,  # Must observe 10 layers before exit
            pseudo_stability_threshold=0.940,
            pseudo_centroid_threshold=0.200,
            min_alternations_for_oscillation=5,
            max_allowed_dips=3,
        )


@dataclass
class SpectralSignal:
    """Result of spectral analysis for a token at current layer."""
    
    # Current layer info
    layer_idx: int = 0
    total_layers: int = 0
    
    # Basic stability metrics
    current_stability: float = 0.0
    stability_mean: float = 0.0
    stability_std: float = 0.0
    stability_min: float = 0.0
    
    # Spectral features
    spectral_centroid: float = 0.0
    spectral_spread: float = 0.0
    oscillation_strength: float = 0.0  # AC / Total power
    
    # Trend analysis
    early_mean: float = 0.0
    late_mean: float = 0.0
    stability_trend: float = 0.0  # Slope of linear fit
    
    # Dip analysis
    num_dips: int = 0
    max_dip_depth: float = 0.0
    
    # Oscillation analysis
    zero_crossings: int = 0
    alternation_count: int = 0
    
    # Classification
    convergence_type: ConvergenceType = ConvergenceType.UNKNOWN
    risk_score: float = 0.0  # 0 = safe, 1 = high risk
    
    # Exit recommendation
    safe_to_exit: bool = False
    confidence: float = 0.0


# =============================================================================
# Spectral Calculator
# =============================================================================

class SpectralCalculator:
    """
    Computes spectral features from stability sequence.
    
    This is the core computation engine, stateless per call.
    """
    
    @staticmethod
    def compute_stability(
        h_prev: torch.Tensor,
        h_curr: torch.Tensor
    ) -> float:
        """Compute normalized cosine similarity between two hidden states."""
        cos_sim = torch.nn.functional.cosine_similarity(
            h_prev.unsqueeze(0).float(),
            h_curr.unsqueeze(0).float()
        ).item()
        return (cos_sim + 1.0) / 2.0  # Map [-1, 1] → [0, 1]
    
    @staticmethod
    def compute_spectral_features(stability_sequence: np.ndarray) -> Dict:
        """
        Compute FFT-based spectral features.
        
        Returns dict with:
        - dc_power: Power at frequency 0 (average level)
        - ac_power: Power at non-zero frequencies (variations)
        - oscillation_strength: AC / Total
        - spectral_centroid: Weighted average frequency
        - spectral_spread: Weighted std of frequency
        """
        n = len(stability_sequence)
        if n < 4:
            return {
                "valid": False,
                "dc_power": 0.0,
                "ac_power": 0.0,
                "oscillation_strength": 0.0,
                "spectral_centroid": 0.0,
                "spectral_spread": 0.0,
            }
        
        # FFT on raw signal (not centered)
        fft_result = np.fft.rfft(stability_sequence)
        power_spectrum = np.abs(fft_result) ** 2
        frequencies = np.fft.rfftfreq(n)
        
        total_power = np.sum(power_spectrum)
        if total_power < 1e-10:
            return {
                "valid": True,
                "dc_power": 0.0,
                "ac_power": 0.0,
                "oscillation_strength": 0.0,
                "spectral_centroid": 0.0,
                "spectral_spread": 0.0,
            }
        
        dc_power = power_spectrum[0]
        ac_power = np.sum(power_spectrum[1:])
        
        # Spectral centroid and spread (on AC components only)
        if ac_power > 1e-10 and len(frequencies) > 1:
            ac_freqs = frequencies[1:]
            ac_powers = power_spectrum[1:]
            
            spectral_centroid = float(np.sum(ac_freqs * ac_powers) / ac_power)
            spectral_spread = float(np.sqrt(
                np.sum(ac_powers * (ac_freqs - spectral_centroid) ** 2) / ac_power
            ))
        else:
            spectral_centroid = 0.0
            spectral_spread = 0.0
        
        return {
            "valid": True,
            "dc_power": float(dc_power),
            "ac_power": float(ac_power),
            "oscillation_strength": float(ac_power / total_power),
            "spectral_centroid": spectral_centroid,
            "spectral_spread": spectral_spread,
        }
    
    @staticmethod
    def compute_zero_crossing_analysis(stability_sequence: np.ndarray) -> Dict:
        """
        Analyze zero crossings (crossings of mean) to detect oscillation.
        
        True oscillation requires consecutive alternations (high→low→high→low).
        """
        n = len(stability_sequence)
        if n < 4:
            return {"zero_crossings": 0, "alternation_count": 0}
        
        mean_val = np.mean(stability_sequence)
        deviations = stability_sequence - mean_val
        signs = np.sign(deviations)
        
        # Count zero crossings
        sign_changes = np.diff(signs)
        zero_crossings = int(np.sum(sign_changes != 0))
        
        # Count consecutive alternations
        alternation_count = 0
        current_run = 0
        
        for i in range(1, len(signs)):
            if signs[i] != signs[i-1] and signs[i] != 0 and signs[i-1] != 0:
                current_run += 1
            else:
                if current_run >= 2:
                    alternation_count += current_run
                current_run = 0
        
        if current_run >= 2:
            alternation_count += current_run
        
        return {
            "zero_crossings": zero_crossings,
            "alternation_count": alternation_count,
        }
    
    @staticmethod
    def detect_dips(
        stability_sequence: np.ndarray,
        threshold: float = 0.05
    ) -> Tuple[int, float]:
        """
        Detect significant drops in stability.
        
        Returns: (num_dips, max_dip_depth)
        """
        n = len(stability_sequence)
        if n < 2:
            return 0, 0.0
        
        dips = []
        for i in range(1, n):
            drop = stability_sequence[i-1] - stability_sequence[i]
            if drop > threshold:
                dips.append(drop)
        
        if not dips:
            return 0, 0.0
        
        return len(dips), float(max(dips))
    
    @staticmethod
    def compute_trend(stability_sequence: np.ndarray) -> float:
        """Compute linear trend (slope) of stability sequence."""
        n = len(stability_sequence)
        if n < 2:
            return 0.0
        
        x = np.arange(n)
        # Simple least squares
        x_mean = np.mean(x)
        y_mean = np.mean(stability_sequence)
        
        numerator = np.sum((x - x_mean) * (stability_sequence - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator < 1e-10:
            return 0.0
        
        return float(numerator / denominator)


# =============================================================================
# Spectral Monitor (Stateful)
# =============================================================================

class SpectralMonitor:
    """
    Stateful monitor that tracks spectral features across layers for a token.
    
    Usage:
        monitor = SpectralMonitor(config)
        
        for layer_idx, hidden in enumerate(layers):
            signal = monitor.step(hidden, layer_idx, total_layers)
            
            if signal.convergence_type == ConvergenceType.PSEUDO_CONVERGENCE:
                # Don't exit! This looks stable but is risky
                continue
            
            if signal.safe_to_exit:
                break
        
        monitor.reset()  # For next token
    """
    
    def __init__(self, config: Optional[SpectralConfig] = None):
        self.config = config or SpectralConfig.balanced()
        self.calculator = SpectralCalculator()
        
        # State
        self._prev_hidden: Optional[torch.Tensor] = None
        self._stability_history: List[float] = []
        self._layer_count: int = 0
    
    def reset(self) -> None:
        """Reset state for new token."""
        self._prev_hidden = None
        self._stability_history = []
        self._layer_count = 0
    
    def step(
        self,
        hidden: torch.Tensor,
        layer_idx: int,
        total_layers: int
    ) -> SpectralSignal:
        """
        Process one layer and return spectral signal.
        
        Args:
            hidden: Hidden state tensor [hidden_dim] or [1, hidden_dim]
            layer_idx: Current layer index (0-based)
            total_layers: Total number of layers in model
            
        Returns:
            SpectralSignal with analysis results and recommendations
        """
        signal = SpectralSignal(
            layer_idx=layer_idx,
            total_layers=total_layers
        )
        
        # Flatten if needed
        if hidden.dim() > 1:
            hidden = hidden.squeeze(0)
        
        # Compute stability
        if self._prev_hidden is not None:
            stability = self.calculator.compute_stability(self._prev_hidden, hidden)
            self._stability_history.append(stability)
            signal.current_stability = stability
        
        # Update state
        self._prev_hidden = hidden.detach().clone()
        self._layer_count += 1
        
        # Need minimum layers for spectral analysis
        if len(self._stability_history) < self.config.min_layers_for_analysis:
            signal.convergence_type = ConvergenceType.UNKNOWN
            signal.safe_to_exit = False
            signal.confidence = 0.0
            return signal
        
        # Convert to numpy for analysis
        seq = np.array(self._stability_history)
        
        # Basic statistics
        signal.stability_mean = float(np.mean(seq))
        signal.stability_std = float(np.std(seq))
        signal.stability_min = float(np.min(seq))
        
        # Early/late analysis
        early_window = min(self.config.early_layer_window, len(seq) // 2)
        signal.early_mean = float(np.mean(seq[:early_window]))
        signal.late_mean = float(np.mean(seq[-early_window:]))
        
        # Trend
        signal.stability_trend = self.calculator.compute_trend(seq)
        
        # Spectral features
        spectral = self.calculator.compute_spectral_features(seq)
        if spectral.get("valid", False):
            signal.spectral_centroid = spectral["spectral_centroid"]
            signal.spectral_spread = spectral["spectral_spread"]
            signal.oscillation_strength = spectral["oscillation_strength"]
        
        # Zero-crossing / oscillation
        zc = self.calculator.compute_zero_crossing_analysis(seq)
        signal.zero_crossings = zc["zero_crossings"]
        signal.alternation_count = zc["alternation_count"]
        
        # Dip detection
        num_dips, max_dip = self.calculator.detect_dips(seq, self.config.dip_threshold)
        signal.num_dips = num_dips
        signal.max_dip_depth = max_dip
        
        # Classify convergence type
        signal.convergence_type = self._classify_convergence(signal)
        
        # Compute risk score
        signal.risk_score = self._compute_risk_score(signal)
        
        # Exit recommendation
        signal.safe_to_exit, signal.confidence = self._compute_exit_recommendation(signal)
        
        return signal
    
    def _classify_convergence(self, signal: SpectralSignal) -> ConvergenceType:
        """Classify the convergence type based on spectral features."""
        cfg = self.config
        
        # Check for oscillation first
        if (signal.alternation_count >= cfg.min_alternations_for_oscillation and
            signal.stability_std >= cfg.oscillation_amplitude_threshold):
            return ConvergenceType.OSCILLATING
        
        # Check for instability (too many dips)
        if signal.num_dips > cfg.max_allowed_dips:
            return ConvergenceType.UNSTABLE
        
        # Check for pseudo-convergence
        # Key insight: Pseudo tokens have HIGH stability but LOW spectral centroid
        if (signal.stability_mean > cfg.pseudo_stability_threshold and
            signal.spectral_centroid < cfg.pseudo_centroid_threshold):
            return ConvergenceType.PSEUDO_CONVERGENCE
        
        # If stability is reasonably high and no red flags
        if signal.stability_mean > 0.90 and signal.stability_std < 0.15:
            return ConvergenceType.TRUE_CONVERGENCE
        
        return ConvergenceType.UNSTABLE
    
    def _compute_risk_score(self, signal: SpectralSignal) -> float:
        """
        Compute risk score based on spectral features.
        
        Score 0.0 = very safe
        Score 1.0 = very risky
        """
        cfg = self.config
        
        # Normalize features to [0, 1] risk contribution
        
        # Spectral centroid: lower is riskier
        # Based on data: true=0.214, pseudo=0.207
        centroid_risk = 1.0 - min(1.0, signal.spectral_centroid / 0.25)
        
        # Stability mean: higher can be riskier (pseudo-convergence paradox)
        # Only risky if very high AND centroid is low
        if signal.stability_mean > 0.93 and signal.spectral_centroid < 0.21:
            stability_risk = (signal.stability_mean - 0.90) / 0.10
        else:
            stability_risk = 0.0
        
        # Max dip: larger dips are concerning
        dip_risk = min(1.0, signal.max_dip_depth / 0.40)
        
        # Early mean: higher early mean with pseudo-convergence pattern is risky
        early_risk = max(0, signal.early_mean - 0.94) / 0.06 if signal.early_mean > 0.94 else 0.0
        
        # Weighted combination
        risk = (
            cfg.weight_spectral_centroid * centroid_risk +
            cfg.weight_stability_mean * stability_risk +
            cfg.weight_max_dip * dip_risk +
            cfg.weight_early_mean * early_risk
        )
        
        # Boost risk for known dangerous patterns
        if signal.convergence_type == ConvergenceType.PSEUDO_CONVERGENCE:
            risk = min(1.0, risk + 0.3)
        elif signal.convergence_type == ConvergenceType.OSCILLATING:
            risk = min(1.0, risk + 0.4)
        elif signal.convergence_type == ConvergenceType.UNSTABLE:
            risk = min(1.0, risk + 0.2)
        
        return min(1.0, max(0.0, risk))
    
    def _compute_exit_recommendation(
        self,
        signal: SpectralSignal
    ) -> Tuple[bool, float]:
        """
        Compute exit recommendation based on convergence type and risk.
        
        Returns: (safe_to_exit, confidence)
        """
        # Never exit on dangerous patterns
        if signal.convergence_type in [
            ConvergenceType.PSEUDO_CONVERGENCE,
            ConvergenceType.OSCILLATING,
            ConvergenceType.UNSTABLE,
            ConvergenceType.UNKNOWN,
        ]:
            return False, 0.0
        
        # Only exit on true convergence with low risk
        if signal.convergence_type == ConvergenceType.TRUE_CONVERGENCE:
            if signal.risk_score < 0.3:
                confidence = 1.0 - signal.risk_score
                return True, confidence
            elif signal.risk_score < 0.5:
                # Borderline - could exit but with lower confidence
                confidence = 0.5 * (1.0 - signal.risk_score)
                return True, confidence
        
        return False, 0.0
    
    def get_state_summary(self) -> Dict:
        """Get summary of current state for debugging/logging."""
        if not self._stability_history:
            return {"layers_processed": 0, "ready": False}
        
        seq = np.array(self._stability_history)
        return {
            "layers_processed": len(self._stability_history),
            "ready": len(self._stability_history) >= self.config.min_layers_for_analysis,
            "stability_mean": float(np.mean(seq)),
            "stability_std": float(np.std(seq)),
            "stability_min": float(np.min(seq)),
            "stability_current": float(seq[-1]) if len(seq) > 0 else 0.0,
        }


# =============================================================================
# Integration Helper
# =============================================================================

class SpectralExitGuard:
    """
    High-level guard that wraps existing V7.0 exit logic with spectral checks.
    
    Usage:
        guard = SpectralExitGuard()
        
        # In inference loop
        if v70_says_exit:
            spectral_signal = guard.check(hidden_states_so_far)
            if spectral_signal.safe_to_exit:
                # Actually exit
            else:
                # V7.0 wanted to exit but spectral says it's risky
                # Continue to next layer
    """
    
    def __init__(self, config: Optional[SpectralConfig] = None):
        self.config = config or SpectralConfig.balanced()
        self.monitor = SpectralMonitor(self.config)
    
    def reset(self) -> None:
        """Reset for new token."""
        self.monitor.reset()
    
    def update(
        self,
        hidden: torch.Tensor,
        layer_idx: int,
        total_layers: int
    ) -> SpectralSignal:
        """Update with new layer hidden state."""
        return self.monitor.step(hidden, layer_idx, total_layers)
    
    def should_block_exit(self, signal: SpectralSignal) -> bool:
        """
        Should we block an exit that V7.0 wants to make?
        
        Returns True if spectral analysis says this exit is risky.
        """
        # Block if not enough layers observed for spectral analysis
        if signal.layer_idx < self.config.min_layers_before_exit:
            return True  # Block: need more observation
        
        if signal.convergence_type == ConvergenceType.PSEUDO_CONVERGENCE:
            return True  # Block: looks stable but risky
        
        if signal.convergence_type == ConvergenceType.OSCILLATING:
            return True  # Block: model is hesitating
        
        if signal.risk_score > 0.5:
            return True  # Block: high risk score
        
        return False
    
    def get_exit_delay_recommendation(self, signal: SpectralSignal) -> int:
        """
        If blocking exit, how many more layers should we wait?
        
        Returns recommended additional layers before reconsidering exit.
        """
        if signal.convergence_type == ConvergenceType.OSCILLATING:
            return 5  # Wait longer for oscillation to settle
        
        if signal.convergence_type == ConvergenceType.PSEUDO_CONVERGENCE:
            return 3  # Wait a bit to see if pattern changes
        
        if signal.risk_score > 0.7:
            return 4
        elif signal.risk_score > 0.5:
            return 2
        
        return 1


# =============================================================================
# Factory Functions
# =============================================================================

def create_spectral_monitor(mode: str = "balanced") -> SpectralMonitor:
    """Create a SpectralMonitor with preset configuration."""
    if mode == "conservative":
        return SpectralMonitor(SpectralConfig.conservative())
    elif mode == "aggressive":
        return SpectralMonitor(SpectralConfig.aggressive())
    else:
        return SpectralMonitor(SpectralConfig.balanced())


def create_spectral_guard(mode: str = "balanced") -> SpectralExitGuard:
    """Create a SpectralExitGuard with preset configuration."""
    if mode == "conservative":
        return SpectralExitGuard(SpectralConfig.conservative())
    elif mode == "aggressive":
        return SpectralExitGuard(SpectralConfig.aggressive())
    else:
        return SpectralExitGuard(SpectralConfig.balanced())
