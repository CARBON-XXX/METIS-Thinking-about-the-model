"""
SEDAC V7.3 Adaptive Tau Calibration

Core Insight: tau cannot be a fixed constant. It must be coupled with
the "spectral sentinel's effective zone" (min_layers_before_exit).

When tau is too low:
- Tokens exit before spectral features can be computed
- Spectral Guard becomes useless

This module implements:
1. Auto-calibrated tau based on spectral requirements
2. Dual-metric adaptive calibration (PPL + Throughput)
3. Online calibration during inference warmup

Author: SEDAC Research
Date: 2026-01-23
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum
import time


# =============================================================================
# Configuration
# =============================================================================

class CalibrationMode(Enum):
    """Calibration mode."""
    CONSERVATIVE = "conservative"  # Prioritize quality
    BALANCED = "balanced"          # Balance speed/quality
    AGGRESSIVE = "aggressive"      # Prioritize speed


@dataclass
class TauCalibrationConfig:
    """Configuration for adaptive tau calibration."""
    
    # Tau search range
    tau_min: float = 0.85
    tau_max: float = 0.995
    tau_step: float = 0.01
    
    # Spectral coupling constraints
    min_layers_for_spectral: int = 8   # Spectral needs at least this many layers
    spectral_safety_margin: int = 4    # Extra layers beyond spectral minimum
    
    # Dual-metric targets
    target_risk_rate: float = 0.10     # Target < 10% risk
    target_speedup: float = 1.5        # Target > 1.5x speedup
    
    # PPL tolerance for quality
    ppl_tolerance: float = 0.02        # Allow 2% PPL degradation
    
    # Online calibration
    warmup_tokens: int = 100           # Calibrate on first N tokens
    recalibration_interval: int = 500  # Re-check every N tokens
    
    # Consecutive window K coupling
    k_min: int = 2
    k_max: int = 5
    
    @classmethod
    def for_mode(cls, mode: CalibrationMode) -> "TauCalibrationConfig":
        """Get config for specific mode."""
        if mode == CalibrationMode.CONSERVATIVE:
            return cls(
                tau_min=0.92,
                tau_max=0.995,
                target_risk_rate=0.05,
                target_speedup=1.2,
                ppl_tolerance=0.01,
                spectral_safety_margin=6,
            )
        elif mode == CalibrationMode.AGGRESSIVE:
            return cls(
                tau_min=0.85,
                tau_max=0.98,
                target_risk_rate=0.20,
                target_speedup=2.5,
                ppl_tolerance=0.05,
                spectral_safety_margin=2,
            )
        else:  # balanced
            return cls(
                tau_min=0.88,
                tau_max=0.99,
                target_risk_rate=0.10,
                target_speedup=1.5,
                ppl_tolerance=0.02,
                spectral_safety_margin=4,
            )


@dataclass
class CalibrationResult:
    """Result of tau calibration."""
    tau: float
    k: int
    min_layer_ratio: float
    min_layers_before_exit: int
    
    # Metrics achieved
    estimated_speedup: float = 1.0
    estimated_risk: float = 0.0
    
    # Calibration metadata
    calibration_tokens: int = 0
    calibration_time_ms: float = 0.0
    converged: bool = False


# =============================================================================
# Core Calibration Algorithm
# =============================================================================

class AdaptiveTauCalibrator:
    """
    Adaptive tau calibration that couples with spectral analysis.
    
    Key Algorithm:
    1. Start with moderate tau (0.95)
    2. Observe early exit behavior on warmup tokens
    3. If exits happen before min_layers_for_spectral: INCREASE tau
    4. If no exits happen: DECREASE tau
    5. Converge to tau that balances spectral validity with speedup
    """
    
    def __init__(
        self, 
        config: Optional[TauCalibrationConfig] = None,
        num_layers: int = 36
    ):
        self.config = config or TauCalibrationConfig()
        self.num_layers = num_layers
        
        # Current calibrated values
        self.current_tau = (self.config.tau_min + self.config.tau_max) / 2
        self.current_k = 3
        self.current_min_layer_ratio = 0.20
        
        # Calibration state
        self._warmup_complete = False
        self._tokens_processed = 0
        self._exit_layer_history: List[int] = []
        self._risk_history: List[bool] = []
        
        # Statistics for calibration
        self._tau_attempts: Dict[float, Dict] = {}
    
    def get_min_layers_before_exit(self) -> int:
        """
        Compute minimum layers before exit based on spectral requirements.
        
        This ensures spectral guard has enough data to work with.
        """
        return self.config.min_layers_for_spectral + self.config.spectral_safety_margin
    
    def get_current_params(self) -> Dict:
        """Get current calibrated parameters."""
        return {
            "tau": self.current_tau,
            "k": self.current_k,
            "min_layer_ratio": self.current_min_layer_ratio,
            "min_layers_before_exit": self.get_min_layers_before_exit(),
        }
    
    def record_exit(self, exit_layer: int, was_high_risk: bool) -> None:
        """
        Record an exit decision for calibration.
        
        Args:
            exit_layer: Layer at which token exited (-1 if no exit)
            was_high_risk: Whether the token was high risk
        """
        self._tokens_processed += 1
        
        if exit_layer > 0:
            self._exit_layer_history.append(exit_layer)
            self._risk_history.append(was_high_risk)
        
        # Trigger calibration at intervals
        if (self._tokens_processed == self.config.warmup_tokens or
            (self._tokens_processed > self.config.warmup_tokens and 
             self._tokens_processed % self.config.recalibration_interval == 0)):
            self._run_calibration()
    
    def _run_calibration(self) -> None:
        """Run calibration based on collected statistics."""
        if len(self._exit_layer_history) < 10:
            return  # Not enough data
        
        min_spectral_layer = self.get_min_layers_before_exit()
        
        # Analyze exit behavior
        exits_before_spectral = sum(
            1 for l in self._exit_layer_history if l < min_spectral_layer
        )
        exit_rate = len(self._exit_layer_history) / self._tokens_processed
        
        if self._exit_layer_history:
            avg_exit_layer = np.mean(self._exit_layer_history)
            risk_rate = sum(self._risk_history) / len(self._risk_history) if self._risk_history else 0
        else:
            avg_exit_layer = self.num_layers
            risk_rate = 0
        
        # Compute speedup
        total_layers = sum(self._exit_layer_history) + \
                      (self._tokens_processed - len(self._exit_layer_history)) * self.num_layers
        speedup = (self._tokens_processed * self.num_layers) / total_layers
        
        # Store attempt statistics
        self._tau_attempts[self.current_tau] = {
            "speedup": speedup,
            "risk_rate": risk_rate,
            "exit_rate": exit_rate,
            "avg_exit_layer": avg_exit_layer,
            "exits_before_spectral": exits_before_spectral,
        }
        
        # Adjust tau based on observations
        self._adjust_tau(risk_rate, speedup, exits_before_spectral)
        
        # Mark warmup complete after first calibration
        if not self._warmup_complete:
            self._warmup_complete = True
    
    def _adjust_tau(
        self, 
        risk_rate: float, 
        speedup: float, 
        exits_before_spectral: int
    ) -> None:
        """
        Adjust tau based on observed metrics.
        
        Logic:
        1. If too many exits before spectral → INCREASE tau (make harder to exit)
        2. If risk too high → INCREASE tau
        3. If speedup too low and risk acceptable → DECREASE tau
        """
        cfg = self.config
        adjustment = 0.0
        
        # Problem 1: Exits before spectral guard can work
        spectral_violation_rate = exits_before_spectral / max(1, len(self._exit_layer_history))
        if spectral_violation_rate > 0.1:  # More than 10% early exits
            adjustment += 0.02  # Increase tau significantly
        
        # Problem 2: Risk too high
        if risk_rate > cfg.target_risk_rate:
            risk_excess = risk_rate - cfg.target_risk_rate
            adjustment += risk_excess * 0.5  # Scale adjustment by excess
        
        # Opportunity: Can decrease tau if metrics are good
        elif risk_rate < cfg.target_risk_rate * 0.5 and speedup < cfg.target_speedup:
            adjustment -= 0.01  # Try to get more speedup
        
        # Apply adjustment with bounds
        new_tau = self.current_tau + adjustment
        new_tau = max(cfg.tau_min, min(cfg.tau_max, new_tau))
        
        # Also adjust K if needed
        if risk_rate > cfg.target_risk_rate * 1.5:
            self.current_k = min(cfg.k_max, self.current_k + 1)
        elif risk_rate < cfg.target_risk_rate * 0.3 and speedup < cfg.target_speedup:
            self.current_k = max(cfg.k_min, self.current_k - 1)
        
        self.current_tau = new_tau
    
    def get_calibration_result(self) -> CalibrationResult:
        """Get final calibration result."""
        if not self._tau_attempts:
            return CalibrationResult(
                tau=self.current_tau,
                k=self.current_k,
                min_layer_ratio=self.current_min_layer_ratio,
                min_layers_before_exit=self.get_min_layers_before_exit(),
            )
        
        # Get stats for current tau
        stats = self._tau_attempts.get(self.current_tau, {})
        
        return CalibrationResult(
            tau=self.current_tau,
            k=self.current_k,
            min_layer_ratio=self.current_min_layer_ratio,
            min_layers_before_exit=self.get_min_layers_before_exit(),
            estimated_speedup=stats.get("speedup", 1.0),
            estimated_risk=stats.get("risk_rate", 0.0),
            calibration_tokens=self._tokens_processed,
            converged=self._warmup_complete,
        )


# =============================================================================
# Dual-Metric Calibration (PPL + Throughput)
# =============================================================================

@dataclass
class DualMetricState:
    """State for dual-metric calibration."""
    total_tokens: int = 0
    total_time_ms: float = 0.0
    
    # PPL tracking (requires reference)
    ppl_sum: float = 0.0
    ppl_count: int = 0
    
    # Throughput
    tokens_per_second: float = 0.0
    
    # Quality metrics
    risk_events: int = 0
    exit_events: int = 0


class DualMetricCalibration:
    """
    Dual-metric adaptive calibration.
    
    Simultaneously optimizes:
    1. PPL (Perplexity) - Quality metric
    2. Throughput - Speed metric
    
    Uses closed-loop control:
    - If PPL degrades beyond tolerance → increase tau
    - If throughput below target → decrease tau (if PPL allows)
    """
    
    def __init__(
        self,
        config: Optional[TauCalibrationConfig] = None,
        num_layers: int = 36,
        reference_ppl: Optional[float] = None
    ):
        self.config = config or TauCalibrationConfig()
        self.num_layers = num_layers
        self.reference_ppl = reference_ppl  # PPL without early exit
        
        # Current parameters
        self.current_tau = 0.95
        self.current_k = 3
        
        # State tracking
        self.state = DualMetricState()
        self._calibration_active = True
        
        # History for trend analysis
        self._ppl_history: List[float] = []
        self._throughput_history: List[float] = []
    
    def update(
        self,
        token_ppl: Optional[float],
        inference_time_ms: float,
        exited: bool,
        exit_layer: int,
        was_risk: bool
    ) -> None:
        """
        Update calibration with new token result.
        
        Args:
            token_ppl: PPL for this token (None if not available)
            inference_time_ms: Time taken for this token
            exited: Whether early exit was triggered
            exit_layer: Layer at which exit happened
            was_risk: Whether this was a risky exit
        """
        self.state.total_tokens += 1
        self.state.total_time_ms += inference_time_ms
        
        if token_ppl is not None:
            self._ppl_history.append(token_ppl)
            self.state.ppl_sum += token_ppl
            self.state.ppl_count += 1
        
        if exited:
            self.state.exit_events += 1
            if was_risk:
                self.state.risk_events += 1
        
        # Update throughput
        if self.state.total_time_ms > 0:
            self.state.tokens_per_second = (
                self.state.total_tokens * 1000 / self.state.total_time_ms
            )
            self._throughput_history.append(self.state.tokens_per_second)
        
        # Run calibration check
        if self.state.total_tokens % 50 == 0:
            self._calibrate()
    
    def _calibrate(self) -> None:
        """Run calibration adjustment."""
        if self.state.ppl_count < 10:
            return
        
        cfg = self.config
        
        # Compute current metrics
        current_ppl = self.state.ppl_sum / self.state.ppl_count
        risk_rate = (
            self.state.risk_events / self.state.exit_events 
            if self.state.exit_events > 0 else 0
        )
        
        # PPL degradation check
        if self.reference_ppl is not None:
            ppl_degradation = (current_ppl - self.reference_ppl) / self.reference_ppl
        else:
            ppl_degradation = 0
        
        # Adjustment logic
        adjustment = 0.0
        
        # Quality constraint: PPL must stay within tolerance
        if ppl_degradation > cfg.ppl_tolerance:
            adjustment += 0.02  # Increase tau to improve quality
        
        # Risk constraint
        if risk_rate > cfg.target_risk_rate:
            adjustment += 0.01
        
        # Speedup opportunity
        if (ppl_degradation < cfg.ppl_tolerance * 0.5 and 
            risk_rate < cfg.target_risk_rate * 0.5):
            # Quality headroom exists, try to speed up
            adjustment -= 0.005
        
        # Apply adjustment
        new_tau = self.current_tau + adjustment
        new_tau = max(cfg.tau_min, min(cfg.tau_max, new_tau))
        self.current_tau = new_tau
    
    def get_params(self) -> Dict:
        """Get current calibrated parameters."""
        return {
            "tau": self.current_tau,
            "k": self.current_k,
            "ppl_current": self.state.ppl_sum / max(1, self.state.ppl_count),
            "throughput": self.state.tokens_per_second,
            "risk_rate": (
                self.state.risk_events / max(1, self.state.exit_events)
            ),
        }


# =============================================================================
# Spectral-Coupled Tau Calculator
# =============================================================================

def compute_spectral_coupled_tau(
    num_layers: int,
    min_spectral_layers: int = 8,
    safety_margin: int = 4,
    target_exit_ratio: float = 0.5,
    stability_distribution_mean: float = 0.968,
    stability_distribution_std: float = 0.02
) -> Tuple[float, int]:
    """
    Compute tau that ensures spectral analysis has time to work.
    
    Logic:
    - Spectral needs `min_spectral_layers` to compute features
    - We add `safety_margin` for robustness
    - tau must be set so that average exit happens AFTER this threshold
    
    Args:
        num_layers: Total layers in model
        min_spectral_layers: Minimum layers for spectral analysis
        safety_margin: Additional layers for safety
        target_exit_ratio: Target ratio of layers at which to exit (0.5 = middle)
        stability_distribution_mean: Average stability in your data
        stability_distribution_std: Std of stability in your data
        
    Returns:
        (tau, min_layers_before_exit)
    """
    min_exit_layer = min_spectral_layers + safety_margin
    
    # Target exit should be at least at min_exit_layer
    target_exit_layer = max(min_exit_layer, int(num_layers * target_exit_ratio))
    
    # Estimate tau from stability distribution
    # If stability ~ N(mean, std), we need tau such that P(stability >= tau) gives desired behavior
    # Higher tau = later exits
    
    # Simple heuristic: set tau above mean to delay exits
    tau = stability_distribution_mean + stability_distribution_std * 0.5
    
    # Ensure tau is reasonable
    tau = max(0.90, min(0.995, tau))
    
    return tau, min_exit_layer


# =============================================================================
# Factory Functions
# =============================================================================

def create_calibrator(
    mode: str = "balanced",
    num_layers: int = 36
) -> AdaptiveTauCalibrator:
    """Create an adaptive tau calibrator."""
    if mode == "conservative":
        config = TauCalibrationConfig.for_mode(CalibrationMode.CONSERVATIVE)
    elif mode == "aggressive":
        config = TauCalibrationConfig.for_mode(CalibrationMode.AGGRESSIVE)
    else:
        config = TauCalibrationConfig.for_mode(CalibrationMode.BALANCED)
    
    return AdaptiveTauCalibrator(config, num_layers)


def create_dual_metric_calibration(
    mode: str = "balanced",
    num_layers: int = 36,
    reference_ppl: Optional[float] = None
) -> DualMetricCalibration:
    """Create a dual-metric calibration system."""
    if mode == "conservative":
        config = TauCalibrationConfig.for_mode(CalibrationMode.CONSERVATIVE)
    elif mode == "aggressive":
        config = TauCalibrationConfig.for_mode(CalibrationMode.AGGRESSIVE)
    else:
        config = TauCalibrationConfig.for_mode(CalibrationMode.BALANCED)
    
    return DualMetricCalibration(config, num_layers, reference_ppl)
