"""
SEDAC Calibration Module
========================

Adaptive threshold calibration strategies:
- AdaptiveThreshold: Basic EMA-based calibration
- DualMetricCalibration: PPL + Throughput aware calibration
- AdaptiveTauCalibrator: Spectral-coupled tau calibration (V7.3)
"""

from sedac.calibration.adaptive_threshold import (
    AdaptiveThreshold,
    DualMetricCalibration,
    RollingWindow,
)

from sedac.calibration.adaptive_tau import (
    AdaptiveTauCalibrator,
    DualMetricCalibration as DualMetricCalibrationV2,
    TauCalibrationConfig,
    CalibrationMode,
    CalibrationResult,
    create_calibrator,
    create_dual_metric_calibration,
    compute_spectral_coupled_tau,
)

__all__ = [
    "AdaptiveThreshold",
    "DualMetricCalibration",
    "RollingWindow",
    "AdaptiveTauCalibrator",
    "DualMetricCalibrationV2",
    "TauCalibrationConfig",
    "CalibrationMode",
    "CalibrationResult",
    "create_calibrator",
    "create_dual_metric_calibration",
    "compute_spectral_coupled_tau",
]
