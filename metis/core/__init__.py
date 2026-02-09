"""
METIS Core: Signal Processing Layer
Signal processing layer - entropy computation, statistics, adaptive thresholds, generation-level semantic entropy
"""
from .entropy import SemanticEntropyComputer
from .semantic_entropy import SemanticEntropyEstimator
from .statistics import SlidingWindowStats
from .controller import AdaptiveController, Decision
from .types import (
    CognitiveSignal,
    CognitiveEvent,
    CognitiveTrace,
    ControllerConfig,
    MetaJudgment,
    SemanticEntropyResult,
    GenerationSample,
    SemanticCluster,
    InferenceResult,
)

__all__ = [
    "SemanticEntropyComputer",
    "SemanticEntropyEstimator",
    "SlidingWindowStats",
    "AdaptiveController",
    "Decision",
    "CognitiveSignal",
    "CognitiveEvent",
    "CognitiveTrace",
    "ControllerConfig",
    "MetaJudgment",
    "SemanticEntropyResult",
    "GenerationSample",
    "SemanticCluster",
    "InferenceResult",
]
