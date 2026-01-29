"""
SEDAC V9.0 Production Module

生产级语义熵引导动态注意力核心引擎

符合 NVIDIA TensorRT / Triton Inference Server 标准
"""
from .config import (
    ProductionConfig,
    ModelConfig,
    SEDACConfig,
    PerformanceConfig,
    DeploymentMode,
    PrecisionMode,
    KernelBackend,
    get_default_config,
)
from .metrics import (
    MetricsCollector,
    PerformanceMonitor,
    LatencyStats,
    ThroughputStats,
    SEDACStats,
)
from .engine import (
    ProductionSEDACEngine,
    EntropyComputer,
    AdaptiveThresholdController,
    GhostKVGenerator,
    O1ReasoningController,
    ForwardOutput,
)
from .inference import (
    SEDACInferencePipeline,
    GenerationConfig,
    InferenceResult,
    create_pipeline,
)
from .trainer import (
    GhostKVTrainer,
    TrainingConfig,
    TrainingState,
    train_ghost_kv_from_model,
)
from .benchmark import (
    ProductionBenchmark,
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkReport,
    run_production_benchmark,
)

__all__ = [
    # Config
    "ProductionConfig",
    "ModelConfig",
    "SEDACConfig",
    "PerformanceConfig",
    "DeploymentMode",
    "PrecisionMode",
    "KernelBackend",
    "get_default_config",
    # Metrics
    "MetricsCollector",
    "PerformanceMonitor",
    "LatencyStats",
    "ThroughputStats",
    "SEDACStats",
    # Engine
    "ProductionSEDACEngine",
    "EntropyComputer",
    "AdaptiveThresholdController",
    "GhostKVGenerator",
    "O1ReasoningController",
    "ForwardOutput",
    # Inference
    "SEDACInferencePipeline",
    "GenerationConfig",
    "InferenceResult",
    "create_pipeline",
    # Training
    "GhostKVTrainer",
    "TrainingConfig",
    "TrainingState",
    "train_ghost_kv_from_model",
    # Benchmark
    "ProductionBenchmark",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkReport",
    "run_production_benchmark",
]

__version__ = "9.0.0"
__author__ = "SEDAC Team"
