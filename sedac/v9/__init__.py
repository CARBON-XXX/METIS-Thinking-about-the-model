"""
SEDAC V9.0 - The Entropy Engine (Cognitive Attention Engine)

核心理念重新定义:
    SEDAC = Semantic Entropy-guided Dynamic Attention Core
    
    不是"算得更快"，而是"算得更少"
    不是"加速器"，而是"认知注意力引擎"

哲学基础:
    1. 麦克斯韦妖 - 用信息换能量
    2. 自由能原理 - 最小化预测误差
    3. 计算不可约性探测器 - 区分可约/不可约问题

与DeepSeek-VL2的对偶关系:
    - DeepSeek: 空间稀疏性 (看哪里 - Spatial Attention)
    - SEDAC: 时间稀疏性 (想多深 - Computational Attention)
    
    统一理念: 在信息的荒原中，只开采高密度的矿脉

架构:
    ┌─────────────────────────────────────────┐
    │           SEDAC V9.0 Architecture       │
    ├─────────────────────────────────────────┤
    │  Sensor → Intuition → Metacognition     │
    │    ↓          ↓            ↓            │
    │  Features   Signal      Decision        │
    │                            │            │
    │            ┌───────────────┼────────┐   │
    │            ↓               ↓        ↓   │
    │          EXIT          CONTINUE  INTERVENE
    │        (低熵快退)      (继续)    (高熵干预) │
    └─────────────────────────────────────────┘
"""

__version__ = "9.0.0"

# V9.0 CUDA Optimized Core (推荐使用)
try:
    from sedac.v9.core import (
        SEDACConfig,
        SEDACEngine,
        SEDACDecisionMaker,
        SEDACTokenRouter,
        HuggingFaceAdapter,
        create_sedac_engine,
        GhostKVGenerator,
        GhostKVTrainer,
        create_ghost_kv_for_model,
    )
    CUDA_CORE_AVAILABLE = True
except ImportError:
    CUDA_CORE_AVAILABLE = False

# V9.0 核心组件 - 全自主认知引擎
from sedac.v9.adaptive_engine import (
    AdaptiveCognitiveEngine,
    AdaptiveState,
    OnlineStatistics,
    create_adaptive_engine,
)
from sedac.v9.adaptive_trainer import (
    AdaptiveTrainer,
    AdaptiveTrainingConfig,
    AdaptiveDataset,
    AdaptiveLoss,
)
from sedac.v9.intervention import (
    InterventionManager,
    InterventionType,
    InterventionResult,
    SpeculativeVerifier,
    SelfConsistencyChecker,
    DynamicConfidenceCalibrator,
    create_intervention_manager,
)
from sedac.v9.data_augmentation import (
    DataAugmentor,
    TaskType,
    SyntheticSample,
    augment_training_data,
)

# KV Cache Management (工业级)
from sedac.v9.kv_cache_manager import (
    KVCacheManager,
    KVOnlyProjection,
    AdaptiveLayerScheduler,
    SkipMode,
    LayerDecision,
    create_kv_cache_manager,
    create_layer_scheduler,
)

# Triton Kernels (高性能)
from sedac.v9.triton_kernels import (
    TritonSEDACOps,
    PyTorchSEDACOps,
    create_sedac_ops,
    TRITON_AVAILABLE,
)

# SEDAC-O1 (自适应思考时间)
from sedac.v9.sedac_o1 import (
    SEDACO1Engine,
    ThinkingMode,
    ThinkingState,
    ThinkingConfig,
    AdaptiveComputationController,
    create_sedac_o1_engine,
)

# Token Router (工业级Per-Token动态计算)
from sedac.v9.token_router import (
    TokenRouter,
    TokenState,
    RaggedBatch,
    RouterState,
    BatchScheduler,
    create_token_router,
)

# Ghost KV (TinyMLP预测KV，解决Memory-Bound)
from sedac.v9.ghost_kv import (
    GhostKVGenerator,
    GhostKVConfig,
    GhostKVManager,
    CrossLayerStateReuser,
    GhostKVTrainer,
    create_ghost_kv_manager,
)

# Fused GPU Kernel (全图化，零CPU同步)
from sedac.v9.fused_gpu_kernel import (
    FusedSEDACEngine,
    CUDAGraphWrapper,
    create_fused_engine,
)

# Attention Sinks (关键Token保护)
from sedac.v9.attention_sinks import (
    AttentionSinkProtector,
    AttentionSinkDetector,
    AnchorLayerManager,
    DynamicAttentionMask,
    ProtectionLevel,
    create_attention_sink_protector,
)

# Industrial Integrator (工业级集成)
from sedac.v9.industrial_integrator import (
    IndustrialIntegrator,
    IndustrialConfig,
    IntegrationStrategy,
    SEDACLayerWrapper,
    PerTokenIntegrator,
    InferenceMetrics,
    create_industrial_integrator,
)

# Production Layer (生产级Transformer层)
from sedac.v9.production_layer import (
    ProductionSEDACLayer,
    ProductionSEDACModel,
    LayerConfig,
    SEDACAttention,
    SEDACMLP,
    KVOnlyAttention,
    GhostKVProjection,
    create_production_layer,
)

# Legacy V9.0 (固定阈值版本)
from sedac.v9.engine import (
    CognitiveAttentionEngine as LegacyCognitiveEngine,
    AttentionMode,
    AttentionState,
    EngineConfig,
    create_engine,
)
from sedac.v9.trainer import (
    IntuitionTrainer,
    TrainingConfig,
)

# 继承V8基础架构
from sedac.v8.intuition_network import (
    IntuitionNetwork,
    IntuitionConfig,
    IntuitionSignal,
    FeatureExtractor,
)
from sedac.v8.metacognition import (
    MetacognitionModule,
    SEDACv8 as SEDACv9Base,
    Decision,
    InterventionType,
    MetacognitiveState,
)

# V9.0 Production Module (生产级推荐)
try:
    from sedac.v9.production import (
        ProductionConfig,
        ModelConfig,
        SEDACConfig as ProductionSEDACConfig,
        PerformanceConfig,
        ProductionSEDACEngine,
        SEDACInferencePipeline,
        GenerationConfig,
        InferenceResult,
        create_pipeline,
        GhostKVTrainer as ProductionGhostKVTrainer,
        TrainingConfig,
        ProductionBenchmark,
        BenchmarkConfig,
        run_production_benchmark,
        MetricsCollector,
        PerformanceMonitor,
    )
    PRODUCTION_AVAILABLE = True
except ImportError:
    PRODUCTION_AVAILABLE = False

__all__ = [
    # V9.0 CUDA Optimized Core (推荐)
    "SEDACConfig",
    "SEDACEngine",
    "SEDACDecisionMaker",
    "SEDACTokenRouter",
    "HuggingFaceAdapter",
    "create_sedac_engine",
    "CUDA_CORE_AVAILABLE",
    # V9.0 Adaptive Engine (推荐)
    "AdaptiveCognitiveEngine",
    "AdaptiveState",
    "OnlineStatistics",
    "create_adaptive_engine",
    # Adaptive Training
    "AdaptiveTrainer",
    "AdaptiveTrainingConfig",
    "AdaptiveDataset",
    "AdaptiveLoss",
    # Intervention
    "InterventionManager",
    "InterventionType",
    "InterventionResult",
    "SpeculativeVerifier",
    "SelfConsistencyChecker",
    "DynamicConfidenceCalibrator",
    "create_intervention_manager",
    # Data Augmentation
    "DataAugmentor",
    "TaskType",
    "SyntheticSample",
    "augment_training_data",
    # KV Cache Management
    "KVCacheManager",
    "KVOnlyProjection",
    "AdaptiveLayerScheduler",
    "SkipMode",
    "LayerDecision",
    # Triton Kernels
    "TritonSEDACOps",
    "PyTorchSEDACOps",
    "create_sedac_ops",
    # SEDAC-O1
    "SEDACO1Engine",
    "ThinkingMode",
    "ThinkingState",
    "ThinkingConfig",
    "create_sedac_o1_engine",
    # Token Router (工业级)
    "TokenRouter",
    "TokenState",
    "RaggedBatch",
    "RouterState",
    "BatchScheduler",
    "create_token_router",
    # Ghost KV
    "GhostKVGenerator",
    "GhostKVConfig",
    "GhostKVManager",
    "CrossLayerStateReuser",
    "GhostKVTrainer",
    "create_ghost_kv_manager",
    # Fused GPU Kernel
    "FusedSEDACEngine",
    "CUDAGraphWrapper",
    "create_fused_engine",
    # Attention Sinks
    "AttentionSinkProtector",
    "AttentionSinkDetector",
    "AnchorLayerManager",
    "DynamicAttentionMask",
    "ProtectionLevel",
    "create_attention_sink_protector",
    # Industrial Integrator
    "IndustrialIntegrator",
    "IndustrialConfig",
    "IntegrationStrategy",
    "SEDACLayerWrapper",
    "PerTokenIntegrator",
    "InferenceMetrics",
    "create_industrial_integrator",
    # Production Layer
    "ProductionSEDACLayer",
    "ProductionSEDACModel",
    "LayerConfig",
    "SEDACAttention",
    "SEDACMLP",
    "KVOnlyAttention",
    "GhostKVProjection",
    "create_production_layer",
    # Legacy Engine
    "LegacyCognitiveEngine",
    "AttentionMode",
    "AttentionState",
    "EngineConfig",
    "create_engine",
    # Core Components
    "IntuitionNetwork",
    "IntuitionConfig",
    "IntuitionSignal",
    "FeatureExtractor",
    # Metacognition
    "MetacognitionModule",
    "SEDACv9Base",
    "Decision",
    "MetacognitiveState",
    # Version
    "__version__",
    # V9.0 Production Module (生产级)
    "ProductionConfig",
    "ModelConfig",
    "ProductionSEDACConfig",
    "PerformanceConfig",
    "ProductionSEDACEngine",
    "SEDACInferencePipeline",
    "GenerationConfig",
    "InferenceResult",
    "create_pipeline",
    "ProductionGhostKVTrainer",
    "TrainingConfig",
    "ProductionBenchmark",
    "BenchmarkConfig",
    "run_production_benchmark",
    "MetricsCollector",
    "PerformanceMonitor",
    "PRODUCTION_AVAILABLE",
]
