"""
SEDAC - Semantic Entropy-guided Dynamic Attention Core
======================================================

The Entropy Engine: 认知注意力引擎

核心理念:
    不是"算得更快"，而是"算得更少"
    不是"加速器"，而是"认知协处理器"

与DeepSeek-VL2的对偶关系:
    DeepSeek: 空间稀疏性 → "看哪里" (Spatial Attention)
    SEDAC:    时间稀疏性 → "想多深" (Computational Attention)

Quick Start (V9.0):
    from sedac.v9 import CognitiveAttentionEngine
    
    engine = CognitiveAttentionEngine()
    
    for layer_idx, layer in enumerate(model.layers):
        hidden = layer(hidden)
        state = engine.step(hidden, layer_idx, len(model.layers))
        
        if state.should_exit:
            break
        elif state.should_intervene:
            engine.intervene(state.intervention_type, hidden)

Legacy (V7.x):
    from sedac import auto
    monitor = auto()

Modules:
    - v9: Cognitive Attention Engine (推荐)
    - v8: Intuition Network + Metacognition
    - core: Cascade controller, probe inference, exit strategies
    - calibration: Adaptive threshold calibration
"""

__version__ = "9.0.0-alpha"
__author__ = "CARBON-XXX"

# Zero-config API (recommended)
from sedac.auto import auto, SEDAC, SEDACAutoMonitor, create_monitor

# Legacy API
from sedac.core.cascade_controller import CascadeController, LayerConfig, ExitDecision
from sedac.core.exit_strategy import ExitStrategy, HardExit, SoftExit
from sedac.core.probe_inference import LREProbe, ProbeManager

__all__ = [
    # Zero-config (recommended)
    "auto",
    "SEDAC",
    "SEDACAutoMonitor",
    "create_monitor",
    # Legacy
    "CascadeController",
    "LayerConfig", 
    "ExitDecision",
    "ExitStrategy",
    "HardExit",
    "SoftExit",
    "LREProbe",
    "ProbeManager",
    "__version__",
]
