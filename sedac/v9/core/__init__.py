"""
SEDAC V9.0 Core Module

Universal Early Exit Engine for LLMs
"""
from .sedac_engine import (
    SEDACConfig,
    SEDACState,
    SEDACDecisionMaker,
    SEDACTokenRouter,
    SEDACEngine,
    ModelAdapter,
    HuggingFaceAdapter,
    create_sedac_engine,
)

from .ghost_kv import (
    GhostKVConfig,
    GhostKVGenerator,
    GhostKVTrainer,
    create_ghost_kv_for_model,
)

__all__ = [
    # Engine
    "SEDACConfig",
    "SEDACState",
    "SEDACDecisionMaker",
    "SEDACTokenRouter",
    "SEDACEngine",
    "ModelAdapter",
    "HuggingFaceAdapter",
    "create_sedac_engine",
    # Ghost KV
    "GhostKVConfig",
    "GhostKVGenerator",
    "GhostKVTrainer",
    "create_ghost_kv_for_model",
]

__version__ = "9.0.0"
