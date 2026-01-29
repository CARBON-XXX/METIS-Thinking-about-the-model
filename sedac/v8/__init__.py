"""
SEDAC V8.0 - The Intuition Layer

从"机械加速器"到"元认知模块"的战略升级

核心理念:
- System 1 (SEDAC): 快速直觉判断
- System 2 (LLM): 慢速深度推理

决策输出:
- CONFIDENT: Early Exit (极速输出)
- UNCERTAIN: Full Inference (跑完全程)
- HALLUCINATION_RISK: Intervention (触发干预)
"""

from sedac.v8.intuition_network import (
    IntuitionNetwork,
    IntuitionConfig,
    IntuitionSignal,
)
from sedac.v8.metacognition import (
    MetacognitionModule,
    Decision,
    InterventionType,
)

__all__ = [
    "IntuitionNetwork",
    "IntuitionConfig", 
    "IntuitionSignal",
    "MetacognitionModule",
    "Decision",
    "InterventionType",
]
