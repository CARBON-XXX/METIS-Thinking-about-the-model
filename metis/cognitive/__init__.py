"""
METIS Cognitive Layer
Cognitive layer - System 1/2 switching, epistemic boundary guard, curiosity driver, metacognition, dynamic CoT
"""
from .switch import CognitiveSwitch
from .boundary import EpistemicBoundaryGuard
from .curiosity import CuriosityDriver
from .metacognition import MetacognitiveCore
from .cot import CoTManager

__all__ = [
    "CognitiveSwitch",
    "EpistemicBoundaryGuard",
    "CuriosityDriver",
    "MetacognitiveCore",
    "CoTManager",
]
