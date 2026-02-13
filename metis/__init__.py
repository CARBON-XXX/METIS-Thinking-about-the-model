"""
METIS - Metacognitive Entropy-driven Thinking & Introspection System
=====================================================================

Metacognitive Infrastructure for AGI

Named after Μῆτις — the Greek Titaness of wisdom, cunning counsel, and deep thought.

Core Philosophy:
    Not making AI faster, but making AI wiser.
    Know when to speak vs. when to think deeply.
    Know what it knows and what it doesn't know.
    Record confusion and drive self-evolution.

Four Core Capabilities:
    1. Cognitive Switch   - Kahneman System 1/2 dual-process switching
    2. Boundary Guard     - Epistemic boundary detection, anti-hallucination
    3. Curiosity Driver   - Autonomous knowledge gap recording for self-evolution
    4. MetacognitiveCore  - Introspection and self-assessment

Quick Start:
    from metis import Metis, Decision, BoundaryAction

    metis = Metis.attach(model)

    signal = metis.step(logits)

    if signal.decision == Decision.DEEP:
        # System 2: trigger CoT / MCTS
        ...

    if signal.boundary_action == BoundaryAction.SEEK:
        # Trigger RAG / Tool Call
        ...

"""

__version__ = "10.0.0"
__author__ = "CARBON-XXX"

from .metis import Metis
from .inference import MetisInference

from .core.types import (
    Decision,
    EpistemicState,
    BoundaryAction,
    CognitiveSignal,
    CognitiveEvent,
    CognitiveTrace,
    ControllerConfig,
    KnowledgeGap,
    LatencyProfile,
    MetaJudgment,
    SemanticEntropyResult,
    GenerationSample,
    SemanticCluster,
    InferenceResult,
)
from .core.semantic_entropy import SemanticEntropyEstimator
from .cognitive.metacognition import MetacognitiveCore
from .cognitive.phase import CognitivePhase, CognitivePhaseDetector

__all__ = [
    "Metis",
    "MetisInference",
    "SemanticEntropyEstimator",
    "MetacognitiveCore",
    "Decision",
    "EpistemicState",
    "BoundaryAction",
    "CognitiveSignal",
    "CognitiveEvent",
    "CognitiveTrace",
    "ControllerConfig",
    "KnowledgeGap",
    "LatencyProfile",
    "MetaJudgment",
    "SemanticEntropyResult",
    "GenerationSample",
    "SemanticCluster",
    "InferenceResult",
    "CognitivePhase",
    "CognitivePhaseDetector",
    "__version__",
]
