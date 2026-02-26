"""
METIS Experiment Pipeline — Modular Architecture

Phases:
    1. Generate & Score  →  generator_phase.py
    2. DPO Training      →  trainer_phase.py
    3. Evaluation         →  evaluator_phase.py
    4. Report             →  (integrated into evaluator_phase.py)

Shared types and config live in config.py.
"""
from metis.pipeline.config import ExperimentConfig, EvalMetrics, C
from metis.pipeline.generator_phase import phase1_generate
from metis.pipeline.trainer_phase import phase2_train
from metis.pipeline.evaluator_phase import phase3_evaluate, phase4_report

__all__ = [
    "ExperimentConfig",
    "EvalMetrics",
    "C",
    "phase1_generate",
    "phase2_train",
    "phase3_evaluate",
    "phase4_report",
]
