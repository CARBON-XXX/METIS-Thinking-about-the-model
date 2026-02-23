"""
METIS - Metacognitive Entropy-driven Thinking & Introspection System
Metacognitive Infrastructure for AGI

Usage:
    from metis import Metis

    metis = Metis.attach(model)
    signal = metis.step(logits)

    if signal.decision == Decision.DEEP:
        # System 2: CoT / MCTS
        ...

    if signal.boundary_action == BoundaryAction.SEEK:
        # RAG / Tool Call
        ...

    judgment = metis.introspect()
    gap = metis.end_session()
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Optional

from .core.entropy import SemanticEntropyComputer
from .core.controller import AdaptiveController
from .core.semantic_entropy import SemanticEntropyEstimator
from .core.types import (
    CognitiveSignal,
    CognitiveTrace,
    ControllerConfig,
    Decision,
    KnowledgeGap,
    MetaJudgment,
    SemanticEntropyResult,
)
from .cognitive.switch import CognitiveSwitch
from .cognitive.boundary import EpistemicBoundaryGuard
from .cognitive.curiosity import CuriosityDriver
from .cognitive.metacognition import MetacognitiveCore
from .cognitive.phase import CognitivePhaseDetector


class Metis:
    """
    METIS main class - Unified metacognitive core.
    
    Integrates signal processing, decision-making, and cognitive layers
    into a single interface.
    """
    
    def __init__(
        self,
        config: Optional[ControllerConfig] = None,
        knowledge_gap_path: Optional[str] = None,
        se_method: str = "nli",
        se_nli_model: str = "cross-encoder/nli-deberta-v3-base",
        se_n_samples: int = 5,
        se_temperature: float = 0.7,
        se_uncertainty_threshold: float = 0.5,
    ):
        self._config = config or ControllerConfig()
        
        # -- Signal layer: Token-level heuristic (System 1 fast signal) --
        self._entropy = SemanticEntropyComputer(self._config)
        
        # -- Decision layer: Adaptive threshold control --
        self._controller = AdaptiveController(self._config)
        
        # -- Cognitive layer: Three core components --
        self._switch = CognitiveSwitch()
        self._boundary = EpistemicBoundaryGuard()
        self._curiosity = CuriosityDriver(
            storage_path=knowledge_gap_path or "metis_knowledge_gaps.json"
        )
        
        # -- Verification layer: Generation-level SE (System 2, Kuhn et al.) --
        self._se_estimator = SemanticEntropyEstimator(
            method=se_method,
            nli_model_name=se_nli_model,
            n_samples=se_n_samples,
            temperature=se_temperature,
            uncertainty_threshold=se_uncertainty_threshold,
        )
        
        # -- Metacognitive layer: Introspection analysis --
        self._metacognition = MetacognitiveCore()
        
        # -- Cognitive phase detection (higher-level abstraction) --
        self._phase_detector = CognitivePhaseDetector()
        
        # Model/tokenizer references (set by attach())
        self._model: Optional[nn.Module] = None
        self._tokenizer: Optional[object] = None
        
        # Last signal
        self._last_signal: Optional[CognitiveSignal] = None
        
        # Session-level cognitive trace
        self._trace: Optional[CognitiveTrace] = None
        self._step_index = 0
        
        # Trend detection buffer
        self._entropy_buffer: List[float] = []
        self._trend_window: int = 10
    
    # =============================================================
    # Factory Method
    # =============================================================
    
    @classmethod
    def attach(
        cls,
        model: nn.Module,
        tokenizer=None,
        **kwargs,
    ) -> "Metis":
        """
        Attach METIS to a model in one line.
        
        Automatically extracts the embedding matrix for token-level heuristics.
        Stores model/tokenizer references for generation-level SE.
        
        Usage:
            metis = Metis.attach(model, tokenizer)
        """
        instance = cls(**kwargs)
        instance._model = model
        instance._tokenizer = tokenizer
        
        try:
            embed = model.get_input_embeddings()
            if embed is not None and hasattr(embed, 'weight'):
                instance.set_embedding_matrix(embed.weight.data)
        except Exception:
            pass
        
        return instance
    
    # =============================================================
    # Core API
    # =============================================================
    
    def set_embedding_matrix(self, embedding_matrix: torch.Tensor) -> None:
        """Set embedding matrix (for semantic diversity computation)"""
        self._entropy.set_embedding_matrix(embedding_matrix)
    
    @torch.inference_mode()
    def step(self, logits: torch.Tensor) -> CognitiveSignal:
        """
        Process one step of inference logits, return full cognitive signal.
        
        This is the core METIS method. Called once per token generation.
        
        Args:
            logits: Model output logits [batch, seq, vocab] or [batch, vocab]
            
        Returns:
            CognitiveSignal: Complete signal with entropy, decision, and cognitive state
        """
        # 1. Compute semantic entropy
        semantic_entropy, token_entropy, semantic_diversity, confidence = (
            self._entropy.compute(logits)
        )
        
        # 2. Update controller and get decision
        self._controller.update(semantic_entropy, confidence)
        decision = self._controller.decide(semantic_entropy, confidence)
        z_score = self._controller.get_z_score(semantic_entropy)
        
        # 3. Detect entropy trend
        trend = self._detect_trend(semantic_entropy)
        
        # 4. Predictive signals (gradient + momentum)
        entropy_gradient, entropy_momentum = self._controller.get_predictive_signals()

        # 5. Build signal
        signal = CognitiveSignal(
            token_entropy=token_entropy,
            semantic_diversity=semantic_diversity,
            semantic_entropy=semantic_entropy,
            confidence=confidence,
            decision=decision,
            entropy_trend=trend,
            entropy_gradient=entropy_gradient,
            entropy_momentum=entropy_momentum,
            z_score=z_score,
            cusum_alarm=self._controller.stats.get("change_detected", False),
        )
        
        # 5b. Cognitive phase detection (uses windowed signal statistics)
        phase = self._phase_detector.observe(signal)
        signal.cognitive_phase = phase.value
        
        # 5. Epistemic boundary evaluation (using Controller's dynamic z thresholds)
        # Get adaptive thresholds: z_unc ~ 85%, z_unk ~ 98% (skewness/kurtosis corrected)
        z_unc_dyn, z_unk_dyn = self._controller.get_dynamic_z_thresholds()
        
        epistemic_state, boundary_action, explanation = self._boundary.evaluate(
            signal, 
            thresholds=(z_unc_dyn, z_unk_dyn)
        )
        signal.epistemic_state = epistemic_state
        signal.boundary_action = boundary_action
        signal.introspection = explanation
        signal.adaptive_thresholds = (z_unc_dyn, z_unk_dyn)
        
        # 6. Curiosity observation (pass z_score for adaptive gap detection)
        self._curiosity.observe(semantic_entropy, z_score)
        
        # 7. Cognitive switch
        self._switch.process(signal)
        
        # 8. Record to cognitive trace
        if self._trace is not None:
            self._trace.add_event(signal, self._step_index)
            self._step_index += 1
        
        self._last_signal = signal
        return signal
    
    def start_session(self, query: str, context: str = "") -> None:
        """Start a new conversation session"""
        self._controller.reset_session()
        self._boundary.reset()
        self._switch.reset()
        self._phase_detector.reset()
        self._curiosity.start_session(query, context)
        self._entropy_buffer.clear()
        self._last_signal = None
        self._trace = CognitiveTrace(query=query)
        self._step_index = 0
    
    def end_session(self) -> Optional[KnowledgeGap]:
        """
        End session, return any knowledge gap discovered.
        
        Returns:
            KnowledgeGap if the session revealed a knowledge gap, else None
        """
        return self._curiosity.end_session()
    
    def introspect(
        self, se_result: Optional[SemanticEntropyResult] = None
    ) -> MetaJudgment:
        """
        Metacognitive introspection — analyze the cognitive trace of the current session.
        
        Should be called after generation is complete, before end_session().
        
        Args:
            se_result: Optional Kuhn et al. semantic entropy result
            
        Returns:
            MetaJudgment: Metacognitive judgment
        """
        if self._trace is None:
            return MetaJudgment(reasoning="No active session")
        return self._metacognition.introspect(self._trace, se_result)
    
    @property
    def trace(self) -> Optional[CognitiveTrace]:
        """Current session's cognitive trace"""
        return self._trace
    
    # =============================================================
    # Query Interface
    # =============================================================
    
    @property
    def last_signal(self) -> Optional[CognitiveSignal]:
        """Most recent cognitive signal"""
        return self._last_signal
    
    @property
    def model(self) -> Optional[nn.Module]:
        """Attached model reference"""
        return self._model
    
    @property
    def tokenizer(self):
        """Attached tokenizer reference"""
        return self._tokenizer
    
    @property
    def knowledge_gaps(self):
        """All unresolved knowledge gaps"""
        return self._curiosity.get_unresolved_gaps()
    
    @property
    def se_estimator(self) -> SemanticEntropyEstimator:
        """Generation-level semantic entropy estimator (Kuhn et al.)"""
        return self._se_estimator
    
    def feed_surprise(self, surprise: float) -> None:
        """Feed token surprise back to boundary guard and update trace (1-step lag feedback)."""
        self._boundary.feed_surprise(surprise)
        # Back-fill surprise into the last signal and trace event
        # (surprise is computed AFTER step() returns, so we patch retroactively)
        if self._last_signal is not None:
            self._last_signal.token_surprise = surprise
        if self._trace is not None and self._trace.events:
            self._trace.events[-1].token_surprise = surprise

    def get_uncertainty_score(self) -> float:
        """Get boundary guard's accumulated uncertainty score"""
        return self._boundary.get_uncertainty_score()
    
    def record_se_gap(
        self,
        query: str,
        semantic_entropy: float,
        n_clusters: int,
        n_samples: int,
    ) -> None:
        """Record SE-triggered knowledge gap to curiosity driver"""
        self._curiosity.record_se_gap(
            query=query,
            semantic_entropy=semantic_entropy,
            n_clusters=n_clusters,
            n_samples=n_samples,
        )
    
    def regulate(self, meta_judgment: "MetaJudgment") -> dict:
        """Metacognitive regulation: suggest behavioral adjustments based on MetaJudgment"""
        return self._metacognition.regulate(meta_judgment)
    
    # =============================================================
    # Generation-level Semantic Entropy (System 2 Verification)
    # =============================================================
    
    def evaluate_semantic_entropy(
        self,
        prompt: str,
        model: Optional[nn.Module] = None,
        tokenizer=None,
        n_samples: Optional[int] = None,
        chat_template: bool = True,
    ) -> SemanticEntropyResult:
        """
        Kuhn et al. (ICLR 2023) generation-level semantic entropy evaluation.
        
        This is the authoritative System 2 verification method:
        Sample N complete answers -> bidirectional entailment clustering -> compute SE
        
        Called when token-level signals indicate uncertainty for a final verdict.
        
        Args:
            prompt: User input
            model: LLM (defaults to the attached model)
            tokenizer: Tokenizer (defaults to the attached one)
            n_samples: Override default sample count
            chat_template: Whether to use chat template
            
        Returns:
            SemanticEntropyResult: Complete semantic entropy analysis result
            
        Raises:
            ValueError: If model/tokenizer not provided
        """
        m = model or self._model
        t = tokenizer or self._tokenizer
        
        if m is None or t is None:
            raise ValueError(
                "model and tokenizer required for semantic entropy evaluation. "
                "Use Metis.attach(model, tokenizer) or pass them explicitly."
            )
        
        return self._se_estimator.estimate(
            model=m,
            tokenizer=t,
            prompt=prompt,
            n_samples=n_samples,
            chat_template=chat_template,
        )
    
    @property
    def stats(self) -> dict:
        """Complete statistics"""
        return {
            "controller": self._controller.stats,
            "switch": self._switch.stats,
            "uncertainty_score": self._boundary.get_uncertainty_score(),
            "knowledge_gaps": self._curiosity.gap_count,
        }
    
    # =============================================================
    # Internal
    # =============================================================
    
    def _detect_trend(self, entropy: float) -> str:
        """Detect entropy trend"""
        self._entropy_buffer.append(entropy)
        if len(self._entropy_buffer) > self._trend_window * 2:
            self._entropy_buffer = self._entropy_buffer[-self._trend_window * 2:]
        
        if len(self._entropy_buffer) < self._trend_window:
            return "stable"
        
        recent = self._entropy_buffer[-self._trend_window:]
        
        # Linear slope
        n = len(recent)
        x_mean = (n - 1) / 2
        y_mean = sum(recent) / n
        num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / (den + 1e-6)
        
        # Oscillation detection
        diffs = [recent[i+1] - recent[i] for i in range(n - 1)]
        sign_changes = sum(1 for i in range(len(diffs) - 1) if diffs[i] * diffs[i+1] < 0)
        osc_ratio = sign_changes / (len(diffs) - 1 + 1e-6)
        
        if osc_ratio > 0.6:
            return "oscillating"
        if slope > 0.1:
            return "rising"
        if slope < -0.1:
            return "falling"
        return "stable"
