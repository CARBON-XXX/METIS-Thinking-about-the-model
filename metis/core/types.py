"""
METIS Core Types
"""
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Callable, Any, Tuple, Dict
import json


class Decision(Enum):
    """
    Three-level cognitive decision.
    
    Maps to Kahneman's dual-process theory:
    - FAST -> System 1 (intuition, low entropy, automatic)
    - NORMAL -> Standard reasoning
    - DEEP -> System 2 (deliberate reasoning, high entropy, reflective)
    """
    FAST = "fast"       # Low entropy: fast output, no extra compute
    NORMAL = "normal"   # Medium entropy: standard reasoning
    DEEP = "deep"       # High entropy: trigger CoT/MCTS/verification


class CoTStrategy(Enum):
    """Chain-of-Thought injection strategy"""
    NONE = "none"               # No injection
    STANDARD = "standard"       # Standard: "Let me think about this carefully"
    CLARIFICATION = "clarify"   # Clarify: "Let me verify the relevant definitions"
    DECOMPOSITION = "decompose" # Decompose: "This is complex, let me break it down"
    REFLECTION = "reflect"      # Reflect: "Something seems contradictory, let me re-check"


class EpistemicState(Enum):
    """Epistemic confidence level — output of the Epistemic Boundary Guard"""
    KNOWN = "known"             # Confidently known
    LIKELY = "likely"           # Reasonably confident
    UNCERTAIN = "uncertain"     # Uncertain
    UNKNOWN = "unknown"         # Unknown -> should refuse or seek help


class BoundaryAction(Enum):
    """Action triggered by epistemic boundary guard"""
    GENERATE = "generate"       # Normal generation
    HEDGE = "hedge"             # Annotate with uncertainty
    SEEK = "seek"               # Trigger external retrieval (RAG/Tool)
    REFUSE = "refuse"           # Refuse to answer, admit ignorance


@dataclass
class TokenEntropyResult:
    """
    Token-level entropy result (System 1)
    """
    semantic_entropy: float = 0.0       # Combined semantic entropy
    token_entropy: float = 0.0          # Shannon entropy
    semantic_diversity: float = 0.0     # Semantic diversity
    confidence: float = 0.0             # Confidence


@dataclass
class CognitiveSignal:
    """
    Cognitive Signal — the core per-step output of METIS.
    
    This is the primary interface data structure.
    Upstream systems (Agent/RAG/CoT) consume this signal for decision-making.
    """
    # Entropy signals
    token_entropy: float = 0.0          # Shannon entropy
    semantic_entropy: float = 0.0       # Combined semantic entropy
    semantic_diversity: float = 0.0     # Semantic diversity
    confidence: float = 0.0             # Confidence
    
    # Decision
    decision: Decision = Decision.NORMAL
    epistemic_state: EpistemicState = EpistemicState.LIKELY
    boundary_action: BoundaryAction = BoundaryAction.GENERATE

    # Trend
    entropy_trend: str = "stable"       # rising / falling / stable / oscillating
    
    # Introspection (natural language explanation)
    introspection: str = ""

    # -- Predictive cognitive signals --
    token_surprise: float = 0.0             # -log2(p(sampled_token)) — prediction error
    entropy_gradient: float = 0.0           # d(entropy)/dt — instantaneous rate of change
    entropy_momentum: float = 0.0           # EMA of gradient — captures acceleration/deceleration

    # -- Cognitive phase (higher-level abstraction) --
    cognitive_phase: str = "recall"         # fluent / recall / reasoning / exploration / confusion

    # -- Internal state (for debugging/visualization) --
    z_score: float = 0.0                    # Current token z-score
    cusum_alarm: bool = False               # Whether CUSUM change-point alarm triggered
    adaptive_thresholds: Optional[Tuple[float, float]] = None # (z_unc, z_unk) dynamic thresholds

    def __str__(self):
        return (
            f"CognitiveSignal(mode={self.decision.name}, "
            f"H={self.semantic_entropy:.2f}, z={self.z_score:.2f}, "
            f"conf={self.confidence:.2f}, boundary={self.boundary_action.name})"
        )


@dataclass
class ControllerConfig:
    """Adaptive controller configuration"""
    # Sliding window
    window_size: int = 500
    
    # Adaptive forgetting factor
    forgetting_factor: float = 0.995
    
    # Siegmund CUSUM
    target_arl0: int = 200              # Average run length (false alarm interval)
    cusum_k: float = 0.5               # Reference drift
    
    # Decision-theoretic cost ratio
    cost_ratio: float = 5.0            # Miss cost / FalseAlarm cost
    
    # Cold start
    min_samples: int = 10
    cold_start_entropy_mean: float = 1.0
    cold_start_entropy_std: float = 1.0

    # Semantic entropy
    semantic_weight: float = 0.15       # lambda: semantic diversity weight
    top_k: int = 10                     # top-k for semantic distance computation


@dataclass
class KnowledgeGap:
    """
    Knowledge Gap — output of the Curiosity Driver.
    
    Records knowledge deficits discovered during runtime,
    used for targeted learning in the Dreaming Phase.
    """
    query: str
    context: str = ""
    entropy_peak: float = 0.0
    entropy_mean: float = 0.0
    category: str = "unknown"           # complete_unknown / sustained / spike
    timestamp: str = ""
    resolved: bool = False


# =============================================================
# Cognitive Trace (Session-level cognitive event tracking)
# =============================================================

@dataclass
class CognitiveEvent:
    """Single-step cognitive event — atomic unit of CognitiveTrace"""
    step: int
    token_entropy: float = 0.0
    semantic_entropy: float = 0.0
    confidence: float = 0.0
    z_score: float = 0.0
    token_surprise: float = 0.0
    entropy_gradient: float = 0.0
    entropy_momentum: float = 0.0
    cognitive_phase: str = "recall"
    decision: Decision = Decision.NORMAL
    epistemic_state: EpistemicState = EpistemicState.LIKELY
    boundary_action: BoundaryAction = BoundaryAction.GENERATE
    entropy_trend: str = "stable"
    cusum_alarm: bool = False


@dataclass
class CognitiveTrace:
    """
    Session-level cognitive trace — complete introspection data.
    
    Records the cognitive state at each step of an inference session.
    Used by MetacognitiveCore for analysis and post-hoc auditing.
    """
    query: str = ""
    events: List[CognitiveEvent] = field(default_factory=list)
    
    # Session-level aggregate statistics (populated by MetacognitiveCore)
    total_tokens: int = 0
    fast_count: int = 0
    deep_count: int = 0
    hedge_count: int = 0
    seek_count: int = 0
    refuse_count: int = 0
    peak_z_score: float = 0.0
    mean_entropy: float = 0.0
    mean_confidence: float = 0.0
    entropy_trend_summary: str = "stable"   # Overall trend
    
    # Surprise statistics (populated by MetacognitiveCore)
    mean_surprise: float = 0.0
    peak_surprise: float = 0.0
    high_surprise_count: int = 0        # Tokens where surprise > 2*mean

    # Text content for R_thinking_quality (set by inference pipeline)
    thinking_text: str = ""              # Content inside <thinking> blocks
    answer_text: str = ""                # Final answer text (post-thinking)

    def add_event(self, signal: 'CognitiveSignal', step: int) -> None:
        """Create event from CognitiveSignal and append"""
        event = CognitiveEvent(
            step=step,
            token_entropy=signal.token_entropy,
            semantic_entropy=signal.semantic_entropy,
            confidence=signal.confidence,
            z_score=signal.z_score,
            token_surprise=signal.token_surprise,
            entropy_gradient=signal.entropy_gradient,
            entropy_momentum=signal.entropy_momentum,
            cognitive_phase=signal.cognitive_phase,
            decision=signal.decision,
            epistemic_state=signal.epistemic_state,
            boundary_action=signal.boundary_action,
            entropy_trend=signal.entropy_trend,
            cusum_alarm=signal.cusum_alarm,
        )
        self.events.append(event)
        self.total_tokens = len(self.events)

    def to_dict(self) -> Dict[str, Any]:
        """Export trace as a serializable dictionary."""
        def _event_dict(e: CognitiveEvent) -> Dict[str, Any]:
            return {
                "step": e.step,
                "token_entropy": round(e.token_entropy, 4),
                "semantic_entropy": round(e.semantic_entropy, 4),
                "confidence": round(e.confidence, 4),
                "z_score": round(e.z_score, 4),
                "token_surprise": round(e.token_surprise, 4),
                "entropy_gradient": round(e.entropy_gradient, 4),
                "entropy_momentum": round(e.entropy_momentum, 4),
                "cognitive_phase": e.cognitive_phase,
                "decision": e.decision.value,
                "epistemic_state": e.epistemic_state.value,
                "boundary_action": e.boundary_action.value,
                "entropy_trend": e.entropy_trend,
                "cusum_alarm": e.cusum_alarm,
            }
        return {
            "query": self.query,
            "total_tokens": self.total_tokens,
            "fast_count": self.fast_count,
            "deep_count": self.deep_count,
            "hedge_count": self.hedge_count,
            "seek_count": self.seek_count,
            "refuse_count": self.refuse_count,
            "peak_z_score": round(self.peak_z_score, 4),
            "mean_entropy": round(self.mean_entropy, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "mean_surprise": round(self.mean_surprise, 4),
            "peak_surprise": round(self.peak_surprise, 4),
            "high_surprise_count": self.high_surprise_count,
            "entropy_trend_summary": self.entropy_trend_summary,
            "has_thinking": bool(self.thinking_text),
            "events": [_event_dict(e) for e in self.events],
        }

    def to_json(self, indent: int = 2) -> str:
        """Export trace as JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass
class MetaJudgment:
    """
    Metacognitive Judgment — output of MetacognitiveCore.
    
    Contains the introspection results:
    - Overall epistemic confidence
    - Cognitive load assessment
    - Boundary status summary
    - Suggested action
    """
    # Overall assessment
    epistemic_confidence: float = 0.5       # Combined epistemic confidence [0, 1]
    cognitive_load: float = 0.0             # Cognitive load [0, 1] (DEEP ratio + trend)
    hallucination_risk: float = 0.0         # Hallucination risk [0, 1]
    
    # Key findings
    boundary_status: str = "stable"         # stable / warning / breached
    dominant_state: EpistemicState = EpistemicState.LIKELY
    stability: str = "stable"               # stable / volatile / degrading
    
    # Suggested action
    suggested_action: str = "continue"      # continue / verify / abort / hedge
    reasoning: str = ""                     # Natural language reasoning


@dataclass
class SwitchResult:
    """
    Cognitive Switch result — output of CognitiveSwitch.
    
    Guides upstream systems (e.g., Inference Engine) in compute resource allocation.
    """
    mode: str = "standard"                  # system1 / system2 / standard
    should_trigger_cot: bool = False        # Whether to inject chain-of-thought
    strategy: CoTStrategy = CoTStrategy.NONE # CoT injection strategy
    reflection_priority: float = 0.0        # Reflection priority [0, 1]
    should_use_draft_model: bool = False    # Whether draft model can be used for speedup
    compute_budget: float = 0.5             # Compute budget [0, 1]
    trend: str = "stable"                   # Entropy trend


# ═══════════════════════════════════════════════════════════════
# Generation-level Semantic Entropy (Kuhn et al., ICLR 2023)
# ═══════════════════════════════════════════════════════════════

@dataclass
class GenerationSample:
    """
    Single generation sample.
    
    Kuhn et al. method requires sampling multiple complete answers for the same prompt,
    each with its log-probability for weighted cluster probability computation.
    """
    text: str
    log_prob: float = 0.0               # Sequence log-probability (for weighting)
    tokens: List[int] = field(default_factory=list)


@dataclass
class SemanticCluster:
    """Semantic equivalence class — a group of mutually entailing generations"""
    members: List[int] = field(default_factory=list)   # indices into GenerationSample list
    probability: float = 0.0            # Cluster probability: |members|/N or sum(exp(log_prob))


@dataclass
class LatencyProfile:
    """
    Latency breakdown for System 2 semantic entropy pipeline.
    
    Enables latency vs. accuracy trade-off analysis by tracking
    time spent in each stage of the Kuhn et al. pipeline.
    """
    sampling_ms: float = 0.0            # Time for N forward-pass generations
    entailment_ms: float = 0.0          # Time for NxN entailment / similarity checks
    clustering_ms: float = 0.0          # Time for Union-Find clustering
    total_ms: float = 0.0               # End-to-end pipeline time
    n_samples_actual: int = 0           # Actual samples used (may be < N if early-exited)
    n_entailment_calls: int = 0         # NLI/embedding calls made (reduced by hybrid pre-filter)
    early_exit: bool = False            # Whether early-exit was triggered
    method: str = ""                    # "nli", "embedding", or "hybrid"


@dataclass
class SemanticEntropyResult:
    """
    Kuhn et al. (ICLR 2023) generation-level semantic entropy result.
    
    Academic definition:
        SE = -sum p(C_k) log p(C_k)
        where C_k are semantic equivalence classes obtained via bidirectional entailment clustering,
        p(C_k) = |C_k| / N (frequency estimate) or sum p(gen) for gen in C_k (probability weighted).
    """
    semantic_entropy: float = 0.0       # SE value (bits)
    n_clusters: int = 1                 # Number of semantic equivalence classes
    n_samples: int = 0                  # Total samples
    clusters: List[SemanticCluster] = field(default_factory=list)
    generations: List[GenerationSample] = field(default_factory=list)
    entailment_matrix: Optional[List[List[float]]] = None  # [N, N] entailment scores
    majority_answer: str = ""           # Representative answer from largest cluster
    majority_cluster_prob: float = 0.0  # Probability of largest cluster
    is_uncertain: bool = False          # SE > threshold -> genuinely uncertain
    latency_profile: Optional[LatencyProfile] = None  # Pipeline latency breakdown


@dataclass
class InferenceResult:
    """
    METIS-aware inference result.
    
    Contains generated text + complete cognitive metadata.
    Upstream systems use this to judge answer reliability.
    """
    text: str = ""                      # Final output text (thinking stripped)
    thinking_text: str = ""             # Extracted <thinking> block content (empty if none)
    tokens_generated: int = 0
    latency_ms: float = 0.0

    # Cognitive metadata
    final_decision: Decision = Decision.NORMAL
    final_epistemic_state: EpistemicState = EpistemicState.LIKELY
    final_boundary_action: BoundaryAction = BoundaryAction.GENERATE
    avg_token_entropy: float = 0.0
    avg_confidence: float = 0.0
    uncertainty_score: float = 0.0

    # Generation-level semantic entropy (Kuhn et al.)
    semantic_entropy_result: Optional[SemanticEntropyResult] = None

    # System 1/2 statistics
    system1_ratio: float = 0.0          # FAST token ratio
    system2_ratio: float = 0.0          # DEEP token ratio
    system2_triggered: bool = False     # Whether System 2 verification was triggered

    # Behavioral record
    boundary_interventions: int = 0     # Boundary guard intervention count
    was_hedged: bool = False            # Whether answer was hedged with uncertainty
    was_refused: bool = False           # Whether answer was refused
    was_verified: bool = False          # Whether SE verification ran
    introspection: str = ""             # Metacognitive introspection summary
