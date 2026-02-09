"""
METIS Cognitive Switch
Kahneman dual-process theory implementation

System 1 (fast thinking): low entropy → intuition, automatic, low cost
System 2 (slow thinking): high entropy → logic, reasoning, high cost

Without this mechanism, AGI is like someone who chatters endlessly but never thinks deeply.
With it, AGI knows when to speak automatically and when to think carefully.
"""
from typing import Optional, Dict
import collections

from ..core.types import Decision, CognitiveSignal, SwitchResult, CoTStrategy

# -- Oscillation detection parameters --
OSCILLATION_WINDOW = 8          # Detection window size
OSCILLATION_SWITCH_THRESHOLD = 4  # Switch count threshold within window


class CognitiveSwitch:
    """
    Kahneman dual-system cognitive switch.
    
    Consumes METIS core Decision signals, maintains cognitive mode state,
    and provides compute resource allocation recommendations to upstream systems.
    """
    
    def __init__(self):
        self._mode_history: collections.deque = collections.deque(maxlen=100)
        self._entropy_history: collections.deque = collections.deque(maxlen=50)
        self._consecutive_deep: int = 0
        self._total_fast: int = 0
        self._total_deep: int = 0
        self._total_normal: int = 0
        self._is_oscillating: bool = False
        self._reflection_priority: float = 0.0
    
    def process(self, signal: CognitiveSignal) -> SwitchResult:
        """
        Process cognitive signal, return compute mode recommendation.
        
        Enhanced logic:
        - Oscillation detection: decisions alternating between FAST/DEEP -> force reflection
        - Reflection priority: combine z-score, trend, oscillation into [0, 1] priority
        - CoT strategy recommendation: recommend strategy type based on signal characteristics
        
        Returns:
            SwitchResult: contains mode, compute_budget, strategy, reflection_priority, etc.
        """
        decision = signal.decision
        self._entropy_history.append(signal.semantic_entropy)
        self._mode_history.append(decision)

        # -- Oscillation detection --
        self._is_oscillating = self._detect_oscillation()
        self._reflection_priority = self._compute_reflection_priority(signal)

        # -- Oscillation forced upgrade: even if Decision is NORMAL, oscillation means deep thinking needed --
        if self._is_oscillating and decision != Decision.DEEP:
            self._consecutive_deep = 0
            self._total_normal += 1
            return SwitchResult(
                mode="system2",
                should_trigger_cot=True,
                strategy=CoTStrategy.REFLECTION,
                reflection_priority=self._reflection_priority,
                should_use_draft_model=False,
                compute_budget=0.8,  # Oscillation upgrade but not full budget, control latency
                trend=signal.entropy_trend,
            )

        if decision == Decision.FAST:
            self._consecutive_deep = 0
            self._total_fast += 1
            return SwitchResult(
                mode="system1",
                should_trigger_cot=False,
                strategy=CoTStrategy.NONE,
                reflection_priority=0.0,
                should_use_draft_model=True,
                compute_budget=0.2,
                trend=signal.entropy_trend,
            )
        
        if decision == Decision.DEEP:
            self._consecutive_deep += 1
            self._total_deep += 1
            strategy = self._recommend_strategy(signal)
            return SwitchResult(
                mode="system2",
                should_trigger_cot=True,
                strategy=strategy,
                reflection_priority=self._reflection_priority,
                should_use_draft_model=False,
                compute_budget=1.0,
                trend=signal.entropy_trend,
            )
        
        # NORMAL
        self._consecutive_deep = 0
        self._total_normal += 1
        
        # If entropy trend rising, preemptively increase budget
        budget = 0.5
        if signal.entropy_trend == "rising":
            budget = 0.7
        
        return SwitchResult(
            mode="standard",
            should_trigger_cot=False,
            strategy=CoTStrategy.NONE,
            reflection_priority=self._reflection_priority,
            should_use_draft_model=False,
            compute_budget=budget,
            trend=signal.entropy_trend,
        )
    
    @property
    def is_stuck(self) -> bool:
        """Whether stuck in sustained deep reasoning (may need external intervention)"""
        return self._consecutive_deep > 10
    
    @property
    def stats(self) -> Dict:
        total = self._total_fast + self._total_normal + self._total_deep
        if total == 0:
            return {"system1_ratio": 0, "system2_ratio": 0, "total": 0}
        return {
            "system1_ratio": self._total_fast / total,
            "system2_ratio": self._total_deep / total,
            "total": total,
            "consecutive_deep": self._consecutive_deep,
        }
    
    def reset(self) -> None:
        self._mode_history.clear()
        self._entropy_history.clear()
        self._consecutive_deep = 0
        self._total_fast = 0
        self._total_deep = 0
        self._total_normal = 0
        self._is_oscillating = False
        self._reflection_priority = 0.0

    def _detect_oscillation(self) -> bool:
        """
        Detect decision oscillation: frequent Decision switches in recent N steps.
        
        FAST -> DEEP -> FAST -> DEEP indicates model oscillating at knowledge boundary.
        This typically means the question touches the model's knowledge edge, requiring REFLECTION.
        """
        if len(self._mode_history) < OSCILLATION_WINDOW:
            return False
        
        recent = list(self._mode_history)[-OSCILLATION_WINDOW:]
        switches = sum(
            1 for i in range(1, len(recent)) if recent[i] != recent[i - 1]
        )
        return switches >= OSCILLATION_SWITCH_THRESHOLD

    def _compute_reflection_priority(self, signal: CognitiveSignal) -> float:
        """
        Compute reflection priority [0, 1].
        
        Combined factors:
        - High z-score -> entropy anomaly, needs reflection
        - Oscillation -> unstable knowledge boundary
        - Rising trend -> uncertainty is growing
        """
        priority = 0.0
        
        # z-score contribution: accumulates from z > 1.0, max at z=4.0
        if signal.z_score > 1.0:
            priority += min((signal.z_score - 1.0) / 3.0, 1.0) * 0.4
        
        # Oscillation contribution
        if self._is_oscillating:
            priority += 0.3
        
        # Trend contribution
        if signal.entropy_trend == "rising":
            priority += 0.15
        elif signal.entropy_trend == "oscillating":
            priority += 0.25
        
        # Low confidence contribution
        if signal.confidence < 0.3:
            priority += 0.15
        
        return min(priority, 1.0)

    def _recommend_strategy(self, signal: CognitiveSignal) -> CoTStrategy:
        """
        Recommend CoT strategy for DEEP decision.
        
        Priority: REFLECTION > DECOMPOSITION > CLARIFICATION > STANDARD
        """
        if self._is_oscillating:
            return CoTStrategy.REFLECTION
        
        if self._consecutive_deep >= 5:
            return CoTStrategy.DECOMPOSITION
        
        if signal.semantic_diversity > 0.6 and signal.confidence < 0.3:
            return CoTStrategy.CLARIFICATION
        
        return CoTStrategy.STANDARD

    @property
    def is_oscillating(self) -> bool:
        return self._is_oscillating
