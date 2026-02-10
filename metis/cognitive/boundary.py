"""
METIS Epistemic Boundary Guard

Core AGI safety requirement: it must know what it does not know.
The biggest LLM problem is hallucination — confidently generating false information.
High probability does not mean factual correctness.

METIS measures cognitive uncertainty via semantic consistency (not lexical probability).
High semantic entropy → should not force generation, but acknowledge uncertainty or seek external tools.
This is the necessary path from "probabilistic generator" to "reliable agent".
"""
from typing import Tuple, Optional, Callable, Dict, Any
import collections

from ..core.types import EpistemicState, BoundaryAction, CognitiveSignal

# ── Constants ──
Z_UNCERTAIN_DEFAULT = 1.0       # z > 1.0 → UNCERTAIN
Z_UNKNOWN_DEFAULT = 2.0         # z > 2.0 → UNKNOWN
Z_KNOWN_DEFAULT = -0.5          # z < -0.5 → KNOWN
MIN_WARMUP_TOKENS = 20          # Cold-start token count

CONFIDENCE_REFUSE = 0.3         # c < 0.3 → REFUSE (when unknown)
CONFIDENCE_SEEK = 0.7           # c < 0.7 → SEEK (when unknown)
CONFIDENCE_KNOWN = 0.7          # c > 0.7 → KNOWN (when low z)

AVG_Z_HEDGE_THRESHOLD = 0.5     # avg z > 0.5 -> HEDGE
STREAK_HEDGE_THRESHOLD = 5      # consecutive high z > 5 -> HEDGE

# Structural uncertainty filtering
MIN_STREAK_FOR_REFUSE = 5       # REFUSE/SEEK requires >= 5 consecutive high-z tokens
                                # 3-4 token streaks are common at reasoning transitions in math/logic
UNCERTAIN_CONFIDENCE_GATE = 0.3 # UNCERTAIN zone: c > this means top token still dominant -> no HEDGE

# Semantic diversity gate: filters structurally-predictable tokens
# In high-dimensional embedding space (~3584 dims), cosine similarity concentrates near 0,
# so absolute diversity values are compressed into 0.5-1.0 range.
# Observed sd range for content tokens: 0.60-0.97 (mean ~0.85).
# Gate at 0.88 filters ~60% of content tokens (structural/synonym choices)
# and only flags tokens with genuinely dispersed semantic candidates.
SEMANTIC_DIVERSITY_GATE = 0.88


class EpistemicBoundaryGuard:
    """
    Epistemic Boundary Guard.
    
    Responsibilities:
    1. Assess current epistemic state (KNOWN / LIKELY / UNCERTAIN / UNKNOWN)
    2. Decide boundary action (GENERATE / HEDGE / SEEK / REFUSE)
    3. Track accumulated uncertainty to prevent long-range hallucination
    
    All thresholds are based on z-score (relative to observed entropy distribution),
    not absolute values. This ensures METIS auto-adapts to any model's entropy range.
    """

    def __init__(
        self,
        uncertain_z: float = Z_UNCERTAIN_DEFAULT,
        unknown_z: float = Z_UNKNOWN_DEFAULT,
        known_z: float = Z_KNOWN_DEFAULT,
        min_warmup_tokens: int = MIN_WARMUP_TOKENS,
        on_action: Optional[Callable[[BoundaryAction, str], None]] = None,
    ):
        # z-score thresholds (relative to historical distribution)
        self._uncertain_z = uncertain_z
        self._unknown_z = unknown_z
        self._known_z = known_z
        
        # Cold-start protection: z-score unreliable for first N tokens
        # (SlidingWindowStats needs sufficient samples for meaningful mean/std)
        self._min_warmup_tokens = min_warmup_tokens
        
        # Accumulated uncertainty tracking (with exponential decay)
        self._uncertainty_accumulator = 0.0
        self._uncertainty_decay = 0.995  # ~200-token effective window
        self._token_count = 0
        self._high_z_streak = 0
        
        # Action callback: invoked when non-GENERATE action is triggered
        self._on_action = on_action
        
        # Action statistics
        self._action_counts: Dict[BoundaryAction, int] = {
            a: 0 for a in BoundaryAction
        }
    
    def evaluate(
        self, 
        signal: CognitiveSignal,
        thresholds: Optional[Tuple[float, float]] = None
    ) -> Tuple[EpistemicState, BoundaryAction, str]:
        """
        Evaluate epistemic boundary.
        
        Uses signal.z_score (computed by AdaptiveController) for relative assessment.
        
        Args:
            signal: METIS cognitive signal
            thresholds: (optional) dynamically computed thresholds (z_uncertain, z_unknown).
                        If provided, overrides the default values.
            
        Returns:
            (epistemic_state, boundary_action, explanation)
        """
        z = signal.z_score
        c = signal.confidence
        self._token_count += 1
        
        # Determine which thresholds to use
        z_unc = thresholds[0] if thresholds else self._uncertain_z
        z_unk = thresholds[1] if thresholds else self._unknown_z
        z_kno = self._known_z  # KNOWN threshold relatively fixed (may be dynamic in the future)
        
        # -- Cold-start protection --
        # z-score depends on AdaptiveController's mean/std estimates.
        # First N tokens have insufficient samples, z-score unreliable (may be extremely biased).
        # During this period, only collect statistics, do not trigger non-GENERATE actions.
        if self._token_count <= self._min_warmup_tokens:
            return self._emit(EpistemicState.LIKELY, BoundaryAction.GENERATE, "")
        
        # Accumulate uncertainty (based on z-score) with exponential decay
        # Decay prevents early high-entropy tokens from permanently tainting
        # the HEDGE decision for the rest of the generation.
        self._uncertainty_accumulator *= self._uncertainty_decay
        if z > z_unc:
            self._uncertainty_accumulator += z
            self._high_z_streak += 1
        else:
            self._high_z_streak = 0
        
        # -- Clearly beyond cognitive boundary --
        # z > z_unk: entropy significantly anomalous
        # Key distinction: single-token entropy spike at paragraph/sentence boundary != cognitive uncertainty
        # Therefore REFUSE/SEEK requires consecutive high-z streak as confirmation
        #
        # CRITICAL: semantic_diversity gate — high entropy with LOW diversity means
        # top-k tokens are synonyms (lexical choice), NOT epistemic uncertainty.
        # E.g., "包括"/"例如"/"有" all mean the same thing → don't HEDGE.
        sd = signal.semantic_diversity
        if z > z_unk:
            if sd < SEMANTIC_DIVERSITY_GATE:
                # Low diversity: high entropy from synonyms/paraphrases, not uncertainty
                return self._emit(EpistemicState.LIKELY, BoundaryAction.GENERATE, "")
            if self._high_z_streak >= MIN_STREAK_FOR_REFUSE:
                # Sustained high z + high diversity -> genuine cognitive uncertainty
                if c < CONFIDENCE_REFUSE:
                    return self._emit(
                        EpistemicState.UNKNOWN,
                        BoundaryAction.REFUSE,
                        "Insufficient data, answer unreliable",
                    )
                if c < CONFIDENCE_SEEK:
                    return self._emit(
                        EpistemicState.UNKNOWN,
                        BoundaryAction.SEEK,
                        "External verification required",
                    )
            # High diversity + high z-score -> HEDGE only if sustained (streak >= 2)
            # Single z_unk spike is common at topic/sentence boundaries and should NOT trigger HEDGE
            if self._high_z_streak >= 2:
                return self._emit(
                    EpistemicState.UNCERTAIN,
                    BoundaryAction.HEDGE,
                    "Confidence-Entropy conflict, potential hallucination risk",
                )
            # Single spike: structural boundary, not cognitive uncertainty
            return self._emit(EpistemicState.LIKELY, BoundaryAction.GENERATE, "")
        
        # -- Sustained uncertainty check (priority over single-token judgment) --
        # Even if individual tokens pass confidence gate (structural uncertainty),
        # long-term accumulated uncertainty still needs to be flagged
        avg_z = self._uncertainty_accumulator / max(self._token_count, 1)
        if (avg_z > AVG_Z_HEDGE_THRESHOLD or self._high_z_streak > STREAK_HEDGE_THRESHOLD) and sd >= SEMANTIC_DIVERSITY_GATE:
            return self._emit(
                EpistemicState.UNCERTAIN,
                BoundaryAction.HEDGE,
                "Low overall certainty",
            )
        
        # -- Uncertain zone (z_unc < z < z_unk) --
        # Two gates to prevent false HEDGE:
        # 1. Semantic diversity: low diversity = synonyms → NOT uncertainty
        # 2. Confidence: high confidence = model knows → structural uncertainty
        if z > z_unc:
            if sd < SEMANTIC_DIVERSITY_GATE:
                # Low diversity: synonym choices, not real uncertainty
                return self._emit(EpistemicState.LIKELY, BoundaryAction.GENERATE, "")
            if self._high_z_streak >= MIN_STREAK_FOR_REFUSE:
                if c < UNCERTAIN_CONFIDENCE_GATE:
                    return self._emit(
                        EpistemicState.UNCERTAIN,
                        BoundaryAction.HEDGE,
                        f"I am not sure about this (z={z:.2f} > {z_unc:.2f})",
                    )
            # Structural uncertainty: confidence acceptable, or single spike -> normal generation
            return self._emit(EpistemicState.LIKELY, BoundaryAction.GENERATE, "")
        
        # -- Known zone --
        if z < z_kno and c > CONFIDENCE_KNOWN:
            return self._emit(EpistemicState.KNOWN, BoundaryAction.GENERATE, "")
        
        return self._emit(EpistemicState.LIKELY, BoundaryAction.GENERATE, "")
    
    def get_uncertainty_score(self) -> float:
        """Get accumulated uncertainty score [0, inf)"""
        if self._token_count == 0:
            return 0.0
        return self._uncertainty_accumulator / self._token_count
    
    def set_on_action(self, callback: Callable[[BoundaryAction, str], None]) -> None:
        """Register action callback — invoked when HEDGE/SEEK/REFUSE is triggered"""
        self._on_action = callback
    
    @property
    def action_counts(self) -> Dict[BoundaryAction, int]:
        """Cumulative trigger count per action"""
        return dict(self._action_counts)
    
    def _emit(
        self,
        state: EpistemicState,
        action: BoundaryAction,
        explanation: str,
    ) -> Tuple[EpistemicState, BoundaryAction, str]:
        """Emit epistemic boundary event and trigger callback"""
        self._action_counts[action] = self._action_counts.get(action, 0) + 1
        
        if action != BoundaryAction.GENERATE and self._on_action is not None:
            try:
                self._on_action(action, explanation)
            except Exception:
                pass
        
        return (state, action, explanation)
    
    def reset(self) -> None:
        """Reset (called at start of new conversation)"""
        self._uncertainty_accumulator = 0.0
        self._token_count = 0
        self._high_z_streak = 0
        self._action_counts = {a: 0 for a in BoundaryAction}
