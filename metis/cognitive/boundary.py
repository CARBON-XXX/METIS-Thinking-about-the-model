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

from ..core.types import EpistemicState, BoundaryAction, CognitiveSignal

# ── Rust native acceleration (optional) ──
try:
    from metis_native import BoundaryGuardNative as _NativeBoundary
    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False

# Enum lookup tables for Rust integer codes
_STATE_FROM_INT = [EpistemicState.KNOWN, EpistemicState.LIKELY, EpistemicState.UNCERTAIN, EpistemicState.UNKNOWN]
_ACTION_FROM_INT = [BoundaryAction.GENERATE, BoundaryAction.HEDGE, BoundaryAction.SEEK, BoundaryAction.REFUSE]

# ── Constants ──
Z_UNCERTAIN_DEFAULT = 1.0       # z > 1.0 → UNCERTAIN (1 stddev above mean)
Z_UNKNOWN_DEFAULT = 1.2         # z > 1.2 → UNKNOWN (lowered for short sequences)
Z_KNOWN_DEFAULT = -0.5          # z < -0.5 → KNOWN
MIN_WARMUP_TOKENS = 4           # Cold-start cognitive events (was 20 — too large for short gen)

CONFIDENCE_REFUSE = 0.3         # c < 0.3 → REFUSE (extreme uncertainty)
CONFIDENCE_SEEK = 0.7           # c < 0.7 → SEEK (high uncertainty)
CONFIDENCE_KNOWN = 0.7          # c > 0.7 → KNOWN (when low z)

# ── CUSUM (Cumulative Sum) control chart parameters ──
# Principled change-point detection replacing hardcoded streak/gate heuristics.
#
# Formula: S(t) = max(0, S(t-1) + (z - k) * sd)
#   - k (allowance): absorbs normal entropy fluctuations
#   - sd weighting: high z + high sd = genuine uncertainty → fast accumulation
#                   high z + low sd  = synonyms/paraphrases → slow accumulation
#   - Captures BOTH duration and magnitude in a single statistic
#
# On confident tokens (z < 0): S(t) = S(t-1) * decay
#   - Gradual forgetting, not hard reset → tolerates brief confident interjections
#
# After triggering (HEDGE/REFUSE): S(t) = 0 (reset, ready for next detection)
CUSUM_K = 0.5               # Allowance: z < 0.5 absorbed as lexical variation
CUSUM_HEDGE_H = 4.0         # HEDGE threshold (~8 tokens of genuine uncertainty)
CUSUM_REFUSE_H = 8.0        # REFUSE threshold (sustained high uncertainty)
CUSUM_DECAY = 0.85          # Decay factor on confident tokens (faster forgetting)

# Surprise-based CUSUM boost (prediction error feedback)
# When the model generates tokens it doesn't believe in (high surprise),
# this accelerates the CUSUM independently of z-score.
SURPRISE_BASELINE = 2.5     # bits; moderate surprise is normal, only flag high
SURPRISE_WEIGHT = 0.25      # CUSUM contribution per excess surprise bit


class EpistemicBoundaryGuard:
    """
    Epistemic Boundary Guard — CUSUM-based sustained uncertainty detection.
    
    Uses a sd-weighted CUSUM (Cumulative Sum) control chart to detect
    sustained epistemic uncertainty. This is superior to streak-counting:
    - Captures both **duration** and **magnitude** of uncertainty
    - sd-weighting filters synonym/lexical noise automatically
    - Single statistic replaces 5+ hardcoded thresholds
    - Principled: CUSUM is the optimal sequential detection test
      for shift-in-mean under Gaussian assumptions (Page, 1954)
    
    Epistemic state (KNOWN/LIKELY/UNCERTAIN/UNKNOWN) is classified by
    the current z-score for diagnostic reporting. Boundary actions
    (GENERATE/HEDGE/SEEK/REFUSE) are driven by the CUSUM level.
    """

    def __init__(
        self,
        uncertain_z: float = Z_UNCERTAIN_DEFAULT,
        unknown_z: float = Z_UNKNOWN_DEFAULT,
        known_z: float = Z_KNOWN_DEFAULT,
        min_warmup_tokens: int = MIN_WARMUP_TOKENS,
        on_action: Optional[Callable[[BoundaryAction, str], None]] = None,
    ):
        # z-score thresholds (for epistemic state classification only)
        self._uncertain_z = uncertain_z
        self._unknown_z = unknown_z
        self._known_z = known_z
        
        # Cold-start protection
        self._min_warmup_tokens = min_warmup_tokens
        self._token_count = 0
        
        # CUSUM state: single statistic for sustained uncertainty
        self._cusum = 0.0
        
        # Surprise feedback (1-step lag from inference sampling)
        self._last_surprise = 0.0
        
        # Action callback
        self._on_action = on_action
        
        # Action statistics
        self._action_counts: Dict[BoundaryAction, int] = {
            a: 0 for a in BoundaryAction
        }
        
        # Rust native accelerator (if available)
        self._native = None
        if _HAS_NATIVE:
            self._native = _NativeBoundary(
                uncertain_z=uncertain_z,
                unknown_z=unknown_z,
                known_z=known_z,
                min_warmup=min_warmup_tokens,
                cusum_k=CUSUM_K,
                hedge_h=CUSUM_HEDGE_H,
                refuse_h=CUSUM_REFUSE_H,
                decay=CUSUM_DECAY,
                surprise_base=SURPRISE_BASELINE,
                surprise_w=SURPRISE_WEIGHT,
                conf_refuse=CONFIDENCE_REFUSE,
                conf_seek=CONFIDENCE_SEEK,
                conf_known=CONFIDENCE_KNOWN,
            )
    
    def evaluate(
        self, 
        signal: CognitiveSignal,
        thresholds: Optional[Tuple[float, float]] = None
    ) -> Tuple[EpistemicState, BoundaryAction, str]:
        """
        Evaluate epistemic boundary using CUSUM control chart.
        """
        z = signal.z_score
        c = signal.confidence
        sd = signal.semantic_diversity
        self._token_count += 1
        
        # ── Dynamic Allowance (K) ──
        # Higher tolerance in the early generation phase to allow for intent exploration
        # (e.g., when the prompt is short/ambiguous like "A) Yes")
        current_k = CUSUM_K
        if self._token_count <= 20:
            current_k = CUSUM_K + 0.5  # K=1.0 for first 20 tokens
        
        # ── Rust fast path ──
        if self._native is not None:
            z_unc_arg = thresholds[0] if thresholds else None
            z_unk_arg = thresholds[1] if thresholds else None
            state_i, action_i, explanation = self._native.evaluate(
                z, c, sd, z_unc_arg, z_unk_arg
            )
            state = _STATE_FROM_INT[state_i]
            action = _ACTION_FROM_INT[action_i]
            return self._emit(state, action, explanation)
        
        # ── Python fallback ──
        # Thresholds for epistemic state classification
        z_unc = thresholds[0] if thresholds else self._uncertain_z
        z_unk = thresholds[1] if thresholds else self._unknown_z
        z_kno = self._known_z
        
        # Cold-start: z-score unreliable before sufficient samples
        if self._token_count <= self._min_warmup_tokens:
            return self._emit(EpistemicState.LIKELY, BoundaryAction.GENERATE, "")
        
        # ── CUSUM Update ──
        # Two independent contributions:
        # 1. z-score * sd (distribution-level uncertainty)
        # 2. surprise boost (token-level prediction error)
        if z > current_k:
            self._cusum += (z - current_k) * sd
        elif z < 0:
            # Confident token → decay accumulated uncertainty
            self._cusum *= CUSUM_DECAY
        # z in [0, CUSUM_K): within normal variation, no z-score change
        
        # Surprise boost: if last sampled token had high prediction error,
        # accelerate CUSUM regardless of z-score.
        # This catches hallucination where model confidently outputs wrong tokens.
        if self._last_surprise > SURPRISE_BASELINE:
            self._cusum += (self._last_surprise - SURPRISE_BASELINE) * SURPRISE_WEIGHT
        
        # ── Epistemic State (diagnostic, based on current z) ──
        if z > z_unk:
            state = EpistemicState.UNKNOWN
        elif z > z_unc:
            state = EpistemicState.UNCERTAIN
        elif z < z_kno and c > CONFIDENCE_KNOWN:
            state = EpistemicState.KNOWN
        else:
            state = EpistemicState.LIKELY
        
        # ── Boundary Action (adaptive, based on CUSUM level) ──
        # SEEK: extreme sustained uncertainty OR early high semantic diversity (intent clarification)
        # If in early exploration with high diversity, prefer SEEK over HEDGE
        # Broader window + relaxed diversity threshold to catch multi-meaning
        # exploration (e.g., "agent" = software/AI/customer-service/physical)
        is_intent_exploration = self._token_count <= 80 and sd >= 0.7
        
        # REFUSE: extreme sustained uncertainty + low confidence
        if self._cusum >= CUSUM_REFUSE_H and c < CONFIDENCE_REFUSE:
            cusum_val = self._cusum
            self._cusum = 0.0  # Reset after triggering
            return self._emit(
                EpistemicState.UNKNOWN,
                BoundaryAction.REFUSE,
                f"Sustained extreme uncertainty (cusum={cusum_val:.1f})",
            )
        
        # SEEK: high sustained uncertainty + moderate confidence, OR intent clarification
        if self._cusum >= CUSUM_REFUSE_H and c < CONFIDENCE_SEEK:
            cusum_val = self._cusum
            self._cusum = 0.0
            return self._emit(
                EpistemicState.UNKNOWN,
                BoundaryAction.SEEK,
                f"External verification needed (cusum={cusum_val:.1f})",
            )
            
        # HEDGE: moderate sustained uncertainty
        if self._cusum >= CUSUM_HEDGE_H:
            cusum_val = self._cusum
            self._cusum = 0.0  # Reset after triggering
            
            # Upgrade HEDGE to SEEK if it looks like intent exploration
            if is_intent_exploration:
                return self._emit(
                    EpistemicState.UNCERTAIN,
                    BoundaryAction.SEEK,
                    f"Intent clarification needed (cusum={cusum_val:.1f})",
                )
                
            return self._emit(
                EpistemicState.UNCERTAIN,
                BoundaryAction.HEDGE,
                f"Accumulated uncertainty (cusum={cusum_val:.1f})",
            )
        
        return self._emit(state, BoundaryAction.GENERATE, "")
    
    def feed_surprise(self, surprise: float) -> None:
        """Feed back the sampled token's surprise for next step's CUSUM.
        
        Called from inference.py after each token sampling.
        1-step lag: this surprise affects the NEXT evaluate() call.
        """
        self._last_surprise = surprise
        if self._native is not None:
            self._native.feed_surprise(surprise)

    def get_uncertainty_score(self) -> float:
        """Get CUSUM-based uncertainty score (higher = more uncertain)"""
        if self._native is not None:
            return self._native.uncertainty_score
        return self._cusum
    
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
        self._cusum = 0.0
        self._last_surprise = 0.0
        self._token_count = 0
        self._action_counts = {a: 0 for a in BoundaryAction}
        if self._native is not None:
            self._native.reset()
