"""
METIS vLLM LogitsProcessor — Native CUSUM monitoring inside vLLM

Replaces Python per-token callback with vLLM-native logits processing.
This eliminates the CPU bottleneck in high-throughput DGX Spark scenarios.

Processors:
    1. MetisCUSUMProcessor  — Monitors entropy per-token, records CUSUM state
    2. CoTBoostProcessor    — Boosts thinking-token logits when CUSUM fires
    3. MetisCompositeProcessor — Combines both for full METIS integration

Usage with vLLM offline:
    from dgx.vllm_processor import MetisCompositeProcessor

    processor = MetisCompositeProcessor(tokenizer)
    params = SamplingParams(
        logits_processors=[processor],
        max_tokens=1024,
        temperature=0.7,
    )
    outputs = llm.generate(prompts, params)

    # Access cognitive traces after generation
    for trace in processor.get_traces():
        reward = reward_computer.compute(trace)

Usage with vLLM server (OpenAI-compatible):
    Not directly supported (server doesn't expose logits_processors).
    Use the offline LLM class instead.
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# Configuration (mirrors boundary.py defaults)
# ═══════════════════════════════════════════════════════════

@dataclass
class CUSUMConfig:
    """CUSUM monitoring parameters (synced with metis/cognitive/boundary.py)."""
    cusum_k: float = 0.5            # Allowance parameter (noise filter)
    cusum_hedge_h: float = 4.0      # HEDGE threshold
    cusum_refuse_h: float = 8.0     # REFUSE threshold
    cusum_decay: float = 0.85       # Decay on confident tokens
    surprise_baseline: float = 2.5  # Normal surprise level
    surprise_weight: float = 0.3    # Weight of surprise in CUSUM

    # Dynamic K for early tokens (absorb prompt-transition entropy)
    dynamic_k_tokens: int = 80      # First N tokens get boosted K (matches boundary.py)
    dynamic_k_boost: float = 0.5    # Additional K for early tokens

    # CoT boost parameters
    cot_boost_strength: float = 5.0  # Logit boost for thinking tokens
    cot_boost_duration: int = 50     # How many tokens to boost after trigger

    # Z-score thresholds
    z_uncertain: float = 1.0
    z_unknown: float = 2.5


# ═══════════════════════════════════════════════════════════
# Per-Request State (one per vLLM sequence)
# ═══════════════════════════════════════════════════════════

@dataclass
class CUSUMState:
    """Per-sequence CUSUM tracking state."""
    cusum: float = 0.0
    token_count: int = 0
    entropy_sum: float = 0.0
    entropy_sq_sum: float = 0.0
    hedge_count: int = 0
    refuse_count: int = 0
    cot_boost_remaining: int = 0

    # Rolling statistics (Welford's online algorithm)
    mean: float = 0.0
    m2: float = 0.0

    # Event log for trace reconstruction
    events: List[Dict[str, float]] = field(default_factory=list)

    def update_stats(self, entropy: float) -> Tuple[float, float]:
        """Update running mean/variance, return (z_score, std_dev)."""
        self.token_count += 1
        n = self.token_count

        # Welford's online algorithm
        delta = entropy - self.mean
        self.mean += delta / n
        delta2 = entropy - self.mean
        self.m2 += delta * delta2

        variance = self.m2 / max(n - 1, 1)
        std = math.sqrt(variance) if variance > 0 else 0.1

        z_score = (entropy - self.mean) / std if std > 0.01 else 0.0
        return z_score, std


# ═══════════════════════════════════════════════════════════
# Core CUSUM Processor
# ═══════════════════════════════════════════════════════════

class MetisCUSUMProcessor:
    """
    vLLM-native CUSUM monitoring processor.

    Computes token-level entropy from logits and maintains a CUSUM
    accumulator per sequence. Records events for post-hoc trace
    reconstruction.

    This processor does NOT modify logits — it only observes.
    Use MetisCompositeProcessor for observation + intervention.
    """

    def __init__(self, config: Optional[CUSUMConfig] = None):
        self._config = config or CUSUMConfig()
        # Per-request state (keyed by request/sequence index)
        self._states: Dict[int, CUSUMState] = {}
        self._request_counter = 0

    def __call__(
        self, token_ids: List[int], logits: torch.Tensor
    ) -> torch.Tensor:
        """
        vLLM logits processor interface.

        Args:
            token_ids: Previously generated token IDs for this sequence
            logits: Current step logits [vocab_size]

        Returns:
            Unmodified logits (observation only)
        """
        seq_id = id(token_ids)  # Use list identity as sequence key
        if seq_id not in self._states:
            self._states[seq_id] = CUSUMState()

        state = self._states[seq_id]
        cfg = self._config

        # Compute entropy from logits
        probs = F.softmax(logits.float(), dim=-1)
        log_probs = torch.log2(probs + 1e-10)
        entropy = -(probs * log_probs).sum().item()

        # Confidence: 1 - normalized entropy
        max_entropy = math.log2(logits.shape[-1])
        confidence = 1.0 - min(entropy / max_entropy, 1.0)

        # Update running statistics
        z_score, std = state.update_stats(entropy)

        # Dynamic K for early tokens
        current_k = cfg.cusum_k
        if state.token_count <= cfg.dynamic_k_tokens:
            current_k += cfg.dynamic_k_boost

        # CUSUM update
        if z_score > current_k:
            state.cusum += (z_score - current_k) * confidence
        elif confidence > 0.9:
            state.cusum *= cfg.cusum_decay

        # Record event
        event = {
            "step": state.token_count,
            "entropy": round(entropy, 4),
            "z_score": round(z_score, 4),
            "confidence": round(confidence, 4),
            "cusum": round(state.cusum, 4),
        }

        # Check thresholds
        if state.cusum >= cfg.cusum_refuse_h:
            state.refuse_count += 1
            state.cusum = 0.0
            event["action"] = "refuse"
        elif state.cusum >= cfg.cusum_hedge_h:
            state.hedge_count += 1
            state.cusum = 0.0
            event["action"] = "hedge"
            # Signal CoT boost
            state.cot_boost_remaining = cfg.cot_boost_duration
        else:
            event["action"] = "generate"

        state.events.append(event)
        return logits  # No modification

    def get_state(self, token_ids: List[int]) -> Optional[CUSUMState]:
        """Get CUSUM state for a specific sequence."""
        return self._states.get(id(token_ids))

    def get_all_states(self) -> Dict[int, CUSUMState]:
        """Get all sequence states."""
        return self._states

    def reset(self) -> None:
        """Clear all states (call between batches)."""
        self._states.clear()


# ═══════════════════════════════════════════════════════════
# CoT Boost Processor
# ═══════════════════════════════════════════════════════════

class CoTBoostProcessor:
    """
    Boosts logits of thinking/reasoning tokens when CUSUM fires.

    When the CUSUM processor detects a HEDGE event, this processor
    increases the probability of tokens that trigger deeper reasoning
    (e.g., "Let me think", "Wait", "Actually", thinking markers).

    This is the vLLM-native equivalent of CoT injection.
    """

    def __init__(
        self,
        tokenizer: Any,
        cusum_processor: MetisCUSUMProcessor,
        config: Optional[CUSUMConfig] = None,
    ):
        self._tokenizer = tokenizer
        self._cusum = cusum_processor
        self._config = config or CUSUMConfig()

        # Pre-compute thinking token IDs
        self._thinking_token_ids = self._build_thinking_tokens(tokenizer)
        logger.info(
            f"CoTBoost: {len(self._thinking_token_ids)} thinking tokens indexed"
        )

    def _build_thinking_tokens(self, tokenizer: Any) -> List[int]:
        """Identify token IDs that indicate reasoning/thinking."""
        thinking_phrases = [
            "Let me think",
            "Wait",
            "Actually",
            "However",
            "But",
            "On the other hand",
            "Consider",
            "First",
            "Second",
            "Therefore",
            "Because",
            "Since",
            "If we",
            "Suppose",
            "Assume",
            "Let's",
            "Step",
            "Note that",
            "Important",
            "Key",
            "Alternatively",
            "\n\n",  # Paragraph break (often precedes reasoning)
        ]

        token_ids: set[int] = set()
        for phrase in thinking_phrases:
            try:
                ids = tokenizer.encode(phrase, add_special_tokens=False)
                if ids:
                    token_ids.add(ids[0])  # First token of phrase
            except Exception:
                continue

        return list(token_ids)

    def __call__(
        self, token_ids: List[int], logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Boost thinking token logits if CUSUM triggered recently.

        Args:
            token_ids: Previously generated token IDs
            logits: Current step logits [vocab_size]

        Returns:
            Modified logits with boosted thinking tokens
        """
        state = self._cusum.get_state(token_ids)
        if state is None or state.cot_boost_remaining <= 0:
            return logits

        # Apply boost to thinking tokens
        boost = self._config.cot_boost_strength
        for tid in self._thinking_token_ids:
            if tid < logits.shape[-1]:
                logits[tid] += boost

        state.cot_boost_remaining -= 1
        return logits


# ═══════════════════════════════════════════════════════════
# Composite Processor (CUSUM + CoT Boost)
# ═══════════════════════════════════════════════════════════

class MetisCompositeProcessor:
    """
    Combined METIS processor: CUSUM monitoring + CoT boost.

    Single callable that wraps both observation and intervention.

    Usage:
        processor = MetisCompositeProcessor(tokenizer)
        params = SamplingParams(logits_processors=[processor], ...)
        outputs = llm.generate(prompts, params)
    """

    def __init__(
        self,
        tokenizer: Any,
        config: Optional[CUSUMConfig] = None,
    ):
        cfg = config or CUSUMConfig()
        self._cusum = MetisCUSUMProcessor(cfg)
        self._cot_boost = CoTBoostProcessor(tokenizer, self._cusum, cfg)

    def __call__(
        self, token_ids: List[int], logits: torch.Tensor
    ) -> torch.Tensor:
        """Process logits: observe (CUSUM) then intervene (CoT boost)."""
        logits = self._cusum(token_ids, logits)
        logits = self._cot_boost(token_ids, logits)
        return logits

    def get_cusum_processor(self) -> MetisCUSUMProcessor:
        """Access the underlying CUSUM processor for state inspection."""
        return self._cusum

    def get_traces_summary(self) -> List[Dict[str, Any]]:
        """
        Get summary of cognitive monitoring for all sequences.

        Returns:
            List of dicts with hedge_count, refuse_count, event stats
        """
        summaries: List[Dict[str, Any]] = []
        for seq_id, state in self._cusum.get_all_states().items():
            n = state.token_count
            if n == 0:
                continue

            entropies = [e["entropy"] for e in state.events]
            mean_h = sum(entropies) / n if n > 0 else 0.0

            summaries.append({
                "seq_id": seq_id,
                "total_tokens": n,
                "hedge_count": state.hedge_count,
                "refuse_count": state.refuse_count,
                "mean_entropy": round(mean_h, 4),
                "peak_cusum": round(
                    max((e["cusum"] for e in state.events), default=0.0), 4
                ),
                "events": state.events,
            })
        return summaries

    def reset(self) -> None:
        """Reset all states between batches."""
        self._cusum.reset()
