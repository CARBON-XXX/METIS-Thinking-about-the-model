"""
METIS Cognitive GRPO (Group Relative Policy Optimization)

Implements cognitive-reward-driven GRPO following DeepSeek-R1 methodology,
but replacing the LLM-as-judge reward with METIS information-theoretic signals.

Pipeline:
═══════════════════════════════════════════════════════════════════
1. For each prompt, generate N responses with METIS instrumentation
2. Compute CognitiveReward for each response's trace
3. Rank responses by cognitive reward
4. Compute GRPO advantages: A_i = (R_i - mean(R)) / std(R)
5. Export ranked groups for policy optimization

Key difference from standard GRPO:
- Reward is NOT from another LLM or human preference
- Reward is from objective information-theoretic measurements
- This means reward is:
  (a) deterministic (same trace → same reward)
  (b) decomposable (can debug which component caused low reward)
  (c) cheap (no extra inference)
═══════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..core.types import CognitiveTrace
from .rewards import CognitiveRewardComputer, RewardBreakdown, RewardConfig

logger = logging.getLogger(__name__)


@dataclass
class GRPOSample:
    """Single sample in a GRPO group."""
    prompt: str
    response: str
    trace: CognitiveTrace
    reward: RewardBreakdown
    advantage: float = 0.0          # Normalized advantage within group
    rank: int = 0                   # Rank within group (0 = best)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "reward_total": round(self.reward.total, 4),
            "advantage": round(self.advantage, 4),
            "rank": self.rank,
            "reward_breakdown": self.reward.to_dict(),
            "trace_summary": {
                "total_tokens": self.trace.total_tokens,
                "mean_entropy": round(self.trace.mean_entropy, 4),
                "mean_surprise": round(self.trace.mean_surprise, 4),
                "fast_count": self.trace.fast_count,
                "deep_count": self.trace.deep_count,
            },
        }


@dataclass
class GRPOGroup:
    """A group of N samples for one prompt, ranked by cognitive reward."""
    prompt: str
    samples: List[GRPOSample] = field(default_factory=list)

    @property
    def best(self) -> Optional[GRPOSample]:
        return self.samples[0] if self.samples else None

    @property
    def worst(self) -> Optional[GRPOSample]:
        return self.samples[-1] if self.samples else None

    @property
    def reward_spread(self) -> float:
        """Reward difference between best and worst (signal strength)."""
        if len(self.samples) < 2:
            return 0.0
        return self.samples[0].reward.total - self.samples[-1].reward.total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "n_samples": len(self.samples),
            "reward_spread": round(self.reward_spread, 4),
            "samples": [s.to_dict() for s in self.samples],
        }


@dataclass
class GRPOConfig:
    """GRPO generation and ranking configuration."""
    n_samples: int = 4                  # Number of samples per prompt
    temperature_spread: float = 0.1     # Temperature variation across samples
    base_temperature: float = 0.7       # Base sampling temperature
    min_reward_spread: float = 0.05     # Minimum spread to consider group useful
    advantage_clip: float = 2.0         # Clip advantage to [-clip, clip]


class CognitiveGRPO:
    """
    Cognitive GRPO: Generate, Rank, and Prepare training data.

    Usage:
        from metis.training import CognitiveGRPO

        grpo = CognitiveGRPO(metis_inference_engine)
        group = grpo.generate_group("What is quantum entanglement?")
        print(group.best.response)
        print(group.reward_spread)

        # Export for training
        grpo.export_groups([group], "grpo_data.jsonl")
    """

    def __init__(
        self,
        inference_fn: Optional[Callable] = None,
        reward_config: Optional[RewardConfig] = None,
        grpo_config: Optional[GRPOConfig] = None,
    ):
        """
        Args:
            inference_fn: Callable(prompt, temperature) -> (response_text, CognitiveTrace)
                If None, use rank_traces() for offline processing.
            reward_config: Reward computation configuration
            grpo_config: GRPO generation configuration
        """
        self._inference_fn = inference_fn
        self._reward_computer = CognitiveRewardComputer(reward_config)
        self._config = grpo_config or GRPOConfig()

    def generate_group(self, prompt: str) -> GRPOGroup:
        """
        Generate N responses for a prompt and rank by cognitive reward.

        Requires inference_fn to be set.

        Args:
            prompt: Input prompt

        Returns:
            GRPOGroup with ranked samples
        """
        if self._inference_fn is None:
            raise RuntimeError(
                "inference_fn not set. Use rank_traces() for offline processing."
            )

        cfg = self._config
        samples: List[GRPOSample] = []

        for i in range(cfg.n_samples):
            # Vary temperature slightly across samples for diversity
            temp = cfg.base_temperature + (i - cfg.n_samples / 2) * cfg.temperature_spread
            temp = max(0.1, min(2.0, temp))

            response_text, trace = self._inference_fn(prompt, temp)
            reward = self._reward_computer.compute(trace)

            samples.append(GRPOSample(
                prompt=prompt,
                response=response_text,
                trace=trace,
                reward=reward,
            ))

        return self._rank_and_normalize(prompt, samples)

    def rank_traces(
        self,
        prompt: str,
        responses: List[str],
        traces: List[CognitiveTrace],
    ) -> GRPOGroup:
        """
        Offline ranking: given pre-computed traces, rank by cognitive reward.

        Args:
            prompt: Original prompt
            responses: List of response texts
            traces: List of CognitiveTrace objects (one per response)

        Returns:
            GRPOGroup with ranked samples
        """
        if len(responses) != len(traces):
            raise ValueError(
                f"responses ({len(responses)}) and traces ({len(traces)}) must have same length"
            )

        samples: List[GRPOSample] = []
        for resp, trace in zip(responses, traces):
            reward = self._reward_computer.compute(trace)
            samples.append(GRPOSample(
                prompt=prompt,
                response=resp,
                trace=trace,
                reward=reward,
            ))

        return self._rank_and_normalize(prompt, samples)

    def _rank_and_normalize(
        self, prompt: str, samples: List[GRPOSample]
    ) -> GRPOGroup:
        """Sort by reward, compute normalized advantages."""
        # Sort descending by total reward
        samples.sort(key=lambda s: s.reward.total, reverse=True)

        # Assign ranks
        for i, s in enumerate(samples):
            s.rank = i

        # Compute GRPO advantages: A_i = (R_i - mean(R)) / std(R)
        rewards = [s.reward.total for s in samples]
        mean_r = sum(rewards) / len(rewards)
        var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
        std_r = math.sqrt(var_r) if var_r > 1e-8 else 1.0

        clip = self._config.advantage_clip
        for s in samples:
            raw_adv = (s.reward.total - mean_r) / std_r
            s.advantage = max(-clip, min(clip, raw_adv))

        group = GRPOGroup(prompt=prompt, samples=samples)

        logger.info(
            f"[GRPO] Ranked {len(samples)} samples | "
            f"spread={group.reward_spread:.4f} | "
            f"best={samples[0].reward.total:.4f} | "
            f"worst={samples[-1].reward.total:.4f}"
        )

        return group

    @staticmethod
    def export_groups(groups: List[GRPOGroup], path: str) -> None:
        """
        Export GRPO groups to JSONL file for training.

        Each line is a complete group with ranked samples and advantages.

        Args:
            groups: List of GRPOGroup objects
            path: Output file path (.jsonl)
        """
        with open(path, "w", encoding="utf-8") as f:
            for group in groups:
                f.write(json.dumps(group.to_dict(), ensure_ascii=False) + "\n")

        logger.info(f"[GRPO] Exported {len(groups)} groups to {path}")
