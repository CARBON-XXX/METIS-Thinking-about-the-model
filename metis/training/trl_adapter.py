"""
METIS TRL Adapter — Direct integration with HuggingFace TRL

Bridges METIS cognitive rewards with TRL's DPOTrainer, KTOTrainer,
and custom reward functions.

Provides:
1. MetisDPODataCollator  — Wraps DPO pairs into TRL-expected format
2. MetisRewardFunction   — Callable reward function for online GRPO
3. prepare_dpo_dataset   — One-shot: traces → HuggingFace Dataset
4. prepare_kto_dataset   — One-shot: traces → HuggingFace Dataset

Compatibility:
- trl >= 0.7.0 (DPOTrainer v2 format)
- datasets >= 2.0
- transformers >= 4.36

Usage:
    from metis.training.trl_adapter import prepare_dpo_dataset
    dataset = prepare_dpo_dataset(prompts, responses_list, traces_list)
    # Feed directly into DPOTrainer
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from ..core.types import CognitiveTrace
from .rewards import CognitiveRewardComputer, RewardBreakdown, RewardConfig
from .grpo import CognitiveGRPO, GRPOGroup
from .dataset import PreferencePairGenerator, GeneratorConfig

logger = logging.getLogger(__name__)


def prepare_dpo_dataset(
    prompts: List[str],
    responses_per_prompt: List[List[str]],
    traces_per_prompt: List[List[CognitiveTrace]],
    reward_config: Optional[RewardConfig] = None,
    generator_config: Optional[GeneratorConfig] = None,
    min_reward_margin: float = 0.05,
) -> List[Dict[str, str]]:
    """
    One-shot conversion: (prompts, responses, traces) → DPO dataset rows.

    Each row is a dict with keys: "prompt", "chosen", "rejected"
    compatible with TRL DPOTrainer and datasets.Dataset.from_list().

    Args:
        prompts: List of N prompts
        responses_per_prompt: List of N lists, each containing M response texts
        traces_per_prompt: List of N lists, each containing M CognitiveTrace objects
        reward_config: Reward computation configuration
        generator_config: Pair generation configuration
        min_reward_margin: Minimum reward margin to include a pair

    Returns:
        List of dicts with "prompt", "chosen", "rejected" keys

    Example:
        rows = prepare_dpo_dataset(prompts, responses, traces)
        from datasets import Dataset
        dataset = Dataset.from_list(rows)
        # Use with DPOTrainer
    """
    if len(prompts) != len(responses_per_prompt) != len(traces_per_prompt):
        raise ValueError("prompts, responses_per_prompt, and traces_per_prompt must have same length")

    grpo = CognitiveGRPO(reward_config=reward_config)
    gen_cfg = generator_config or GeneratorConfig(min_reward_margin=min_reward_margin)
    gen = PreferencePairGenerator(gen_cfg)

    groups: List[GRPOGroup] = []
    for prompt, responses, traces in zip(prompts, responses_per_prompt, traces_per_prompt):
        group = grpo.rank_traces(prompt, responses, traces)
        groups.append(group)

    pairs = gen.from_groups(groups)

    # Convert to TRL DPO format
    rows: List[Dict[str, str]] = []
    for p in pairs:
        rows.append({
            "prompt": p.prompt,
            "chosen": p.chosen,
            "rejected": p.rejected,
        })

    logger.info(f"[TRL] Prepared {len(rows)} DPO rows from {len(prompts)} prompts")
    return rows


def prepare_kto_dataset(
    prompts: List[str],
    responses_per_prompt: List[List[str]],
    traces_per_prompt: List[List[CognitiveTrace]],
    reward_config: Optional[RewardConfig] = None,
    desirable_threshold: float = 0.3,
    undesirable_threshold: float = -0.1,
) -> List[Dict[str, Any]]:
    """
    One-shot conversion: (prompts, responses, traces) → KTO dataset rows.

    Each row is a dict with keys: "prompt", "completion", "label"
    compatible with TRL KTOTrainer.

    Args:
        prompts: List of N prompts
        responses_per_prompt: List of N lists of response texts
        traces_per_prompt: List of N lists of CognitiveTrace objects
        reward_config: Reward computation configuration
        desirable_threshold: Reward above this → label=True
        undesirable_threshold: Reward below this → label=False

    Returns:
        List of dicts with "prompt", "completion", "label" keys
    """
    grpo = CognitiveGRPO(reward_config=reward_config)
    gen_cfg = GeneratorConfig(
        kto_desirable_threshold=desirable_threshold,
        kto_undesirable_threshold=undesirable_threshold,
    )
    gen = PreferencePairGenerator(gen_cfg)

    groups: List[GRPOGroup] = []
    for prompt, responses, traces in zip(prompts, responses_per_prompt, traces_per_prompt):
        group = grpo.rank_traces(prompt, responses, traces)
        groups.append(group)

    kto_samples = gen.to_kto(groups)

    rows: List[Dict[str, Any]] = []
    for s in kto_samples:
        rows.append({
            "prompt": s.prompt,
            "completion": s.completion,
            "label": s.label,
        })

    logger.info(f"[TRL] Prepared {len(rows)} KTO rows from {len(prompts)} prompts")
    return rows


class MetisRewardFunction:
    """
    Callable reward function for online RL training.

    Wraps CognitiveRewardComputer into the interface expected by
    TRL's PPOTrainer reward_model or custom GRPO implementations.

    Usage:
        reward_fn = MetisRewardFunction(metis_engine)
        # In training loop:
        reward = reward_fn(prompt, response)  # Returns float
    """

    def __init__(
        self,
        metis_inference: Any = None,
        reward_config: Optional[RewardConfig] = None,
    ):
        """
        Args:
            metis_inference: MetisInference instance (for online reward computation)
            reward_config: Reward computation configuration
        """
        self._inference = metis_inference
        self._computer = CognitiveRewardComputer(reward_config)

    def __call__(self, prompt: str, response: str) -> float:
        """
        Compute cognitive reward for a (prompt, response) pair.

        If metis_inference is provided, runs inference to get trace.
        Otherwise raises RuntimeError.

        Args:
            prompt: Input prompt
            response: Generated response

        Returns:
            Total cognitive reward (float)
        """
        if self._inference is None:
            raise RuntimeError(
                "metis_inference not provided. "
                "Use from_trace() for offline reward computation."
            )

        # Run METIS-instrumented inference to get cognitive trace
        result = self._inference.generate_cognitive(prompt)
        trace = self._inference._metis.trace
        reward = self._computer.compute(trace)
        return reward.total

    def from_trace(self, trace: CognitiveTrace) -> RewardBreakdown:
        """
        Compute reward from a pre-existing trace (offline mode).

        Args:
            trace: CognitiveTrace from previous inference

        Returns:
            Full RewardBreakdown
        """
        return self._computer.compute(trace)

    def batch_from_traces(
        self, traces: List[CognitiveTrace]
    ) -> List[RewardBreakdown]:
        """
        Batch compute rewards from traces.

        Args:
            traces: List of CognitiveTrace objects

        Returns:
            List of RewardBreakdown objects
        """
        return [self._computer.compute(t) for t in traces]
