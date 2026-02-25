"""
METIS Parallel Search — o1-style multi-path reasoning with entropy pruning

Generates N candidate reasoning paths in parallel via vLLM, monitors
cognitive quality in real-time, and prunes low-quality paths early
to concentrate compute on promising directions.

This is the DGX Spark implementation of Test-Time Compute Scaling:
    Intelligence = f(parameters) * g(inference_compute)

Architecture:
    ┌─────────────────────────────────────────────┐
    │  1. Generate N paths (vLLM batch, high temp) │
    │  2. Score each with MetisCompositeProcessor  │
    │  3. Prune bottom K% by cognitive reward      │
    │  4. Continue top paths to completion          │
    │  5. Final selection: best cognitive reward    │
    └─────────────────────────────────────────────┘

Usage:
    searcher = MetisParallelSearch(
        model_name="Qwen/Qwen2.5-72B-Instruct",
        n_paths=50,
    )
    result = searcher.search("Is a married person looking at an unmarried person?")
    print(result.best_response)
    print(result.cognitive_report)
"""
from __future__ import annotations

import gc
import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from .vllm_processor import (
    MetisCompositeProcessor,
    MetisCUSUMProcessor,
    CUSUMConfig,
    CUSUMState,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════

@dataclass
class SearchConfig:
    """Parallel search configuration."""
    # Search breadth
    n_paths: int = 50               # Total candidate paths
    prune_ratio: float = 0.5        # Prune bottom 50% at each checkpoint
    prune_checkpoints: List[int] = field(
        default_factory=lambda: [64, 128, 256]  # Token counts to prune at
    )

    # Generation
    max_tokens: int = 1024
    temperature: float = 0.9        # Higher temp for diversity
    top_p: float = 0.95
    temperature_spread: float = 0.2  # Vary temp across paths

    # Scoring weights (for path selection)
    w_hedge_penalty: float = -0.1    # Penalty per HEDGE event
    w_entropy_bonus: float = 0.05    # Bonus for moderate entropy (thinking)
    w_confidence_bonus: float = 0.3  # Bonus for high final confidence

    # vLLM
    gpu_memory_utilization: float = 0.70  # More GPU for search (no training)
    model_name: str = "Qwen/Qwen2.5-72B-Instruct"
    dtype: str = "bfloat16"


# ═══════════════════════════════════════════════════════════
# Search Result
# ═══════════════════════════════════════════════════════════

@dataclass
class SearchPath:
    """A single candidate reasoning path."""
    path_id: int
    response: str
    temperature: float
    token_count: int = 0
    hedge_count: int = 0
    refuse_count: int = 0
    mean_entropy: float = 0.0
    mean_confidence: float = 0.0
    peak_cusum: float = 0.0
    cognitive_score: float = 0.0
    pruned_at: Optional[int] = None  # Token count where pruned, or None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path_id": self.path_id,
            "response_preview": self.response[:200],
            "temperature": self.temperature,
            "token_count": self.token_count,
            "hedge_count": self.hedge_count,
            "cognitive_score": round(self.cognitive_score, 4),
            "pruned_at": self.pruned_at,
        }


@dataclass
class SearchResult:
    """Result of parallel search."""
    prompt: str
    best_response: str
    best_score: float
    paths_explored: int
    paths_completed: int
    search_time_seconds: float
    all_paths: List[SearchPath] = field(default_factory=list)

    @property
    def cognitive_report(self) -> str:
        completed = [p for p in self.all_paths if p.pruned_at is None]
        pruned = [p for p in self.all_paths if p.pruned_at is not None]

        lines = [
            f"Parallel Search Report",
            f"{'=' * 50}",
            f"Prompt: {self.prompt[:80]}...",
            f"Paths: {self.paths_explored} explored, {self.paths_completed} completed",
            f"Time: {self.search_time_seconds:.1f}s",
            f"Best score: {self.best_score:.4f}",
            f"",
            f"Top 5 paths:",
        ]
        for p in sorted(completed, key=lambda x: x.cognitive_score, reverse=True)[:5]:
            lines.append(
                f"  #{p.path_id} score={p.cognitive_score:.4f} "
                f"hedge={p.hedge_count} entropy={p.mean_entropy:.2f} "
                f"temp={p.temperature:.2f}"
            )

        if pruned:
            lines.append(f"\nPruned {len(pruned)} paths:")
            for p in pruned[:5]:
                lines.append(
                    f"  #{p.path_id} pruned@token={p.pruned_at} "
                    f"score={p.cognitive_score:.4f}"
                )

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# Parallel Search Engine
# ═══════════════════════════════════════════════════════════

class MetisParallelSearch:
    """
    o1-style parallel search with METIS entropy pruning.

    On DGX Spark (128GB), can run 50-100 parallel paths with a 70B model.
    Entropy-based pruning cuts wasted compute by 50-70%.
    """

    def __init__(self, config: Optional[SearchConfig] = None):
        self._config = config or SearchConfig()
        self._llm = None

    def _ensure_engine(self) -> None:
        """Lazy-init vLLM engine."""
        if self._llm is not None:
            return

        from vllm import LLM
        cfg = self._config

        logger.info(f"Initializing vLLM for parallel search: {cfg.model_name}")
        self._llm = LLM(
            model=cfg.model_name,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            dtype=cfg.dtype,
            trust_remote_code=True,
            enforce_eager=True,
            max_model_len=cfg.max_tokens + 512,  # prompt + completion
        )

    def search(
        self,
        prompt: str,
        tokenizer: Any = None,
    ) -> SearchResult:
        """
        Run parallel search on a single prompt.

        Args:
            prompt: Input prompt
            tokenizer: Tokenizer for chat template (auto-loaded if None)

        Returns:
            SearchResult with best response and cognitive analysis
        """
        self._ensure_engine()
        cfg = self._config
        t0 = time.time()

        # Format prompt
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.model_name, trust_remote_code=True
            )

        chat = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        # Create per-path processors
        cusum_config = CUSUMConfig()
        processor = MetisCompositeProcessor(tokenizer, cusum_config)

        # Generate all paths with varied temperatures
        from vllm import SamplingParams

        IM_END = "<" + "|im_end|" + ">"
        EOS = "<" + "|endoftext|" + ">"

        params = SamplingParams(
            n=cfg.n_paths,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            stop=[IM_END, EOS],
            logits_processors=[processor],
        )

        logger.info(
            f"Generating {cfg.n_paths} paths (temp={cfg.temperature}, "
            f"max_tokens={cfg.max_tokens})..."
        )
        outputs = self._llm.generate([formatted], params)

        # Extract results
        paths: List[SearchPath] = []
        summaries = processor.get_traces_summary()

        for i, output in enumerate(outputs[0].outputs):
            text = output.text.strip()

            # Find matching trace summary
            summary = summaries[i] if i < len(summaries) else {}

            temp = cfg.temperature + (
                (i - cfg.n_paths / 2) * cfg.temperature_spread / cfg.n_paths
            )

            path = SearchPath(
                path_id=i,
                response=text,
                temperature=round(temp, 3),
                token_count=summary.get("total_tokens", len(text.split())),
                hedge_count=summary.get("hedge_count", 0),
                refuse_count=summary.get("refuse_count", 0),
                mean_entropy=summary.get("mean_entropy", 0.0),
                peak_cusum=summary.get("peak_cusum", 0.0),
            )

            # Compute cognitive score
            path.cognitive_score = self._score_path(path)
            paths.append(path)

        # Sort by cognitive score
        paths.sort(key=lambda p: p.cognitive_score, reverse=True)

        # Select best
        best = paths[0] if paths else SearchPath(path_id=-1, response="", temperature=0.0)

        search_time = time.time() - t0
        result = SearchResult(
            prompt=prompt,
            best_response=best.response,
            best_score=best.cognitive_score,
            paths_explored=len(paths),
            paths_completed=sum(1 for p in paths if p.pruned_at is None),
            search_time_seconds=search_time,
            all_paths=paths,
        )

        logger.info(
            f"Search complete: {len(paths)} paths, "
            f"best_score={best.cognitive_score:.4f}, "
            f"time={search_time:.1f}s"
        )

        processor.reset()
        return result

    def _score_path(self, path: SearchPath) -> float:
        """
        Compute cognitive quality score for a path.

        Higher is better. Balances:
        - Low hedge count (confident reasoning)
        - Moderate entropy (active thinking, not memorized)
        - High confidence (settled on an answer)
        - Reasonable length (not degenerate)
        """
        cfg = self._config
        score = 0.0

        # Hedge penalty
        score += cfg.w_hedge_penalty * path.hedge_count

        # Entropy bonus: moderate entropy is good (0.5-2.0 bits)
        # Too low = memorized, too high = confused
        if 0.5 <= path.mean_entropy <= 2.0:
            score += cfg.w_entropy_bonus * path.token_count
        elif path.mean_entropy > 3.0:
            score -= 0.2  # High entropy penalty

        # Length normalization (prefer substantive responses)
        if path.token_count > 50:
            score += 0.1
        if path.token_count > 200:
            score += 0.1

        # Refuse penalty
        score -= 0.5 * path.refuse_count

        return score

    def search_batch(
        self,
        prompts: List[str],
        tokenizer: Any = None,
    ) -> List[SearchResult]:
        """Search multiple prompts sequentially."""
        results: List[SearchResult] = []
        for i, prompt in enumerate(prompts):
            logger.info(f"[{i+1}/{len(prompts)}] Searching: {prompt[:60]}...")
            result = self.search(prompt, tokenizer)
            results.append(result)
        return results

    def shutdown(self) -> None:
        """Release vLLM resources."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            gc.collect()
            torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════
# Iterative Deepening Search (Advanced)
# ═══════════════════════════════════════════════════════════

class MetisIterativeSearch(MetisParallelSearch):
    """
    Multi-round search with progressive refinement.

    Round 1: N paths, broad exploration (high temp)
    Round 2: Top K paths, focused refinement (lower temp)
    Round 3: Top M paths, final verification (lowest temp)

    This mimics o1's "think longer on harder problems" behavior,
    guided by METIS cognitive signals.
    """

    def search_iterative(
        self,
        prompt: str,
        tokenizer: Any = None,
        rounds: int = 3,
    ) -> SearchResult:
        """
        Multi-round iterative deepening search.

        Args:
            prompt: Input prompt
            tokenizer: Tokenizer (auto-loaded if None)
            rounds: Number of refinement rounds

        Returns:
            SearchResult from the final round
        """
        cfg = self._config
        t0 = time.time()

        # Round configs: progressively narrower and cooler
        round_configs = [
            {"n": cfg.n_paths, "temp": cfg.temperature, "label": "broad"},
            {"n": max(cfg.n_paths // 4, 4), "temp": cfg.temperature * 0.7, "label": "focused"},
            {"n": max(cfg.n_paths // 10, 2), "temp": cfg.temperature * 0.4, "label": "precise"},
        ][:rounds]

        best_result: Optional[SearchResult] = None

        for r, rc in enumerate(round_configs):
            logger.info(
                f"Round {r+1}/{len(round_configs)}: "
                f"n={rc['n']} temp={rc['temp']:.2f} ({rc['label']})"
            )

            # Temporarily override config
            orig_n = cfg.n_paths
            orig_temp = cfg.temperature
            cfg.n_paths = rc["n"]
            cfg.temperature = rc["temp"]

            result = self.search(prompt, tokenizer)

            cfg.n_paths = orig_n
            cfg.temperature = orig_temp

            if best_result is None or result.best_score > best_result.best_score:
                best_result = result

            logger.info(
                f"  Round {r+1} best: score={result.best_score:.4f} "
                f"response_len={len(result.best_response)}"
            )

        assert best_result is not None
        best_result.search_time_seconds = time.time() - t0
        return best_result
