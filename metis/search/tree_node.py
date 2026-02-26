"""
METIS Search — Tree Node with KV Cache Snapshot Management

Each TreeNode represents a partial generation state in the search tree.
Stores token sequence, KV cache snapshot, and METIS cognitive signals
for efficient branching and backtracking.

Memory model:
    KV cache is the dominant memory consumer. For a 72B model with
    80 layers × 2 (K+V) × hidden_dim × seq_len × bf16:
    ~50MB per 100 tokens per node.

    With beam_width=16 and max_depth=256: ~25GB peak — fits in DGX 128GB.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


# ─────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────

@dataclass
class SearchConfig:
    """Configuration for entropy-guided tree search."""
    # Branching thresholds
    entropy_branch_threshold: float = 2.0    # H(s) > θ → expand node
    entropy_converge_threshold: float = 0.5  # H(s) < θ → path is promising
    z_score_branch_threshold: float = 1.5    # z > θ → anomalous entropy → branch

    # Tree structure
    beam_width: int = 16                     # DGX: wide beam for thorough exploration
    max_depth: int = 256                     # DGX: deep reasoning chains
    max_nodes: int = 2048                    # DGX 128GB: generous node budget

    # UCB1 selection
    exploration_weight: float = 1.414        # √2 = UCB1 standard
    entropy_penalty_weight: float = 0.5      # How much entropy penalizes node value

    # Temperature spread for child sampling
    base_temperature: float = 0.7
    temperature_spread: float = 0.15         # Each child offset from base

    # Pruning
    min_branch_tokens: int = 4               # Don't branch before N tokens
    prune_ratio: float = 0.5                 # Keep top 50% of nodes at each level

    # Counterfactual
    counterfactual_branches: int = 3         # Number of CF branches per trigger
    sensitivity_threshold: float = 0.3       # Entropy variance > θ → fragile conclusion


# ─────────────────────────────────────────────────────
# Tree Node
# ─────────────────────────────────────────────────────

class TreeNode:
    """
    Single node in the entropy-guided search tree.

    Each node represents a partial generation state:
    - Token sequence generated so far
    - KV cache snapshot (for efficient continuation)
    - METIS cognitive signal at this position
    - UCB1 statistics for selection

    Tree structure:
        root (prompt)
        ├── child_0 (low entropy path)
        │   └── child_0_0 (converged → terminal)
        ├── child_1 (medium entropy)
        │   ├── child_1_0 (branched again)
        │   └── child_1_1
        └── child_2 (high entropy → pruned)
    """

    __slots__ = (
        "token_ids", "kv_cache", "parent", "children", "depth",
        "entropy", "z_score", "confidence", "cognitive_phase",
        "decision", "cusum_alarm", "entropy_gradient",
        "visit_count", "value_sum", "is_terminal", "is_pruned",
        "temperature", "_text_cache",
    )

    def __init__(
        self,
        token_ids: List[int],
        kv_cache: Any = None,
        parent: Optional[TreeNode] = None,
        depth: int = 0,
        entropy: float = 0.0,
        z_score: float = 0.0,
        confidence: float = 0.0,
        cognitive_phase: str = "recall",
        decision: str = "normal",
        cusum_alarm: bool = False,
        entropy_gradient: float = 0.0,
        temperature: float = 0.7,
    ):
        self.token_ids = token_ids
        self.kv_cache = kv_cache
        self.parent = parent
        self.children: List[TreeNode] = []
        self.depth = depth

        # METIS cognitive state at this node
        self.entropy = entropy
        self.z_score = z_score
        self.confidence = confidence
        self.cognitive_phase = cognitive_phase
        self.decision = decision
        self.cusum_alarm = cusum_alarm
        self.entropy_gradient = entropy_gradient
        self.predictive_entropy_value: Optional[float] = None  # Expected terminal entropy E[H(T)]

        # UCB1 statistics
        self.visit_count = 0
        self.value_sum = 0.0

        # Node state
        self.is_terminal = False
        self.is_pruned = False
        self.temperature = temperature
        self._text_cache: Optional[str] = None

    @property
    def value(self) -> float:
        """Average node value (lower entropy = higher value)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def path_entropy(self) -> float:
        """Average entropy along the path from root to this node."""
        node = self
        entropies = []
        while node is not None:
            entropies.append(node.entropy)
            node = node.parent
        return sum(entropies) / max(len(entropies), 1)

    def ucb1_score(self, parent_visits: int, c: float = 1.414) -> float:
        """
        UCB1 score for tree policy selection.

        UCB1 = Q(s) + c * √(ln(N_parent) / N_child)

        Modified: Q(s) combines immediate entropy and predicted terminal entropy.
        """
        if self.visit_count == 0:
            return float("inf")  # Unexplored → highest priority
            
        exploitation = self.value
        # If we have a predictive value network estimate, blend it in (alpha=0.5)
        if self.predictive_entropy_value is not None:
            exploitation = 0.5 * exploitation + 0.5 * (-self.predictive_entropy_value)
            
        exploration = c * math.sqrt(math.log(parent_visits) / self.visit_count)
        return exploitation + exploration

    def backpropagate(self, reward: float) -> None:
        """Propagate reward (negative entropy) up to root."""
        node: Optional[TreeNode] = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += reward
            node = node.parent

    def get_path_tokens(self) -> List[int]:
        """Collect full token sequence from root to this node."""
        path: List[List[int]] = []
        node: Optional[TreeNode] = self
        while node is not None:
            path.append(node.token_ids)
            node = node.parent
        path.reverse()
        result: List[int] = []
        for segment in path:
            result.extend(segment)
        return result

    def decode(self, tokenizer: Any) -> str:
        """Decode full path text (cached)."""
        if self._text_cache is None:
            all_tokens = self.get_path_tokens()
            self._text_cache = tokenizer.decode(all_tokens, skip_special_tokens=True)
        return self._text_cache

    def should_branch(self, config: SearchConfig) -> bool:
        """
        Determine if this node should expand children.

        Branch when:
        1. Entropy exceeds threshold (model is uncertain)
        2. CUSUM alarm triggered (sustained difficulty detected)
        3. z-score is anomalously high
        AND depth exceeds minimum
        """
        if self.depth < config.min_branch_tokens:
            return False
        if self.is_terminal or self.is_pruned:
            return False
        if len(self.children) > 0:
            return False  # Already expanded

        return (
            self.entropy > config.entropy_branch_threshold
            or self.cusum_alarm
            or self.z_score > config.z_score_branch_threshold
        )

    def release_kv_cache(self) -> None:
        """Free KV cache memory (call after node is fully expanded or pruned)."""
        self.kv_cache = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node for debugging/visualization."""
        return {
            "depth": self.depth,
            "n_tokens": len(self.token_ids),
            "entropy": round(self.entropy, 4),
            "z_score": round(self.z_score, 4),
            "confidence": round(self.confidence, 4),
            "cognitive_phase": self.cognitive_phase,
            "decision": self.decision,
            "cusum_alarm": self.cusum_alarm,
            "visit_count": self.visit_count,
            "value": round(self.value, 4),
            "is_terminal": self.is_terminal,
            "is_pruned": self.is_pruned,
            "n_children": len(self.children),
        }


# ─────────────────────────────────────────────────────
# KV Cache Management
# ─────────────────────────────────────────────────────

def clone_kv_cache(kv_cache: Any) -> Any:
    """
    Deep-clone KV cache for tree branching.

    Each child node needs an independent copy of the parent's KV cache
    to continue generation without cross-contamination.

    Supports both DynamicCache (transformers>=4.36) and raw tuples.
    """
    if kv_cache is None:
        return None

    try:
        from transformers import DynamicCache
        if isinstance(kv_cache, DynamicCache):
            clone = DynamicCache()
            for layer_idx in range(len(kv_cache)):
                k, v = kv_cache[layer_idx]
                clone.update(k.clone(), v.clone(), layer_idx)
            return clone
    except ImportError:
        pass

    # Fallback: raw tuple format
    if isinstance(kv_cache, tuple):
        return tuple(
            tuple(t.clone() for t in layer) for layer in kv_cache
        )

    return kv_cache
