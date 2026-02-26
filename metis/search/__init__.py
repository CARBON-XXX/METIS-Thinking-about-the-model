"""
METIS Search — Entropy-Guided Tree Search & Counterfactual Simulation

Architecture evolution: from "emergency braking" (AEB) to "autonomous navigation".

Components:
    tree_node.py        — TreeNode with KV cache snapshot management
    entropy_search.py   — Entropy-Guided Tree Search (EGTS) engine
    counterfactual.py   — Counterfactual Simulation at high-risk decision points
"""
from metis.search.tree_node import TreeNode, SearchConfig
from metis.search.entropy_search import EntropyGuidedSearch
from metis.search.counterfactual import CounterfactualSimulator

__all__ = [
    "TreeNode",
    "SearchConfig",
    "EntropyGuidedSearch",
    "CounterfactualSimulator",
]
