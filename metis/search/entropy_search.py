"""
METIS Search — Entropy-Guided Tree Search (EGTS)

Replaces linear generation with tree-structured exploration:
    When METIS detects entropy spike H(s) > θ, instead of blocking/hedging,
    it EXPANDS the search tree to explore alternative reasoning paths,
    then selects the path with lowest entropy convergence.

Algorithm:
═══════════════════════════════════════════════════════════════════
1. GENERATE tokens linearly until H(s) > θ_branch (entropy spike)
2. EXPAND: spawn K child nodes with varied temperatures
3. SIMULATE: continue each child for D tokens, collecting METIS signals
4. EVALUATE: score paths by negative mean entropy (low H = good)
5. SELECT: UCB1-based selection of most promising path
6. REPEAT until convergence (H < θ_converge) or max_depth
═══════════════════════════════════════════════════════════════════

Academic contribution:
    Test-time self-correction via information-theoretic tree search.
    System 2 compute is allocated precisely to high-entropy nodes,
    not uniformly across all tokens (compute-optimal reasoning).

    Key formula:
        h(s) = -E[H(s')]   (heuristic = expected negative entropy of children)
        UCB1(s) = Q(s) + c·√(ln N_parent / N_s)
        Q(s) = -(mean_entropy - α·entropy_gradient)   (reward convergence)
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from metis.search.tree_node import TreeNode, SearchConfig, clone_kv_cache

logger = logging.getLogger("metis.search")


@dataclass
class SearchResult:
    """Result of entropy-guided tree search."""
    text: str = ""
    tokens: List[int] = field(default_factory=list)
    total_nodes_created: int = 0
    total_nodes_pruned: int = 0
    branch_points: int = 0
    best_path_entropy: float = 0.0
    best_path_depth: int = 0
    search_time_ms: float = 0.0
    tree_summary: Dict[str, Any] = field(default_factory=dict)

    # Cognitive trace of the selected path
    path_entropies: List[float] = field(default_factory=list)
    path_phases: List[str] = field(default_factory=list)
    convergence_achieved: bool = False


class EntropyGuidedSearch:
    """
    Entropy-Guided Tree Search (EGTS) — METIS-powered reasoning navigation.

    Instead of linear generation that stops at uncertainty,
    EGTS explores multiple reasoning paths and selects the one
    where the model's entropy converges (genuine understanding).

    Usage:
        from metis.search import EntropyGuidedSearch, SearchConfig

        searcher = EntropyGuidedSearch(model, tokenizer, config=SearchConfig())
        result = searcher.search("What is quantum entanglement?")
        print(result.text)
        print(f"Explored {result.total_nodes_created} nodes, "
              f"branched {result.branch_points} times")
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[SearchConfig] = None,
        metis_config: Optional[Any] = None,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._config = config or SearchConfig()
        self._metis_config = metis_config
        self._device = str(next(model.parameters()).device)

        # Statistics
        self._total_forward_passes = 0

    @torch.inference_mode()
    def search(
        self,
        prompt: str,
        max_tokens: int = 256,
        chat_template: bool = True,
    ) -> SearchResult:
        """
        Run entropy-guided tree search on a prompt.

        Args:
            prompt: Input text
            max_tokens: Maximum total tokens across all paths
            chat_template: Apply chat template to prompt

        Returns:
            SearchResult with best path text and search statistics
        """
        from metis import Metis

        t0 = time.perf_counter()
        config = self._config

        # ── Prepare prompt ──
        if chat_template and hasattr(self._tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            text = prompt

        prompt_ids = self._tokenizer.encode(text, return_tensors="pt")
        if not isinstance(prompt_ids, torch.Tensor):
            prompt_ids = torch.tensor([prompt_ids])
        prompt_ids = prompt_ids.to(self._device)

        # ── Initial forward pass (prompt encoding) ──
        outputs = self._model(
            input_ids=prompt_ids, use_cache=True, return_dict=True,
        )
        kv_cache = outputs.past_key_values
        first_logits = outputs.logits[:, -1, :]
        self._total_forward_passes = 1

        # ── Initialize METIS for heuristic evaluation ──
        metis = Metis.attach(self._model, self._tokenizer)
        metis.start_session(prompt)
        signal = metis.step(first_logits)

        # ── Create root node ──
        root = TreeNode(
            token_ids=[],
            kv_cache=kv_cache,
            depth=0,
            entropy=signal.semantic_entropy,
            z_score=signal.z_score,
            confidence=signal.confidence,
            cognitive_phase=signal.cognitive_phase,
            decision=signal.decision.value,
            cusum_alarm=signal.cusum_alarm,
            entropy_gradient=signal.entropy_gradient,
        )

        # ── Search loop ──
        all_nodes: List[TreeNode] = [root]
        active_leaves: List[TreeNode] = [root]
        branch_count = 0
        prune_count = 0
        eos_id = self._tokenizer.eos_token_id

        for step in range(max_tokens):
            if not active_leaves:
                break
            if len(all_nodes) >= config.max_nodes:
                logger.info(f"[EGTS] Node cap reached ({config.max_nodes})")
                break

            # ── SELECT: pick leaf with highest UCB1 ──
            total_visits = sum(n.visit_count for n in all_nodes) + 1
            best_leaf = max(
                active_leaves,
                key=lambda n: n.ucb1_score(total_visits, config.exploration_weight),
            )

            # ── EXPAND or STEP ──
            if best_leaf.should_branch(config) and len(all_nodes) + config.beam_width <= config.max_nodes:
                # Branch: create K children with different temperatures
                children = self._expand_node(
                    best_leaf, prompt_ids, metis, config,
                )
                all_nodes.extend(children)
                active_leaves.remove(best_leaf)
                active_leaves.extend([c for c in children if not c.is_terminal])
                branch_count += 1

                # Free parent KV cache (children have their own copies)
                best_leaf.release_kv_cache()

                logger.info(
                    f"[EGTS] Branch at depth={best_leaf.depth} "
                    f"H={best_leaf.entropy:.3f} z={best_leaf.z_score:.2f} "
                    f"→ {len(children)} children"
                )
            else:
                # Linear step: generate one token
                child = self._step_node(
                    best_leaf, prompt_ids, metis, eos_id,
                )
                if child is not None:
                    all_nodes.append(child)
                    active_leaves.remove(best_leaf)
                    if not child.is_terminal:
                        active_leaves.append(child)
                    # Free parent KV
                    best_leaf.release_kv_cache()
                else:
                    # Step failed
                    best_leaf.is_terminal = True
                    active_leaves.remove(best_leaf)

            # ── PRUNE: remove low-value leaves periodically ──
            if step > 0 and step % 16 == 0 and len(active_leaves) > config.beam_width:
                keep_n = max(config.beam_width, int(len(active_leaves) * config.prune_ratio))
                active_leaves.sort(key=lambda n: n.value, reverse=True)
                pruned = active_leaves[keep_n:]
                active_leaves = active_leaves[:keep_n]
                for p in pruned:
                    p.is_pruned = True
                    p.release_kv_cache()
                    prune_count += len(pruned)

            # ── Check convergence ──
            converged = [
                n for n in active_leaves
                if n.entropy < config.entropy_converge_threshold
                and n.depth >= config.min_branch_tokens
            ]
            if converged:
                # Found a low-entropy path — select best
                best = min(converged, key=lambda n: n.path_entropy)
                best.backpropagate(reward=-best.path_entropy)
                logger.info(
                    f"[EGTS] Converged at depth={best.depth} "
                    f"path_H={best.path_entropy:.3f}"
                )
                break

        # ── SELECT best terminal/leaf path ──
        candidates = [n for n in all_nodes if n.is_terminal or n.is_leaf]
        if not candidates:
            candidates = all_nodes

        best_node = min(candidates, key=lambda n: n.path_entropy)
        best_tokens = best_node.get_path_tokens()
        best_text = self._tokenizer.decode(best_tokens, skip_special_tokens=True)

        # ── Collect path statistics ──
        path_entropies: List[float] = []
        path_phases: List[str] = []
        node: Optional[TreeNode] = best_node
        while node is not None:
            path_entropies.append(node.entropy)
            path_phases.append(node.cognitive_phase)
            node = node.parent
        path_entropies.reverse()
        path_phases.reverse()

        elapsed = (time.perf_counter() - t0) * 1000

        # ── Cleanup all KV caches ──
        for n in all_nodes:
            n.release_kv_cache()

        result = SearchResult(
            text=best_text,
            tokens=best_tokens,
            total_nodes_created=len(all_nodes),
            total_nodes_pruned=prune_count,
            branch_points=branch_count,
            best_path_entropy=best_node.path_entropy,
            best_path_depth=best_node.depth,
            search_time_ms=elapsed,
            path_entropies=path_entropies,
            path_phases=path_phases,
            convergence_achieved=best_node.entropy < config.entropy_converge_threshold,
            tree_summary={
                "total_forward_passes": self._total_forward_passes,
                "max_tree_depth": max((n.depth for n in all_nodes), default=0),
                "terminal_nodes": sum(1 for n in all_nodes if n.is_terminal),
            },
        )

        logger.info(
            f"[EGTS] Search complete: {len(all_nodes)} nodes, "
            f"{branch_count} branches, {prune_count} pruned, "
            f"best_H={best_node.path_entropy:.3f}, "
            f"{elapsed:.0f}ms"
        )

        return result

    # ─────────────────────────────────────────────────────
    # Internal: Node expansion (branching)
    # ─────────────────────────────────────────────────────

    def _expand_node(
        self,
        node: TreeNode,
        prompt_ids: torch.Tensor,
        metis: Any,
        config: SearchConfig,
    ) -> List[TreeNode]:
        """
        Expand a high-entropy node into K children.

        Each child samples with a different temperature to explore
        diverse reasoning paths. METIS evaluates each child.
        """
        children: List[TreeNode] = []
        kv = node.kv_cache
        if kv is None:
            return children

        # Generate full input for this node's position
        path_tokens = node.get_path_tokens()
        if path_tokens:
            path_tensor = torch.tensor([path_tokens[-1:]], device=self._device)
        else:
            path_tensor = None

        for i in range(config.beam_width):
            # Varied temperature for diversity
            temp = config.base_temperature + (i - config.beam_width / 2) * config.temperature_spread
            temp = max(0.1, min(2.0, temp))

            # Clone KV cache for independent evolution
            child_kv = clone_kv_cache(kv)

            # Forward pass to get logits at this position
            if path_tensor is not None:
                outputs = self._model(
                    input_ids=path_tensor,
                    past_key_values=child_kv,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                # Root expansion — use existing logits
                outputs = self._model(
                    input_ids=prompt_ids[:, -1:],
                    past_key_values=child_kv,
                    use_cache=True,
                    return_dict=True,
                )
            self._total_forward_passes += 1

            logits = outputs.logits[:, -1, :]
            child_kv = outputs.past_key_values

            # Sample token with temperature
            probs = F.softmax(logits / temp, dim=-1)
            token_id = torch.multinomial(probs, num_samples=1).item()

            # METIS evaluation
            signal = metis.step(logits)

            child = TreeNode(
                token_ids=[token_id],
                kv_cache=child_kv,
                parent=node,
                depth=node.depth + 1,
                entropy=signal.semantic_entropy,
                z_score=signal.z_score,
                confidence=signal.confidence,
                cognitive_phase=signal.cognitive_phase,
                decision=signal.decision.value,
                cusum_alarm=signal.cusum_alarm,
                entropy_gradient=signal.entropy_gradient,
                temperature=temp,
            )

            # Check EOS
            if token_id == self._tokenizer.eos_token_id:
                child.is_terminal = True

            # Backpropagate initial evaluation
            reward = -signal.semantic_entropy + 0.3 * signal.confidence
            child.backpropagate(reward)

            children.append(child)
            node.children.append(child)

        return children

    # ─────────────────────────────────────────────────────
    # Internal: Single token step
    # ─────────────────────────────────────────────────────

    def _step_node(
        self,
        node: TreeNode,
        prompt_ids: torch.Tensor,
        metis: Any,
        eos_id: int,
    ) -> Optional[TreeNode]:
        """
        Generate one token from a leaf node.

        Linear extension — no branching. Used for low-entropy regions
        where the model is confident.
        """
        kv = node.kv_cache
        if kv is None:
            return None

        # Get last token as input
        path_tokens = node.get_path_tokens()
        if path_tokens:
            input_ids = torch.tensor([[path_tokens[-1]]], device=self._device)
        else:
            input_ids = prompt_ids[:, -1:]

        outputs = self._model(
            input_ids=input_ids,
            past_key_values=kv,
            use_cache=True,
            return_dict=True,
        )
        self._total_forward_passes += 1

        logits = outputs.logits[:, -1, :]
        child_kv = outputs.past_key_values

        # Sample with node's temperature
        probs = F.softmax(logits / node.temperature, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1).item()

        # METIS evaluation
        signal = metis.step(logits)

        child = TreeNode(
            token_ids=[token_id],
            kv_cache=child_kv,
            parent=node,
            depth=node.depth + 1,
            entropy=signal.semantic_entropy,
            z_score=signal.z_score,
            confidence=signal.confidence,
            cognitive_phase=signal.cognitive_phase,
            decision=signal.decision.value,
            cusum_alarm=signal.cusum_alarm,
            entropy_gradient=signal.entropy_gradient,
            temperature=node.temperature,
        )

        if token_id == eos_id:
            child.is_terminal = True

        # Backpropagate
        reward = -signal.semantic_entropy + 0.3 * signal.confidence
        child.backpropagate(reward)

        node.children.append(child)
        return child
