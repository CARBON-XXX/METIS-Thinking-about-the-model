"""
METIS Search — Counterfactual Simulation

At high-risk decision points (CUSUM alarm / Cornish-Fisher threshold),
forces the model to explore "what if premise A doesn't hold?" branches.

Mechanism:
═══════════════════════════════════════════════════════════════════
1. DETECT: CUSUM alarm or entropy spike triggers counterfactual check
2. BRANCH: Generate K alternative continuations with premise modifiers
3. COMPARE: Measure entropy derivatives across branches
4. JUDGE:
   - Low variance → conclusion is robust (insensitive to premises)
   - High variance → conclusion is FRAGILE (mark as unreliable)
═══════════════════════════════════════════════════════════════════

Premise modifiers inject counterfactual probes:
    - Negation:     "However, if [premise] were NOT true, ..."
    - Weakening:    "Suppose [premise] only partially holds, ..."
    - Alternative:  "What if instead of [premise], we had [alt], ..."

Academic value:
    Sensitivity analysis via information-theoretic probes.
    If H(conclusion | premise) ≈ H(conclusion | ¬premise),
    the model is CONFABULATING — the conclusion doesn't actually
    depend on the reasoning, only on memorized patterns.

    Entropy derivative: dH/dP = H(s|P) - H(s|¬P)
    Fragility score: Var(H across branches) / Mean(H)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from metis.search.tree_node import SearchConfig, clone_kv_cache

logger = logging.getLogger("metis.search")


# ─────────────────────────────────────────────────────
# Counterfactual Probe Templates
# ─────────────────────────────────────────────────────

PROBE_TEMPLATES = {
    "negate": [
        "However, if the opposite were true, ",
        "But what if this assumption is wrong? ",
        "Suppose the premise doesn't hold — then ",
    ],
    "weaken": [
        "If this only partially applies, ",
        "Assuming this is only approximately correct, ",
        "In the case where this is weakly supported, ",
    ],
    "alternative": [
        "Alternatively, consider that ",
        "From a different perspective, ",
        "If we take the opposite approach, ",
    ],
}


# ─────────────────────────────────────────────────────
# Result Types
# ─────────────────────────────────────────────────────

@dataclass
class CounterfactualBranch:
    """Single counterfactual branch result."""
    probe_type: str = ""             # "negate" | "weaken" | "alternative"
    probe_text: str = ""             # Injected probe prefix
    continuation: str = ""           # Generated continuation after probe
    mean_entropy: float = 0.0       # Mean H over continuation
    peak_entropy: float = 0.0       # Max H in continuation
    entropy_gradient: float = 0.0   # dH/dt trend in continuation
    tokens_generated: int = 0
    entropies: List[float] = field(default_factory=list)


@dataclass
class CounterfactualResult:
    """Result of counterfactual simulation at a decision point."""
    # Sensitivity analysis
    fragility_score: float = 0.0     # Var(H) / Mean(H) across branches — high = fragile
    entropy_derivative: float = 0.0  # Max |H(original) - H(counterfactual)|
    is_fragile: bool = False         # fragility > threshold

    # Original path baseline
    original_entropy: float = 0.0

    # Branch results
    branches: List[CounterfactualBranch] = field(default_factory=list)

    # Judgment
    verdict: str = "robust"          # "robust" | "fragile" | "confabulating"
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fragility_score": round(self.fragility_score, 4),
            "entropy_derivative": round(self.entropy_derivative, 4),
            "is_fragile": self.is_fragile,
            "original_entropy": round(self.original_entropy, 4),
            "verdict": self.verdict,
            "explanation": self.explanation,
            "n_branches": len(self.branches),
            "branches": [
                {
                    "type": b.probe_type,
                    "mean_H": round(b.mean_entropy, 4),
                    "peak_H": round(b.peak_entropy, 4),
                    "tokens": b.tokens_generated,
                }
                for b in self.branches
            ],
        }


# ─────────────────────────────────────────────────────
# Counterfactual Simulator
# ─────────────────────────────────────────────────────

class CounterfactualSimulator:
    """
    Counterfactual Simulation Engine — premise sensitivity analysis.

    At high-risk nodes (CUSUM alarm, high z-score), spawns counterfactual
    branches with modified premises. Compares entropy distributions
    across branches to detect confabulation and fragile reasoning.

    Usage:
        from metis.search import CounterfactualSimulator, SearchConfig

        sim = CounterfactualSimulator(model, tokenizer)
        result = sim.simulate(
            prompt_ids=...,
            kv_cache=...,
            generated_so_far=...,
            original_entropy=2.5,
        )
        if result.is_fragile:
            print(f"WARNING: conclusion is fragile (F={result.fragility_score:.2f})")
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[SearchConfig] = None,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._config = config or SearchConfig()
        self._device = str(next(model.parameters()).device)

    @torch.inference_mode()
    def simulate(
        self,
        prompt_ids: torch.Tensor,
        kv_cache: Any,
        generated_tokens: List[int],
        original_entropy: float,
        continuation_length: int = 32,
    ) -> CounterfactualResult:
        """
        Run counterfactual simulation at the current decision point.

        Args:
            prompt_ids: Original prompt token IDs [1, seq_len]
            kv_cache: KV cache state at the decision point
            generated_tokens: Tokens generated so far (before CF point)
            original_entropy: METIS entropy at the trigger point
            continuation_length: How many tokens to generate per CF branch

        Returns:
            CounterfactualResult with sensitivity analysis
        """
        from metis import Metis

        config = self._config
        branches: List[CounterfactualBranch] = []

        # ── Select probe types ──
        # Use one probe of each type, up to counterfactual_branches
        probe_types = ["negate", "weaken", "alternative"]
        n_branches = min(config.counterfactual_branches, len(probe_types))

        for i in range(n_branches):
            probe_type = probe_types[i % len(probe_types)]
            templates = PROBE_TEMPLATES[probe_type]
            probe_text = templates[i % len(templates)]

            branch = self._run_branch(
                prompt_ids=prompt_ids,
                kv_cache=kv_cache,
                generated_tokens=generated_tokens,
                probe_text=probe_text,
                probe_type=probe_type,
                continuation_length=continuation_length,
            )
            branches.append(branch)

        # ── Sensitivity Analysis ──
        if not branches:
            return CounterfactualResult(
                original_entropy=original_entropy,
                verdict="robust",
                explanation="No counterfactual branches generated",
            )

        branch_means = [b.mean_entropy for b in branches]
        all_means = branch_means + [original_entropy]

        global_mean = sum(all_means) / len(all_means)
        variance = sum((h - global_mean) ** 2 for h in all_means) / len(all_means)

        # Fragility = coefficient of variation of entropy across branches
        # High variance relative to mean → conclusion is premise-sensitive
        fragility = math.sqrt(variance) / max(global_mean, 0.01)

        # Entropy derivative: max deviation from original
        max_derivative = max(abs(h - original_entropy) for h in branch_means)

        # Verdict logic
        is_fragile = fragility > config.sensitivity_threshold
        if fragility > config.sensitivity_threshold * 2:
            verdict = "confabulating"
            explanation = (
                f"Entropy varies wildly across premises (F={fragility:.2f}). "
                f"The conclusion does not depend on the stated reasoning — "
                f"likely pattern-matching / confabulation."
            )
        elif is_fragile:
            verdict = "fragile"
            explanation = (
                f"Moderate premise sensitivity (F={fragility:.2f}). "
                f"The conclusion may change if assumptions are altered. "
                f"Max entropy shift: {max_derivative:.2f} bits."
            )
        else:
            verdict = "robust"
            explanation = (
                f"Low premise sensitivity (F={fragility:.2f}). "
                f"Conclusion is stable across counterfactual scenarios."
            )

        logger.info(
            f"[CF] {verdict.upper()}: F={fragility:.3f}, "
            f"dH_max={max_derivative:.3f}, "
            f"orig_H={original_entropy:.3f}, "
            f"branch_H={[round(h, 2) for h in branch_means]}"
        )

        return CounterfactualResult(
            fragility_score=fragility,
            entropy_derivative=max_derivative,
            is_fragile=is_fragile,
            original_entropy=original_entropy,
            branches=branches,
            verdict=verdict,
            explanation=explanation,
        )

    # ─────────────────────────────────────────────────────
    # Internal: Run single CF branch
    # ─────────────────────────────────────────────────────

    def _run_branch(
        self,
        prompt_ids: torch.Tensor,
        kv_cache: Any,
        generated_tokens: List[int],
        probe_text: str,
        probe_type: str,
        continuation_length: int,
    ) -> CounterfactualBranch:
        """
        Run a single counterfactual branch.

        Injects probe_text, then generates continuation_length tokens,
        collecting METIS entropy at each step.
        """
        from metis import Metis

        # ── Inject probe tokens ──
        probe_ids = self._tokenizer.encode(probe_text, add_special_tokens=False)
        if not probe_ids:
            return CounterfactualBranch(probe_type=probe_type, probe_text=probe_text)

        # ── Build input: prompt + generated + probe ──
        all_ids = (
            prompt_ids[0].tolist()
            + generated_tokens
            + probe_ids
        )
        input_tensor = torch.tensor([all_ids], device=self._device)

        # Full forward pass to get KV cache at probe position
        # (Cannot reuse parent KV because probe tokens change the context)
        outputs = self._model(
            input_ids=input_tensor,
            use_cache=True,
            return_dict=True,
        )
        branch_kv = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

        # ── Initialize METIS for this branch ──
        metis = Metis.attach(self._model, self._tokenizer)
        metis.start_session("")

        # ── Generate continuation with METIS tracking ──
        entropies: List[float] = []
        cont_tokens: List[int] = []
        current_kv = branch_kv

        for step in range(continuation_length):
            # METIS evaluation
            signal = metis.step(logits)
            entropies.append(signal.semantic_entropy)

            # Sample next token
            probs = F.softmax(logits / 0.7, dim=-1)
            token_id = torch.multinomial(probs, num_samples=1)
            cont_tokens.append(token_id.item())

            # Check EOS
            if token_id.item() == self._tokenizer.eos_token_id:
                break

            # Next step
            outputs = self._model(
                input_ids=token_id.unsqueeze(0),
                past_key_values=current_kv,
                use_cache=True,
                return_dict=True,
            )
            current_kv = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

        # ── Compute statistics ──
        mean_h = sum(entropies) / max(len(entropies), 1)
        peak_h = max(entropies) if entropies else 0.0

        # Entropy gradient: linear regression slope over continuation
        grad = 0.0
        if len(entropies) >= 3:
            n = len(entropies)
            x_mean = (n - 1) / 2
            y_mean = mean_h
            num = sum((i - x_mean) * (h - y_mean) for i, h in enumerate(entropies))
            den = sum((i - x_mean) ** 2 for i in range(n))
            grad = num / (den + 1e-8)

        continuation = self._tokenizer.decode(cont_tokens, skip_special_tokens=True)

        # Cleanup
        del current_kv, branch_kv

        return CounterfactualBranch(
            probe_type=probe_type,
            probe_text=probe_text,
            continuation=continuation,
            mean_entropy=mean_h,
            peak_entropy=peak_h,
            entropy_gradient=grad,
            tokens_generated=len(cont_tokens),
            entropies=entropies,
        )
