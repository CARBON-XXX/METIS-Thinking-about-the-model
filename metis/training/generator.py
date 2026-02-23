"""
METIS Generator — Token-by-token generation with cognitive tracing.

Uses manual autoregressive loop with KV cache. METIS step() is called
every `metis_stride` tokens (default 4) to reduce CPU overhead while
still producing enough cognitive events for reward computation.

Usage:
    generator = MetisGenerator(model, tokenizer)
    text, trace = generator.generate("What is quantum entanglement?")
    reward = CognitiveRewardComputer().compute(trace)
"""
from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..metis import Metis
from ..core.types import CognitiveTrace, ControllerConfig

logger = logging.getLogger(__name__)


class MetisGenerator:
    """METIS-instrumented text generator with strided cognitive analysis."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        metis_config: Optional[ControllerConfig] = None,
        device: Optional[str] = None,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._device = device or str(next(model.parameters()).device)
        self._metis = Metis.attach(model, tokenizer, config=metis_config)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        metis_stride: int = 4,
    ) -> Tuple[str, CognitiveTrace]:
        """
        Generate text with strided METIS cognitive tracing.

        METIS step() runs every `metis_stride` tokens. For 200 tokens
        with stride=4 → 50 cognitive events, sufficient for rewards.
        """
        inputs = self._tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self._device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

        self._metis.start_session(prompt)

        generated_ids: List[int] = []
        past_key_values = None

        for step in range(max_new_tokens):
            if past_key_values is None:
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
            else:
                last_token = torch.tensor(
                    [[generated_ids[-1]]], device=self._device
                )
                new_mask = torch.cat([
                    attention_mask,
                    torch.ones(1, 1, device=self._device, dtype=attention_mask.dtype),
                ], dim=1)
                outputs = self._model(
                    input_ids=last_token,
                    attention_mask=new_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                attention_mask = new_mask

            logits = outputs.logits[:, -1, :]  # [1, vocab_size]
            past_key_values = outputs.past_key_values

            # METIS cognitive step (strided)
            do_metis = (step % metis_stride == 0)
            if do_metis:
                self._metis.step(logits)

            # Sampling
            log_probs = F.log_softmax(logits.float(), dim=-1)
            next_logits = logits.clone().float()

            if repetition_penalty != 1.0 and generated_ids:
                for prev_id in set(generated_ids):
                    if next_logits[0, prev_id] > 0:
                        next_logits[0, prev_id] /= repetition_penalty
                    else:
                        next_logits[0, prev_id] *= repetition_penalty

            if do_sample and temperature > 0:
                next_logits = next_logits / temperature
                if top_k > 0:
                    topk_vals, _ = torch.topk(next_logits, top_k, dim=-1)
                    min_topk = topk_vals[:, -1].unsqueeze(-1)
                    next_logits[next_logits < min_topk] = float("-inf")
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_logits, descending=True, dim=-1
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                    sorted_logits[sorted_mask] = float("-inf")
                    next_logits.scatter_(1, sorted_indices, sorted_logits)
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = next_logits.argmax(dim=-1)

            token_id = next_token.item()

            if do_metis:
                token_log_prob = log_probs[0, token_id].item()
                surprise = -token_log_prob / math.log(2) if token_log_prob < 0 else 0.0
                self._metis.feed_surprise(surprise)

            generated_ids.append(token_id)
            if token_id == self._tokenizer.eos_token_id:
                break

        del past_key_values, logits, outputs

        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        # CRITICAL: introspect() triggers _aggregate_trace() which populates
        # mean_entropy, mean_surprise, mean_confidence, etc. on the trace.
        # Without this, all aggregate stats remain at default 0.0.
        self._metis.introspect()

        trace = self._metis.trace
        self._metis.end_session()

        return generated_text, trace

    def generate_batch(
        self,
        prompt: str,
        n_samples: int = 4,
        temperatures: Optional[List[float]] = None,
        **kwargs,
    ) -> List[Tuple[str, CognitiveTrace]]:
        """Generate multiple responses for the same prompt."""
        if temperatures is None:
            base_t = kwargs.pop("temperature", 0.7)
            spread = 0.15
            temperatures = [
                max(0.1, base_t + (i - n_samples / 2) * spread / n_samples)
                for i in range(n_samples)
            ]

        results: List[Tuple[str, CognitiveTrace]] = []
        for i, temp in enumerate(temperatures):
            logger.info(f"[Generator] Sample {i+1}/{n_samples} (temp={temp:.2f})")
            text, trace = self.generate(prompt, temperature=temp, **kwargs)
            results.append((text, trace))

        return results
