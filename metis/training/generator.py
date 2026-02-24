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

import gc
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

    @property
    def metis(self) -> Metis:
        """Expose METIS instance for listener registration (dashboard bridge, etc.)."""
        return self._metis

    @torch.inference_mode()
    def _prefill_prompt(self, prompt: str):
        """
        Pre-compute prompt KV cache (one-time cost per prompt).
        Returns (input_ids, attention_mask, past_key_values, last_logits).
        Caller clones past_key_values for each sample to avoid cross-contamination.
        """
        inputs = self._tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self._device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        return input_ids, attention_mask, outputs.past_key_values, outputs.logits[:, -1, :]

    @staticmethod
    def _clone_kv_cache(past_key_values):
        """Deep-clone KV cache tensors so each sample gets independent state."""
        if past_key_values is None:
            return None
        return tuple(
            tuple(t.clone() for t in layer)
            for layer in past_key_values
        )

    @torch.inference_mode()
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
        _prefilled: tuple = None,
    ) -> Tuple[str, CognitiveTrace]:
        """
        Generate text with strided METIS cognitive tracing.

        METIS step() runs every `metis_stride` tokens. For 200 tokens
        with stride=4 → 50 cognitive events, sufficient for rewards.

        Args:
            _prefilled: Optional (attention_mask, past_key_values, first_logits)
                        from _prefill_prompt() for KV cache reuse.
        """
        if _prefilled is not None:
            attention_mask, past_key_values, first_logits = _prefilled
        else:
            inputs = self._tokenizer(
                prompt, return_tensors="pt", add_special_tokens=True
            ).to(self._device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
            past_key_values = None
            first_logits = None

        self._metis.start_session(prompt)

        generated_ids: List[int] = []
        past_key_values = past_key_values  # May be pre-filled or None

        for step in range(max_new_tokens):
            if past_key_values is None:
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
            elif step == 0 and first_logits is not None:
                # Skip redundant prefill — use cached logits directly
                logits = first_logits
                # METIS + sampling handled below, skip model call
                do_metis = (step % metis_stride == 0)
                if do_metis:
                    self._metis.step(logits.clone())

                log_probs = F.log_softmax(logits.float(), dim=-1)
                next_logits = logits.clone().float()

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
                first_logits = None  # Only used for step 0
                continue
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
                self._metis.step(logits.clone())

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
        """Generate multiple responses for the same prompt.

        Optimization: pre-computes prompt KV cache once and reuses it
        for all N samples. Saves ~30% generation time for long prompts.
        """
        if temperatures is None:
            base_t = kwargs.pop("temperature", 0.7)
            spread = 0.15
            temperatures = [
                max(0.1, base_t + (i - n_samples / 2) * spread / n_samples)
                for i in range(n_samples)
            ]

        # Pre-fill prompt KV cache once (shared across all samples)
        _, attn_mask, prompt_kv, first_logits = self._prefill_prompt(prompt)

        results: List[Tuple[str, CognitiveTrace]] = []
        for i, temp in enumerate(temperatures):
            logger.info(f"[Generator] Sample {i+1}/{n_samples} (temp={temp:.2f})")
            # Clone KV cache so each sample evolves independently
            cloned_kv = self._clone_kv_cache(prompt_kv)
            prefilled = (attn_mask.clone(), cloned_kv, first_logits.clone())
            text, trace = self.generate(
                prompt, temperature=temp, _prefilled=prefilled, **kwargs
            )
            results.append((text, trace))

            # Sever cross-sample VRAM fragmentation accumulation
            del cloned_kv, prefilled
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Release shared prompt KV cache
        del prompt_kv, first_logits

        return results
