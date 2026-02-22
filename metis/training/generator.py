"""
METIS Lightweight Generator — Token-by-token generation with cognitive tracing

Pure transformers implementation (no vllm dependency).
Each token's logits pass through Metis.step() to collect CognitiveTrace,
which is then used by CognitiveRewardComputer for training rewards.

This is the bridge between model inference and training pipeline:
    Model.forward() → logits → Metis.step() → CognitiveSignal
                    → sample token → feed_surprise() → next token
                    → CognitiveTrace → CognitiveReward → DPO/GRPO

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
    """
    Lightweight METIS-instrumented text generator.

    Generates text token-by-token using a HuggingFace model,
    running Metis.step() on each token's logits to collect
    full CognitiveTrace for reward computation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        metis_config: Optional[ControllerConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            model: HuggingFace model (AutoModelForCausalLM)
            tokenizer: HuggingFace tokenizer
            metis_config: Optional METIS controller configuration
            device: Device string (auto-detected if None)
        """
        self._model = model
        self._tokenizer = tokenizer
        self._device = device or str(next(model.parameters()).device)

        # Initialize METIS core
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
    ) -> Tuple[str, CognitiveTrace]:
        """
        Generate text with full METIS cognitive tracing.

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty factor
            do_sample: Whether to sample (False = greedy)

        Returns:
            (generated_text, CognitiveTrace) tuple
        """
        # Encode prompt
        inputs = self._tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self._device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

        # Start METIS session
        self._metis.start_session(prompt)

        generated_ids: List[int] = []
        past_key_values = None

        for step in range(max_new_tokens):
            # Forward pass
            if past_key_values is None:
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
            else:
                # Incremental decoding: only pass last token
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

            # ─── METIS cognitive step ───
            signal = self._metis.step(logits)

            # ─── Token surprise computation ───
            log_probs = F.log_softmax(logits.float(), dim=-1)

            # ─── Sampling ───
            next_logits = logits.clone().float()

            # Repetition penalty
            if repetition_penalty != 1.0 and generated_ids:
                for prev_id in set(generated_ids):
                    if next_logits[0, prev_id] > 0:
                        next_logits[0, prev_id] /= repetition_penalty
                    else:
                        next_logits[0, prev_id] *= repetition_penalty

            if do_sample and temperature > 0:
                # Temperature scaling
                next_logits = next_logits / temperature

                # Top-k filtering
                if top_k > 0:
                    topk_vals, _ = torch.topk(next_logits, top_k, dim=-1)
                    min_topk = topk_vals[:, -1].unsqueeze(-1)
                    next_logits[next_logits < min_topk] = float("-inf")

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_logits, descending=True, dim=-1
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    # Remove tokens with cumulative probability above threshold
                    sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                    sorted_logits[sorted_mask] = float("-inf")
                    # Scatter back
                    next_logits.scatter_(1, sorted_indices, sorted_logits)

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = next_logits.argmax(dim=-1)

            token_id = next_token.item()

            # ─── Feed surprise back to METIS ───
            token_log_prob = log_probs[0, token_id].item()
            token_surprise = -token_log_prob / math.log(2) if token_log_prob < 0 else 0.0
            self._metis.feed_surprise(token_surprise)

            generated_ids.append(token_id)

            # Check EOS
            if token_id == self._tokenizer.eos_token_id:
                break

        # Explicitly release GPU tensors before decode
        del past_key_values, logits, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Decode
        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Get trace
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
        """
        Generate multiple responses for the same prompt.

        Used for GRPO: generate N samples, rank by cognitive reward.

        Args:
            prompt: Input prompt
            n_samples: Number of samples to generate
            temperatures: Optional list of temperatures (one per sample).
                         If None, spreads around kwargs['temperature'] or 0.7.
            **kwargs: Additional args passed to generate()

        Returns:
            List of (text, CognitiveTrace) tuples
        """
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
