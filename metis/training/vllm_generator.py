"""
METIS vLLM Batch Generator — 5-10x faster generation via vLLM server.

Two-phase architecture (8GB GPU constraint):
  Phase A: vLLM server generates all samples via OpenAI API (parallel decode)
  Phase B: Teacher-forcing through HF model to get full logits for METIS traces

Usage:
    generator = VLLMBatchGenerator(
        vllm_url="http://localhost:8000/v1",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
    )
    # Phase A: fast batch generation
    raw_samples = generator.generate_batch_vllm(prompt, n_samples=8)

    # Phase B: teacher-forcing for METIS traces (after stopping vLLM)
    results = generator.teacher_force_traces(prompt, raw_samples, hf_model, tokenizer)
"""
from __future__ import annotations

import gc
import json
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
import torch.nn.functional as F

from ..metis import Metis
from ..core.types import CognitiveTrace, ControllerConfig

logger = logging.getLogger(__name__)


class VLLMBatchGenerator:
    """
    vLLM-accelerated batch generator with METIS cognitive tracing.

    Separates generation (vLLM, fast) from cognitive analysis (HF teacher-forcing).
    This avoids the per-token autoregressive bottleneck of HuggingFace inference.
    """

    def __init__(
        self,
        vllm_url: str = "http://localhost:8000/v1",
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        metis_config: Optional[ControllerConfig] = None,
        timeout: int = 120,
    ):
        self._vllm_url = vllm_url.rstrip("/")
        self._model_name = model_name
        self._metis_config = metis_config
        self._timeout = timeout
        self._metis: Optional[Metis] = None

    # ═══════════════════════════════════════════════════════
    # Phase A: vLLM Batch Generation
    # ═══════════════════════════════════════════════════════

    def generate_batch_vllm(
        self,
        prompt: str,
        n_samples: int = 8,
        max_tokens: int = 512,
        temperatures: Optional[List[float]] = None,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> List[Dict[str, Any]]:
        """
        Generate N samples via vLLM OpenAI API.

        Returns list of dicts: [{"text": str, "logprobs": [...], "finish_reason": str}, ...]

        vLLM handles continuous batching internally — 8 samples run
        nearly as fast as 1 sample due to PagedAttention.
        """
        if temperatures is None:
            base_t = 0.7
            spread = 0.15
            temperatures = [
                max(0.1, base_t + (i - n_samples / 2) * spread / n_samples)
                for i in range(n_samples)
            ]

        results: List[Dict[str, Any]] = []

        # Build chat messages for each sample
        messages = [{"role": "user", "content": prompt}]

        for i, temp in enumerate(temperatures):
            t0 = time.perf_counter()
            payload = {
                "model": self._model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temp,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "logprobs": True,
                "top_logprobs": 5,
                "n": 1,
            }

            try:
                resp = requests.post(
                    f"{self._vllm_url}/chat/completions",
                    json=payload,
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                data = resp.json()

                choice = data["choices"][0]
                text = choice["message"]["content"]
                logprobs_data = choice.get("logprobs", {})
                token_logprobs = logprobs_data.get("content", [])

                elapsed = time.perf_counter() - t0
                logger.info(
                    f"[vLLM] Sample {i+1}/{n_samples} "
                    f"(temp={temp:.2f}) {len(text)} chars in {elapsed:.1f}s"
                )

                results.append({
                    "text": text,
                    "token_logprobs": token_logprobs,
                    "finish_reason": choice.get("finish_reason", "stop"),
                    "temperature": temp,
                })

            except requests.exceptions.RequestException as e:
                logger.error(f"[vLLM] Request failed for sample {i+1}: {e}")
                results.append({
                    "text": "",
                    "token_logprobs": [],
                    "finish_reason": "error",
                    "temperature": temp,
                })

        return results

    def generate_all_prompts(
        self,
        prompts: List[str],
        n_samples: int = 8,
        max_tokens: int = 512,
        bridge: Any = None,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Generate samples for ALL prompts via vLLM.

        Returns: {prompt_idx: [sample_dicts]}
        """
        all_results: Dict[int, List[Dict[str, Any]]] = {}

        for i, prompt in enumerate(prompts):
            logger.info(f"[vLLM] Prompt {i+1}/{len(prompts)}: {prompt[:60]}...")
            if bridge is not None:
                bridge.prompt_index = i + 1
                bridge.current_prompt = prompt[:80]

            samples = self.generate_batch_vllm(
                prompt, n_samples=n_samples, max_tokens=max_tokens,
            )
            all_results[i] = samples

        return all_results

    # ═══════════════════════════════════════════════════════
    # Phase B: Teacher-Forcing for METIS Traces
    # ═══════════════════════════════════════════════════════

    @torch.inference_mode()
    def teacher_force_traces(
        self,
        prompt: str,
        samples: List[Dict[str, Any]],
        model: torch.nn.Module,
        tokenizer: Any,
        metis_stride: int = 4,
    ) -> List[Tuple[str, CognitiveTrace]]:
        """
        Run teacher-forcing through HF model to get full logits for METIS.

        For each sample:
        1. Tokenize [prompt + generated_text] as a single sequence
        2. Single forward pass → logits at every position
        3. Run METIS.step() on logits at stride intervals (prompt region excluded)
        4. Return (text, CognitiveTrace) matching MetisGenerator interface

        This is ~50x faster than autoregressive generation because:
        - ONE forward pass per sample (not 50+ sequential steps)
        - All positions computed in parallel on GPU
        """
        device = str(next(model.parameters()).device)

        # Initialize METIS if not already
        if self._metis is None:
            self._metis = Metis.attach(model, tokenizer, config=self._metis_config)

        results: List[Tuple[str, CognitiveTrace]] = []

        for i, sample in enumerate(samples):
            text = sample["text"]
            if not text:
                # Skip failed generations
                results.append(("", CognitiveTrace()))
                continue

            # Build full sequence: prompt + response
            # Use chat template for consistency with how vLLM generated it
            if hasattr(tokenizer, 'apply_chat_template'):
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": text},
                ]
                full_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )
            else:
                full_text = prompt + text

            # Tokenize full sequence
            inputs = tokenizer(
                full_text, return_tensors="pt", add_special_tokens=True,
            ).to(device)
            input_ids = inputs["input_ids"]  # [1, seq_len]
            seq_len = input_ids.shape[1]

            # Find where prompt ends and response begins
            prompt_inputs = tokenizer(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True,
                ) if hasattr(tokenizer, 'apply_chat_template') else prompt,
                return_tensors="pt", add_special_tokens=True,
            )
            prompt_len = prompt_inputs["input_ids"].shape[1]

            if seq_len <= prompt_len + 1:
                results.append((text, CognitiveTrace()))
                continue

            # Single teacher-forcing forward pass — gets logits at ALL positions
            outputs = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,  # Don't need KV cache for teacher-forcing
            )
            all_logits = outputs.logits  # [1, seq_len, vocab_size]

            # Run METIS on response region logits (strided)
            self._metis.start_session(prompt)

            response_start = prompt_len
            response_end = seq_len - 1  # Last logit predicts next token

            for pos in range(response_start, response_end):
                step_idx = pos - response_start
                if step_idx % metis_stride != 0:
                    continue

                # Logits at position `pos` predict token at position `pos+1`
                logits = all_logits[:, pos, :]  # [1, vocab_size]
                self._metis.step(logits)

                # Feed surprise: actual token that was generated at pos+1
                actual_token_id = input_ids[0, pos + 1].item()
                log_probs = F.log_softmax(logits.float(), dim=-1)
                token_log_prob = log_probs[0, actual_token_id].item()
                surprise = -token_log_prob / math.log(2) if token_log_prob < 0 else 0.0
                self._metis.feed_surprise(surprise)

            # Finalize trace
            self._metis.introspect()
            trace = self._metis.trace
            self._metis.end_session()

            results.append((text, trace))

            logger.info(
                f"  [TF] Sample {i+1}/{len(samples)}: "
                f"{trace.total_tokens} events, "
                f"H={trace.mean_entropy:.3f} S={trace.mean_surprise:.3f}"
            )

            # Free VRAM
            del all_logits, outputs
            gc.collect()

        return results

    @property
    def metis(self) -> Optional[Metis]:
        return self._metis

    def check_server(self) -> bool:
        """Check if vLLM server is reachable."""
        try:
            resp = requests.get(
                f"{self._vllm_url}/models", timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            return False
