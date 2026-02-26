"""
METIS Pipeline — Phase 1: Generate & Score

Generates K responses per prompt with METIS cognitive instrumentation,
computes 5-component cognitive rewards for each response.

Supports three generation modes:
  1. HuggingFace token-by-token (default)
  2. vLLM server + teacher-forcing
  3. vLLM pre-generated samples + teacher-forcing only
"""
from __future__ import annotations

import gc
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from metis.pipeline.config import ExperimentConfig, format_chat

logger = logging.getLogger("experiment")


def phase1_generate(
    config: ExperimentConfig,
    vllm_url: Optional[str] = None,
    vllm_data_path: Optional[str] = None,
) -> Tuple[List[Dict], Any, Any]:
    """
    Generate K responses per prompt with METIS instrumentation.
    Returns scored data + model + tokenizer for reuse.

    Modes:
      - Default: HuggingFace token-by-token generation
      - vllm_url: vLLM server + teacher-forcing
      - vllm_data_path: Load pre-generated vLLM samples + teacher-forcing only
    """
    from metis.pipeline.config import TRAIN_PROMPTS

    logger.info(f"{'='*60}")
    mode_str = "vLLM pre-gen" if vllm_data_path else ("vLLM 2-phase" if vllm_url else "HuggingFace")
    logger.info(f"PHASE 1: Generate & Score ({config.n_train_prompts} prompts × {config.n_samples_per_prompt} samples) [{mode_str}]")
    logger.info(f"{'='*60}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from metis.training.rewards import CognitiveRewardComputer

    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = TRAIN_PROMPTS[:config.n_train_prompts]
    reward_computer = CognitiveRewardComputer()

    # ── Dashboard bridge (optional) ──
    bridge = None
    try:
        from metis.bridge import SignalBridge
        bridge = SignalBridge(port=8765)
        bridge.total_prompts = config.n_train_prompts
        bridge.phase = "generate"
        bridge.start()
        logger.info("[Bridge] Dashboard bridge started on ws://0.0.0.0:8765")
    except Exception as e:
        logger.warning(f"[Bridge] Dashboard bridge unavailable: {e}")
        bridge = None

    # ═══════════════════════════════════════════════════════
    # vLLM Pre-generated Data Path (offline batch)
    # ═══════════════════════════════════════════════════════
    if vllm_data_path:
        model, all_data = _generate_vllm_pregen(
            config, device, tokenizer, prompts, reward_computer, bridge, vllm_data_path,
        )

    # ═══════════════════════════════════════════════════════
    # vLLM Two-Phase Path (server mode)
    # ═══════════════════════════════════════════════════════
    elif vllm_url:
        model, all_data = _generate_vllm_server(
            config, device, tokenizer, prompts, reward_computer, bridge, vllm_url,
        )

    # ═══════════════════════════════════════════════════════
    # Standard HuggingFace Path (original)
    # ═══════════════════════════════════════════════════════
    else:
        model, all_data = _generate_hf(
            config, device, tokenizer, prompts, reward_computer, bridge,
        )

    # ── Cleanup bridge ──
    if bridge is not None:
        try:
            bridge.stop()
        except Exception:
            pass

    # Save raw data
    os.makedirs(config.output_dir, exist_ok=True)
    data_path = os.path.join(config.output_dir, "phase1_scored_data.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(all_data)} scored samples to {data_path}")

    return all_data, model, tokenizer


# ─────────────────────────────────────────────────────
# Internal: HuggingFace generation path
# ─────────────────────────────────────────────────────

def _generate_hf(
    config: ExperimentConfig,
    device: str,
    tokenizer: Any,
    prompts: List[str],
    reward_computer: Any,
    bridge: Any,
) -> Tuple[Any, List[Dict]]:
    from transformers import AutoModelForCausalLM
    from metis.training.generator import MetisGenerator

    logger.info(f"Loading model: {config.model_name}")
    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, **model_kwargs
    ).to(device)
    model.eval()

    generator = MetisGenerator(model, tokenizer)

    if bridge is not None:
        generator.metis.add_listener(bridge.on_signal)

    all_data: List[Dict] = []

    for i, prompt in enumerate(prompts):
        logger.info(f"[{i+1}/{len(prompts)}] {prompt[:50]}...")

        if bridge is not None:
            bridge.prompt_index = i + 1
            bridge.current_prompt = prompt

        chat_prompt = format_chat(tokenizer, prompt)

        samples = generator.generate_batch(
            chat_prompt,
            n_samples=config.n_samples_per_prompt,
            max_new_tokens=config.max_new_tokens,
        )

        for j, (text, trace) in enumerate(samples):
            if bridge is not None:
                bridge.sample_index = j + 1
            if trace.total_tokens > 5 and trace.mean_entropy == 0.0:
                raise RuntimeError(
                    f"FATAL: trace.mean_entropy == 0.0 for prompt {i+1} sample {j}. "
                    f"METIS cognitive hook is bypassed — logits not reaching step(). "
                    f"Check generator.py introspect() call chain."
                )

            reward = reward_computer.compute(trace)
            if bridge is not None:
                bridge.push_reward(reward.to_dict(), j, text)
            entry = _build_entry(prompt, chat_prompt, text, j, reward, trace)
            all_data.append(entry)
            logger.info(
                f"  sample {j}: reward={reward.total:+.4f} "
                f"tokens={trace.total_tokens} "
                f"H={trace.mean_entropy:.3f} S={trace.mean_surprise:.3f} "
                f"resp={text[:50]}..."
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if bridge is not None:
        try:
            generator.metis.remove_listener(bridge.on_signal)
        except Exception:
            pass

    return model, all_data


# ─────────────────────────────────────────────────────
# Internal: vLLM pre-generated path
# ─────────────────────────────────────────────────────

def _generate_vllm_pregen(
    config: ExperimentConfig,
    device: str,
    tokenizer: Any,
    prompts: List[str],
    reward_computer: Any,
    bridge: Any,
    vllm_data_path: str,
) -> Tuple[Any, List[Dict]]:
    from transformers import AutoModelForCausalLM
    from metis.training.vllm_generator import VLLMBatchGenerator

    logger.info(f"[vLLM] Loading pre-generated samples from {vllm_data_path}")
    with open(vllm_data_path, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)
    logger.info(f"[vLLM] Loaded {len(raw_samples)} samples")

    # Group by prompt_idx
    by_prompt: Dict[int, List[Dict]] = {}
    for s in raw_samples:
        idx = s["prompt_idx"]
        if idx not in by_prompt:
            by_prompt[idx] = []
        by_prompt[idx].append({
            "text": s["text"],
            "token_logprobs": [],
            "finish_reason": s.get("finish_reason", "stop"),
            "temperature": s.get("temperature", 0.7),
        })

    # Load HF model for teacher-forcing
    logger.info("[vLLM] Loading HF model for teacher-forcing...")
    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, **model_kwargs
    ).to(device)
    model.eval()

    vllm_gen = VLLMBatchGenerator(model_name=config.model_name)

    if bridge is not None:
        bridge.phase = "teacher-force"

    all_data: List[Dict] = []
    t0 = time.time()

    for i, prompt in enumerate(prompts):
        if i not in by_prompt:
            logger.warning(f"[vLLM] No samples for prompt {i}, skipping")
            continue
        logger.info(f"[TF] [{i+1}/{len(prompts)}] {prompt[:50]}...")
        if bridge is not None:
            bridge.prompt_index = i + 1
            bridge.current_prompt = prompt

        samples_for_prompt = by_prompt[i]
        results = vllm_gen.teacher_force_traces(
            prompt, samples_for_prompt, model, tokenizer,
        )

        chat_prompt = format_chat(tokenizer, prompt)
        for j, (text, trace) in enumerate(results):
            if bridge is not None:
                bridge.sample_index = j + 1
            if not text or trace.total_tokens == 0:
                continue
            if trace.total_tokens > 5 and trace.mean_entropy == 0.0:
                logger.warning(f"[TF] Skipping prompt {i+1} sample {j}: mean_entropy==0")
                continue

            reward = reward_computer.compute(trace)
            if bridge is not None:
                bridge.push_reward(reward.to_dict(), j, text)
            entry = _build_entry(prompt, chat_prompt, text, j, reward, trace)
            all_data.append(entry)
            logger.info(
                f"  sample {j}: reward={reward.total:+.4f} "
                f"tokens={trace.total_tokens} "
                f"H={trace.mean_entropy:.3f} S={trace.mean_surprise:.3f}"
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tf_time = time.time() - t0
    logger.info(f"[vLLM] Teacher-forcing complete: {tf_time:.0f}s ({tf_time/60:.1f}m)")

    return model, all_data


# ─────────────────────────────────────────────────────
# Internal: vLLM server path
# ─────────────────────────────────────────────────────

def _generate_vllm_server(
    config: ExperimentConfig,
    device: str,
    tokenizer: Any,
    prompts: List[str],
    reward_computer: Any,
    bridge: Any,
    vllm_url: str,
) -> Tuple[Any, List[Dict]]:
    from transformers import AutoModelForCausalLM
    from metis.training.vllm_generator import VLLMBatchGenerator

    vllm_gen = VLLMBatchGenerator(
        vllm_url=vllm_url,
        model_name=config.model_name,
    )

    if not vllm_gen.check_server():
        raise RuntimeError(
            f"vLLM server not reachable at {vllm_url}. "
            f"Start it with: wsl -e bash vllm_serve.sh"
        )
    logger.info(f"[vLLM] Server connected: {vllm_url}")

    # Phase A: Batch generate all samples via vLLM
    logger.info(f"[vLLM] Phase A: Generating {config.n_train_prompts * config.n_samples_per_prompt} samples...")
    t0 = time.time()
    all_raw = vllm_gen.generate_all_prompts(
        prompts,
        n_samples=config.n_samples_per_prompt,
        max_tokens=config.max_new_tokens,
        bridge=bridge,
    )
    gen_time = time.time() - t0
    logger.info(f"[vLLM] Phase A complete: {gen_time:.0f}s ({gen_time/60:.1f}m)")

    # Phase B: Load HF model for teacher-forcing METIS traces
    logger.info("[vLLM] Phase B: Loading HF model for teacher-forcing...")
    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, **model_kwargs
    ).to(device)
    model.eval()

    # Register bridge to METIS if available
    if bridge is not None and vllm_gen.metis is not None:
        vllm_gen.metis.add_listener(bridge.on_signal)

    all_data: List[Dict] = []
    t1 = time.time()

    for i, prompt in enumerate(prompts):
        logger.info(f"[TF] [{i+1}/{len(prompts)}] {prompt[:50]}...")
        if bridge is not None:
            bridge.prompt_index = i + 1
            bridge.current_prompt = prompt
            bridge.phase = "teacher-force"

        raw_samples = all_raw.get(i, [])
        results = vllm_gen.teacher_force_traces(
            prompt, raw_samples, model, tokenizer,
        )

        chat_prompt = format_chat(tokenizer, prompt)
        for j, (text, trace) in enumerate(results):
            if bridge is not None:
                bridge.sample_index = j + 1

            if not text or trace.total_tokens == 0:
                continue

            if trace.total_tokens > 5 and trace.mean_entropy == 0.0:
                logger.warning(
                    f"[TF] Skipping prompt {i+1} sample {j}: mean_entropy==0"
                )
                continue

            reward = reward_computer.compute(trace)
            if bridge is not None:
                bridge.push_reward(reward.to_dict(), j, text)
            entry = _build_entry(prompt, chat_prompt, text, j, reward, trace)
            all_data.append(entry)
            logger.info(
                f"  sample {j}: reward={reward.total:+.4f} "
                f"tokens={trace.total_tokens} "
                f"H={trace.mean_entropy:.3f} S={trace.mean_surprise:.3f}"
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tf_time = time.time() - t1
    logger.info(
        f"[vLLM] Phase B complete: {tf_time:.0f}s ({tf_time/60:.1f}m). "
        f"Total: {gen_time + tf_time:.0f}s"
    )

    return model, all_data


# ─────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────

def _build_entry(
    prompt: str,
    chat_prompt: str,
    text: str,
    sample_idx: int,
    reward: Any,
    trace: Any,
) -> Dict:
    """Build a scored data entry from generation results."""
    return {
        "prompt": prompt,
        "chat_prompt": chat_prompt,
        "response": text,
        "sample_idx": sample_idx,
        "reward_total": reward.total,
        "reward_breakdown": reward.to_dict(),
        "trace_stats": {
            "total_tokens": trace.total_tokens,
            "mean_entropy": trace.mean_entropy,
            "mean_surprise": trace.mean_surprise,
            "fast_ratio": trace.fast_count / max(trace.total_tokens, 1),
            "deep_ratio": trace.deep_count / max(trace.total_tokens, 1),
        },
    }
