"""
METIS × GRPO Online Training Loop for DGX Spark

End-to-end self-improving LLM pipeline:
    1. vLLM generates G completions per prompt (fast batch decode)
    2. Teacher-forcing extracts METIS cognitive traces
    3. CognitiveRewardComputer scores each completion
    4. TRL GRPOTrainer optimizes policy via group-relative advantages

Hardware target: DGX Spark (128GB unified memory, Grace Blackwell)
    - 70B model in bf16: generation via vLLM, training via HF + LoRA
    - No quantization needed (128GB >> 140GB with memory overcommit)
    - vLLM uses 40% GPU for KV cache, rest for training

Usage:
    # On DGX Spark:
    python dgx/grpo_train.py \
        --model Qwen/Qwen2.5-72B-Instruct \
        --dataset metis_prompts.jsonl \
        --output grpo_output \
        --num-generations 8 \
        --epochs 3

    # Quick dev test (small model):
    python dgx/grpo_train.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --dataset metis_prompts.jsonl \
        --output grpo_test \
        --num-generations 4 \
        --lora-rank 16
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ── Project imports ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metis.metis import Metis
from metis.core.types import CognitiveTrace, ControllerConfig
from metis.training.rewards import CognitiveRewardComputer, RewardConfig, RewardBreakdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("grpo_train")


# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════

@dataclass
class GRPOTrainConfig:
    """Full training configuration for METIS × GRPO."""
    # Model
    model_name_or_path: str = "Qwen/Qwen2.5-72B-Instruct"
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"  # bf16 for DGX Spark, fp16 for consumer GPUs

    # LoRA (None = full fine-tuning, only feasible on DGX)
    lora_rank: Optional[int] = None  # None for full FT, 16-64 for LoRA
    lora_alpha: int = 32
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

    # Generation (vLLM)
    num_generations: int = 8        # G completions per prompt (GRPO group size)
    max_completion_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    vllm_gpu_utilization: float = 0.40  # Reserve 60% for training

    # Training
    epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-7
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01

    # GRPO-specific
    grpo_beta: float = 0.04         # KL penalty coefficient
    advantage_clip: float = 2.0     # Clip normalized advantages

    # METIS teacher-forcing
    metis_stride: int = 4           # METIS step() every N tokens (CPU stall reduction)

    # Dataset
    dataset_path: str = "metis_prompts.jsonl"
    max_prompt_length: int = 512

    # Output
    output_dir: str = "grpo_output"
    save_steps: int = 50
    logging_steps: int = 10

    # Reward config overrides
    reward_config: Optional[RewardConfig] = None


# ═══════════════════════════════════════════════════════════
# METIS Teacher-Forcing Reward Function
# ═══════════════════════════════════════════════════════════

class MetisTeacherForceReward:
    """
    Reward function that teacher-forces completions through the model
    with METIS hooks to extract cognitive traces, then computes rewards.

    This is the bridge between vLLM's fast generation and METIS's
    information-theoretic reward signals.

    Flow:
        completion_text → tokenize → forward pass → logits → METIS.step() → trace → reward
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        reward_config: Optional[RewardConfig] = None,
        metis_stride: int = 4,
        device: str = "cuda",
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._reward_computer = CognitiveRewardComputer(reward_config)
        self._metis_stride = metis_stride
        self._device = device

    @torch.inference_mode()
    def compute_rewards(
        self,
        prompts: List[str],
        completions: List[str],
    ) -> Tuple[List[float], List[RewardBreakdown]]:
        """
        Compute METIS cognitive rewards for a batch of (prompt, completion) pairs.

        Args:
            prompts: List of prompt strings
            completions: List of completion strings

        Returns:
            (reward_scalars, reward_breakdowns)
        """
        rewards: List[float] = []
        breakdowns: List[RewardBreakdown] = []

        for prompt, completion in zip(prompts, completions):
            try:
                trace = self._teacher_force_single(prompt, completion)
                breakdown = self._reward_computer.compute(trace)
                rewards.append(breakdown.total)
                breakdowns.append(breakdown)
            except Exception as e:
                logger.warning(f"Teacher-force failed: {e}")
                rewards.append(0.0)
                breakdowns.append(RewardBreakdown())

        return rewards, breakdowns

    def _teacher_force_single(
        self, prompt: str, completion: str
    ) -> CognitiveTrace:
        """
        Teacher-force a single completion through the model with METIS.

        Runs a single forward pass on [prompt + completion] tokens,
        then feeds completion-region logits to METIS stride-by-stride.
        """
        # Tokenize
        chat = [{"role": "user", "content": prompt}]
        prompt_text = self._tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + completion

        prompt_ids = self._tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = self._tokenizer.encode(full_text, add_special_tokens=False)

        prompt_len = len(prompt_ids)
        completion_len = len(full_ids) - prompt_len

        if completion_len < 2:
            return CognitiveTrace(query=prompt)

        # Forward pass (single batch, full sequence)
        input_ids = torch.tensor([full_ids], device=self._device)

        outputs = self._model(input_ids)
        logits = outputs.logits[0]  # [seq_len, vocab]

        # Extract completion-region logits (shifted by 1 for next-token prediction)
        # logits[i] predicts token[i+1], so completion logits start at prompt_len-1
        comp_logits = logits[prompt_len - 1 : prompt_len - 1 + completion_len]

        # Run METIS on completion logits
        metis = Metis.attach(self._model, self._tokenizer)
        metis.start_session(prompt)

        stride = self._metis_stride
        for i in range(0, len(comp_logits), stride):
            step_logits = comp_logits[i].unsqueeze(0).unsqueeze(0)  # [1, 1, vocab]
            metis.step(step_logits)

        metis.introspect()
        trace = metis.get_trace()
        metis.end_session()

        return trace


# ═══════════════════════════════════════════════════════════
# vLLM Generation Engine
# ═══════════════════════════════════════════════════════════

class VLLMGenerationEngine:
    """
    Manages vLLM offline engine for fast batch generation.

    On DGX Spark, vLLM runs natively with the full model.
    Uses offline LLM class (not server) to avoid network overhead.
    """

    def __init__(
        self,
        model_name: str,
        gpu_memory_utilization: float = 0.40,
        max_model_len: int = 4096,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
    ):
        from vllm import LLM, SamplingParams
        self._SamplingParams = SamplingParams

        logger.info(f"Initializing vLLM engine: {model_name}")
        self._llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            trust_remote_code=True,
            enforce_eager=True,  # Avoid CUDA graph issues
        )
        logger.info("vLLM engine ready.")

    def generate_batch(
        self,
        prompts: List[str],
        n: int = 8,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[List[str]]:
        """
        Generate n completions per prompt.

        Args:
            prompts: List of formatted prompt strings
            n: Number of completions per prompt
            max_tokens: Maximum tokens per completion
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold

        Returns:
            List of lists: completions_per_prompt[i] has n strings
        """
        IM_END = "<" + "|im_end|" + ">"
        EOS = "<" + "|endoftext|" + ">"
        params = self._SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=[IM_END, EOS],
        )

        outputs = self._llm.generate(prompts, params)

        results: List[List[str]] = []
        for output in outputs:
            completions = [o.text.strip() for o in output.outputs]
            results.append(completions)

        return results

    def shutdown(self) -> None:
        """Release vLLM resources."""
        del self._llm


# ═══════════════════════════════════════════════════════════
# Dataset Loading
# ═══════════════════════════════════════════════════════════

def load_prompts(path: str, max_length: int = 512) -> List[str]:
    """
    Load training prompts from JSONL file.

    Expected format: one JSON object per line with a "prompt" field.
    Falls back to plain text lines if no JSON structure found.
    """
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("prompt", obj.get("text", obj.get("instruction", "")))
            except json.JSONDecodeError:
                text = line
            if text and len(text) <= max_length:
                prompts.append(text)

    logger.info(f"Loaded {len(prompts)} prompts from {path}")
    return prompts


def format_chat_prompts(prompts: List[str], tokenizer: Any) -> List[str]:
    """Format raw prompts into chat template strings for vLLM."""
    formatted: List[str] = []
    for p in prompts:
        chat = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        formatted.append(text)
    return formatted


# ═══════════════════════════════════════════════════════════
# GRPO Training Loop
# ═══════════════════════════════════════════════════════════

class MetisGRPOTrainer:
    """
    Full GRPO training loop with METIS cognitive rewards.

    Architecture:
        ┌──────────────────────────────────────────┐
        │  For each epoch:                         │
        │    For each batch of prompts:            │
        │      1. vLLM: generate G completions     │
        │      2. Teacher-force: extract traces    │
        │      3. Compute METIS rewards            │
        │      4. Compute GRPO advantages          │
        │      5. Policy gradient update           │
        └──────────────────────────────────────────┘

    On DGX Spark, step 1 and step 5 share the same GPU via
    vLLM's gpu_memory_utilization parameter.
    """

    def __init__(self, config: GRPOTrainConfig):
        self.config = config
        self._reward_computer = CognitiveRewardComputer(
            config.reward_config or RewardConfig()
        )

    def train(self) -> None:
        """Execute the full GRPO training pipeline."""
        cfg = self.config

        # ── 1. Load tokenizer ──
        from transformers import AutoTokenizer
        logger.info(f"Loading tokenizer: {cfg.model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name_or_path, trust_remote_code=cfg.trust_remote_code
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ── 2. Load prompts ──
        prompts = load_prompts(cfg.dataset_path, cfg.max_prompt_length)
        if not prompts:
            raise RuntimeError(f"No prompts loaded from {cfg.dataset_path}")

        formatted_prompts = format_chat_prompts(prompts, tokenizer)

        # ── 3. Initialize vLLM generation engine ──
        logger.info("Initializing vLLM generation engine...")
        vllm_engine = VLLMGenerationEngine(
            model_name=cfg.model_name_or_path,
            gpu_memory_utilization=cfg.vllm_gpu_utilization,
            dtype=cfg.torch_dtype,
        )

        # ── 4. Generate all completions (Phase A) ──
        logger.info(
            f"Generating {cfg.num_generations} completions per prompt "
            f"for {len(prompts)} prompts..."
        )
        t0 = time.time()
        all_completions = vllm_engine.generate_batch(
            formatted_prompts,
            n=cfg.num_generations,
            max_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )
        gen_time = time.time() - t0
        total_comps = sum(len(c) for c in all_completions)
        logger.info(
            f"Generated {total_comps} completions in {gen_time:.1f}s "
            f"({total_comps / gen_time:.1f} comp/s)"
        )

        # ── 5. Release vLLM, load HF model for training ──
        vllm_engine.shutdown()
        del vllm_engine

        logger.info(f"Loading HF model for training: {cfg.model_name_or_path}")
        from transformers import AutoModelForCausalLM

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_map.get(cfg.torch_dtype, torch.bfloat16)

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=cfg.trust_remote_code,
            device_map="auto",
        )

        # ── 6. Apply LoRA if configured ──
        if cfg.lora_rank is not None:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                target_modules=cfg.lora_target_modules.split(","),
                task_type="CAUSAL_LM",
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(
                f"LoRA applied: {trainable:,} trainable / {total_params:,} total "
                f"({100 * trainable / total_params:.2f}%)"
            )

        # ── 7. Teacher-force + compute rewards (Phase B) ──
        logger.info("Teacher-forcing for METIS traces...")
        reward_fn = MetisTeacherForceReward(
            model=model,
            tokenizer=tokenizer,
            reward_config=cfg.reward_config,
            metis_stride=cfg.metis_stride,
        )

        all_rewards: List[List[float]] = []
        all_breakdowns: List[List[RewardBreakdown]] = []

        for i, (prompt, completions) in enumerate(zip(prompts, all_completions)):
            prompts_repeated = [prompt] * len(completions)
            rewards, breakdowns = reward_fn.compute_rewards(
                prompts_repeated, completions
            )
            all_rewards.append(rewards)
            all_breakdowns.append(breakdowns)

            if (i + 1) % 10 == 0:
                avg_r = sum(rewards) / len(rewards)
                spread = max(rewards) - min(rewards)
                logger.info(
                    f"  [{i+1}/{len(prompts)}] avg_reward={avg_r:.4f} "
                    f"spread={spread:.4f}"
                )

        # ── 8. Compute GRPO advantages ──
        logger.info("Computing GRPO advantages...")
        grpo_data = self._compute_advantages(
            prompts, all_completions, all_rewards
        )

        # ── 9. Save GRPO data for TRL training ──
        os.makedirs(cfg.output_dir, exist_ok=True)
        grpo_path = os.path.join(cfg.output_dir, "grpo_data.jsonl")
        with open(grpo_path, "w", encoding="utf-8") as f:
            for item in grpo_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(grpo_data)} GRPO samples to {grpo_path}")

        # ── 10. Run TRL GRPOTrainer ──
        self._run_trl_grpo(model, tokenizer, grpo_data, cfg)

        logger.info("GRPO training complete!")

    def _compute_advantages(
        self,
        prompts: List[str],
        completions_per_prompt: List[List[str]],
        rewards_per_prompt: List[List[float]],
    ) -> List[Dict[str, Any]]:
        """Compute normalized GRPO advantages within each group."""
        clip = self.config.advantage_clip
        data: List[Dict[str, Any]] = []

        for prompt, completions, rewards in zip(
            prompts, completions_per_prompt, rewards_per_prompt
        ):
            mean_r = sum(rewards) / len(rewards)
            var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
            std_r = math.sqrt(var_r) if var_r > 1e-8 else 1.0

            for comp, reward in zip(completions, rewards):
                adv = (reward - mean_r) / std_r
                adv = max(-clip, min(clip, adv))
                data.append({
                    "prompt": prompt,
                    "completion": comp,
                    "reward": round(reward, 4),
                    "advantage": round(adv, 4),
                    "group_mean": round(mean_r, 4),
                    "group_std": round(std_r, 4),
                })

        # Sort by advantage descending for logging
        pos = sum(1 for d in data if d["advantage"] > 0)
        neg = sum(1 for d in data if d["advantage"] < 0)
        logger.info(
            f"GRPO advantages: {pos} positive, {neg} negative, "
            f"{len(data)} total"
        )
        return data

    def _run_trl_grpo(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        grpo_data: List[Dict[str, Any]],
        cfg: GRPOTrainConfig,
    ) -> None:
        """
        Run policy optimization using TRL.

        Falls back to manual GRPO loss if TRL GRPOTrainer is not available.
        """
        try:
            self._run_trl_native(model, tokenizer, grpo_data, cfg)
        except ImportError:
            logger.warning("TRL GRPOTrainer not available, using manual GRPO loss")
            self._run_manual_grpo(model, tokenizer, grpo_data, cfg)

    def _run_trl_native(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        grpo_data: List[Dict[str, Any]],
        cfg: GRPOTrainConfig,
    ) -> None:
        """Use TRL's DPOTrainer with advantage-weighted pairs."""
        from datasets import Dataset
        from trl import DPOTrainer, DPOConfig

        # Convert GRPO data to DPO pairs (positive vs negative advantages)
        dpo_rows: List[Dict[str, str]] = []
        # Group by prompt
        from collections import defaultdict
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in grpo_data:
            groups[item["prompt"]].append(item)

        for prompt, items in groups.items():
            items.sort(key=lambda x: x["advantage"], reverse=True)
            if len(items) < 2:
                continue
            # Best vs worst as DPO pair
            best = items[0]
            worst = items[-1]
            margin = best["reward"] - worst["reward"]
            if margin < 0.05:
                continue
            dpo_rows.append({
                "prompt": prompt,
                "chosen": best["completion"],
                "rejected": worst["completion"],
            })

        if not dpo_rows:
            logger.warning("No valid DPO pairs generated from GRPO data")
            return

        logger.info(f"Created {len(dpo_rows)} DPO pairs from GRPO advantages")
        dataset = Dataset.from_list(dpo_rows)

        training_args = DPOConfig(
            output_dir=os.path.join(cfg.output_dir, "dpo_checkpoint"),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.per_device_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            max_grad_norm=cfg.max_grad_norm,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            beta=cfg.grpo_beta,
            bf16=(cfg.torch_dtype == "bfloat16"),
            fp16=(cfg.torch_dtype == "float16"),
            logging_steps=cfg.logging_steps,
            save_steps=cfg.save_steps,
            gradient_checkpointing=False,  # DGX 128GB: no memory-compute tradeoff needed
            max_length=cfg.max_completion_tokens + cfg.max_prompt_length,
            max_prompt_length=cfg.max_prompt_length,
            remove_unused_columns=False,
        )

        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        logger.info("Starting DPO training from GRPO advantages...")
        trainer.train()
        trainer.save_model(os.path.join(cfg.output_dir, "final_model"))
        logger.info(f"Model saved to {cfg.output_dir}/final_model")

    def _run_manual_grpo(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        grpo_data: List[Dict[str, Any]],
        cfg: GRPOTrainConfig,
    ) -> None:
        """
        Manual GRPO policy gradient (no TRL dependency).

        Loss = -E[ A_i * log pi(completion_i | prompt_i) ]

        where A_i is the pre-computed normalized advantage.
        """
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        model.train()
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs * len(grpo_data) // cfg.gradient_accumulation_steps,
        )

        global_step = 0
        accum_loss = 0.0

        for epoch in range(cfg.epochs):
            logger.info(f"Epoch {epoch + 1}/{cfg.epochs}")

            for i, item in enumerate(grpo_data):
                advantage = item["advantage"]
                if abs(advantage) < 0.01:
                    continue  # Skip near-zero advantage samples

                # Tokenize prompt + completion
                chat = [{"role": "user", "content": item["prompt"]}]
                prompt_text = tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
                full_text = prompt_text + item["completion"]
                encoding = tokenizer(
                    full_text, return_tensors="pt", truncation=True,
                    max_length=cfg.max_completion_tokens + cfg.max_prompt_length,
                )
                input_ids = encoding["input_ids"].to(model.device)

                prompt_ids = tokenizer.encode(
                    prompt_text, add_special_tokens=False
                )
                prompt_len = len(prompt_ids)

                # Forward pass
                outputs = model(input_ids, labels=input_ids)
                # Extract completion-only loss
                shift_logits = outputs.logits[0, prompt_len - 1:-1, :]
                shift_labels = input_ids[0, prompt_len:]
                completion_len = min(len(shift_logits), len(shift_labels))

                if completion_len < 1:
                    continue

                log_probs = F.log_softmax(
                    shift_logits[:completion_len], dim=-1
                )
                token_log_probs = log_probs.gather(
                    1, shift_labels[:completion_len].unsqueeze(1)
                ).squeeze(1)
                mean_log_prob = token_log_probs.mean()

                # GRPO loss: -advantage * log_prob
                loss = -advantage * mean_log_prob
                loss = loss / cfg.gradient_accumulation_steps
                loss.backward()
                accum_loss += loss.item()

                if (i + 1) % cfg.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % cfg.logging_steps == 0:
                        avg_loss = accum_loss / cfg.logging_steps
                        lr = scheduler.get_last_lr()[0]
                        logger.info(
                            f"  step={global_step} loss={avg_loss:.4f} lr={lr:.2e}"
                        )
                        accum_loss = 0.0

                    if global_step % cfg.save_steps == 0:
                        ckpt_path = os.path.join(
                            cfg.output_dir, f"checkpoint-{global_step}"
                        )
                        model.save_pretrained(ckpt_path)
                        logger.info(f"  Checkpoint saved: {ckpt_path}")

        # Final save
        final_path = os.path.join(cfg.output_dir, "final_model")
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        logger.info(f"Training complete. Model saved to {final_path}")


# ═══════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════

def parse_args() -> GRPOTrainConfig:
    parser = argparse.ArgumentParser(description="METIS x GRPO Training on DGX Spark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--dataset", type=str, default="metis_prompts.jsonl")
    parser.add_argument("--output", type=str, default="grpo_output")
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--vllm-gpu-util", type=float, default=0.40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--bf16", type=str, default="true",
                        choices=["true", "false"])
    parser.add_argument("--metis-stride", type=int, default=4)
    args = parser.parse_args()

    return GRPOTrainConfig(
        model_name_or_path=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        num_generations=args.num_generations,
        epochs=args.epochs,
        learning_rate=args.lr,
        lora_rank=args.lora_rank,
        temperature=args.temperature,
        max_completion_tokens=args.max_tokens,
        vllm_gpu_utilization=args.vllm_gpu_util,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        torch_dtype="bfloat16" if args.bf16 == "true" else "float16",
        metis_stride=args.metis_stride,
    )


if __name__ == "__main__":
    config = parse_args()
    trainer = MetisGRPOTrainer(config)
    trainer.train()
