"""
METIS Pipeline — GRPO Online Training Loop

Bridges METIS cognitive rewards with TRL's GRPOTrainer for online
reinforcement learning. The model generates → METIS scores → GRPO updates,
all in one loop with zero external dependencies.

Core innovation:
    METIS cognitive rewards are computed via teacher-forcing on the model's
    own completions. This means the reward reflects the model's CURRENT
    uncertainty landscape, creating a natural curriculum:
    - Early training: high entropy → large reward gradients
    - Late training: calibrated entropy → fine-grained optimization

Designed for:
    - DGX Spark (128GB unified memory): 70B models, no reload tricks
    - RTX 4060 (8GB): small models with gradient checkpointing
    - Multi-GPU: compatible with accelerate/DeepSpeed via TRL

Usage:
    python -m metis.pipeline.online_loop \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --dataset trl-lib/DeepMath-103K \\
        --output ./grpo_output
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

logger = logging.getLogger("metis.online")


# ─────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────

@dataclass
class OnlineConfig:
    """Configuration for GRPO online training with METIS rewards."""
    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str = "auto"

    # GRPO
    num_generations: int = 4            # Completions per prompt (G in GRPO)
    max_completion_length: int = 512    # Max tokens per completion
    temperature: float = 0.7

    # Training
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-7
    max_grad_norm: float = 0.1
    warmup_ratio: float = 0.1

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # METIS reward
    metis_stride: int = 4               # METIS analysis every N tokens
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "coherence": 0.20,
        "calibration": 0.25,
        "phase_quality": 0.20,
        "epistemic_honesty": 0.20,
        "efficiency": 0.15,
    })

    # Output
    output_dir: str = "./grpo_output"
    logging_steps: int = 1
    save_steps: int = 100
    report_to: str = "none"             # "wandb" | "tensorboard" | "none"

    # Dataset
    dataset_name: str = ""              # HuggingFace dataset name
    dataset_split: str = "train"
    max_samples: int = 0                # 0 = use all
    prompt_column: str = "prompt"       # Column name for prompts


# ─────────────────────────────────────────────────────
# METIS Cognitive Reward Function (TRL-compatible)
# ─────────────────────────────────────────────────────

class MetisCognitiveRewardFn:
    """
    TRL-compatible reward function powered by METIS cognitive signals.

    For each completion, performs teacher-forcing to extract logits,
    feeds them through METIS to build a cognitive trace, then computes
    the multi-component cognitive reward.

    This is the core bridge between TRL's GRPOTrainer and METIS.

    TRL reward function signature:
        def reward_func(completions, prompts, **kwargs) -> list[float]
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        metis_stride: int = 4,
        device: Optional[str] = None,
    ):
        from metis.training.rewards import CognitiveRewardComputer

        self._model = model
        self._tokenizer = tokenizer
        self._stride = metis_stride
        self._device = device or str(next(model.parameters()).device)
        self._reward_computer = CognitiveRewardComputer()
        self._call_count = 0

    @torch.inference_mode()
    def __call__(
        self,
        completions: List[str],
        prompts: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[float]:
        """
        Compute METIS cognitive rewards for a batch of completions.

        Args:
            completions: List of generated completion strings
            prompts: List of corresponding prompt strings
            **kwargs: Additional TRL args (completion_ids, trainer_state, etc.)

        Returns:
            List of float rewards, one per completion
        """
        from metis import MetisInference
        from metis.core.types import CognitiveTrace

        self._call_count += 1
        rewards: List[float] = []

        # Initialize METIS for teacher-forcing
        metis = MetisInference(
            model=self._model,
            tokenizer=self._tokenizer,
        )

        for i, completion in enumerate(completions):
            prompt = prompts[i] if prompts and i < len(prompts) else ""

            try:
                # Build full text for teacher-forcing
                full_text = prompt + completion if prompt else completion

                # Tokenize
                input_ids = self._tokenizer.encode(
                    full_text, add_special_tokens=True, return_tensors="pt",
                )
                if isinstance(input_ids, list):
                    input_ids = torch.tensor([input_ids])
                input_ids = input_ids.to(self._device)

                prompt_ids = self._tokenizer.encode(
                    prompt, add_special_tokens=True,
                ) if prompt else []
                prompt_len = len(prompt_ids)

                # Teacher-forcing: forward pass to get logits
                outputs = self._model(input_ids=input_ids, use_cache=False)
                logits = outputs.logits[0]  # [seq_len, vocab_size]

                # Feed completion tokens through METIS stride-by-stride
                metis.reset()
                completion_len = input_ids.shape[1] - prompt_len

                for pos in range(prompt_len, input_ids.shape[1]):
                    token_id = input_ids[0, pos].item()
                    step_logits = logits[pos - 1] if pos > 0 else logits[0]

                    # Feed to METIS every stride tokens
                    if (pos - prompt_len) % self._stride == 0:
                        metis.step(
                            token_id=token_id,
                            logits=step_logits.unsqueeze(0),
                            position=pos,
                        )

                trace = metis.get_trace()
                reward_breakdown = self._reward_computer.compute(trace)
                rewards.append(reward_breakdown.total)

                del outputs, logits

            except Exception as e:
                logger.warning(f"[METIS Reward] Error on completion {i}: {e}")
                rewards.append(0.0)

        # Periodic VRAM cleanup
        if self._call_count % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self._call_count % 50 == 0:
            avg_r = sum(rewards) / max(len(rewards), 1)
            logger.info(
                f"[METIS Reward] call={self._call_count} "
                f"batch={len(rewards)} avg_reward={avg_r:+.4f}"
            )

        return rewards


# ─────────────────────────────────────────────────────
# Online Training Loop
# ─────────────────────────────────────────────────────

def run_online_grpo(config: OnlineConfig) -> None:
    """
    Run GRPO online training with METIS cognitive rewards.

    Pipeline:
        1. Load model + tokenizer
        2. Load dataset (prompts)
        3. Create METIS reward function
        4. Initialize TRL GRPOTrainer
        5. Train (generate → score → update in each step)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOTrainer, GRPOConfig
    from datasets import load_dataset, Dataset

    logger.info(f"{'='*60}")
    logger.info(f"METIS GRPO Online Training")
    logger.info(f"{'='*60}")
    logger.info(f"  Model:       {config.model_name}")
    logger.info(f"  Generations: {config.num_generations} per prompt")
    logger.info(f"  Max tokens:  {config.max_completion_length}")
    logger.info(f"  LoRA:        r={config.lora_r}" if config.use_lora else "  LoRA:        OFF (full finetune)")
    logger.info(f"  Output:      {config.output_dir}")

    # ─── Device setup ───
    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ─── Load model ───
    logger.info(f"Loading model: {config.model_name}")
    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.bfloat16
        # DGX Spark / large GPU: use device_map for multi-GPU
        if torch.cuda.device_count() > 1:
            model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, **model_kwargs,
    )
    if "device_map" not in model_kwargs:
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ─── Load dataset ───
    if config.dataset_name:
        logger.info(f"Loading dataset: {config.dataset_name}")
        dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    else:
        # Use built-in METIS training prompts
        from metis.pipeline.config import TRAIN_PROMPTS
        logger.info(f"Using built-in METIS prompts ({len(TRAIN_PROMPTS)} prompts)")
        dataset = Dataset.from_dict({
            config.prompt_column: TRAIN_PROMPTS,
        })

    if config.max_samples > 0:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))

    logger.info(f"Dataset: {len(dataset)} samples")

    # ─── Create METIS reward function ───
    logger.info("Initializing METIS cognitive reward function...")
    metis_reward = MetisCognitiveRewardFn(
        model=model,
        tokenizer=tokenizer,
        metis_stride=config.metis_stride,
        device=device,
    )

    # ─── LoRA config ───
    peft_config = None
    if config.use_lora:
        from peft import LoraConfig, TaskType
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )

    # ─── GRPOTrainer config ───
    grpo_config = GRPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_grad_norm=config.max_grad_norm,
        warmup_ratio=config.warmup_ratio,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        report_to=config.report_to,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
    )

    # ─── Initialize trainer ───
    logger.info("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=config.model_name,
        reward_funcs=metis_reward,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # ─── Train ───
    logger.info("Starting GRPO training...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # ─── Save ───
    trainer.save_model(config.output_dir)
    logger.info(f"Training complete in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    logger.info(f"Model saved to {config.output_dir}")

    # ─── Summary ───
    logger.info(f"{'='*60}")
    logger.info(f"METIS GRPO Training Summary")
    logger.info(f"{'='*60}")
    logger.info(f"  Total reward calls: {metis_reward._call_count}")
    logger.info(f"  Training time:      {elapsed:.0f}s")
    logger.info(f"  Output:             {config.output_dir}")


# ─────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="METIS GRPO Online Training — Cognitive RL without LLM-as-judge"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset", type=str, default="",
                        help="HuggingFace dataset name (empty = use METIS built-in prompts)")
    parser.add_argument("--output", type=str, default="./grpo_output")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--report-to", type=str, default="none",
                        choices=["none", "wandb", "tensorboard"])
    args = parser.parse_args()

    config = OnlineConfig(
        model_name=args.model,
        dataset_name=args.dataset,
        output_dir=args.output,
        num_generations=args.num_generations,
        max_completion_length=args.max_tokens,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        use_lora=not args.no_lora,
        max_samples=args.max_samples,
        report_to=args.report_to,
    )

    run_online_grpo(config)


if __name__ == "__main__":
    main()
