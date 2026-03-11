#!/usr/bin/env python3
"""
METIS × GRPO Math Training — Ground-Truth Outcome RL on GSM8K

Architecture (DeepSeek-R1 paradigm, adapted for GB10 single-card):
  Phase A: Generate N=8 solutions per math problem using SFT model
           Model generates freely within <thinking> framework
  Phase B: Outcome-based reward:
           - Correct answer (exact match) → R_outcome = +1.0
           - Wrong answer → R_outcome = -1.0
           - METIS cognitive reward as bonus signal (weighted 0.3)
           - R_total = 0.7 * R_outcome + 0.3 * R_cognitive
  Phase C: GRPO advantage computation (group-relative normalization)
  Phase D: Policy gradient optimization (LoRA, manual GRPO loss)

Key insight: DPO cannot inject reasoning — only RL with absolute
truth verification can force the model to EXPLORE and discover
genuine mathematical reasoning patterns.

Hardware: GB10 (122GB unified memory)
  - Sequential phases (no vLLM — single HF model throughout)
  - LoRA r=64 for memory efficiency
  - Gradient checkpointing enabled

Usage:
  python tools/grpo_math_train.py \
    --sft-model experiment_output_7B_restructured/metis_sft_base \
    --output experiment_output_7B_grpo_math \
    --n-samples 8 --max-prompts 1000 --epochs 2
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metis.training.generator import MetisGenerator
from metis.training.rewards import (
    CognitiveRewardComputer,
    RewardConfig,
    RewardBreakdown,
    extract_final_answer,
)
from metis.training.tokenizer_utils import register_metis_special_tokens
from metis.pipeline.config import format_chat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# Math Answer Extraction & Verification
# ═══════════════════════════════════════════════════════════

def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract numeric answer from GSM8K format: '#### 72' → '72'"""
    match = re.search(r'####\s*(.+)', answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    return ""


def extract_model_number(text: str) -> Optional[str]:
    """Extract the final numeric answer from model generation.

    Prioritized extraction:
      1. FINAL ANSWER: <number>
      2. The answer is <number>
      3. #### <number>
      4. Last standalone number in text (after stripping thinking blocks)
    """
    cleaned = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL).strip()
    if not cleaned:
        return None

    # Tier 1: FINAL ANSWER marker
    m = re.search(r'FINAL\s*ANSWER\s*[:：]\s*([-\d,.\s/]+)', cleaned, re.IGNORECASE)
    if m:
        return _normalize_number(m.group(1))

    # Tier 2: "the answer is" pattern
    m = re.search(r'(?:the\s+)?answer\s+is\s*[:：]?\s*([-\d,.\s/]+)', cleaned, re.IGNORECASE)
    if m:
        return _normalize_number(m.group(1))

    # Tier 3: GSM8K-style #### marker
    m = re.search(r'####\s*([-\d,.\s]+)', cleaned)
    if m:
        return _normalize_number(m.group(1))

    # Tier 4: Boxed answer (LaTeX)
    m = re.search(r'\\boxed\{([^}]+)\}', cleaned)
    if m:
        return _normalize_number(m.group(1))

    # Tier 5: Last number in the last sentence
    numbers = re.findall(r'(?<!\w)([-]?\d[\d,]*\.?\d*)', cleaned)
    if numbers:
        return _normalize_number(numbers[-1])

    return None


def _normalize_number(s: str) -> str:
    """Normalize number string: remove commas, whitespace, trailing dots."""
    s = s.strip().replace(",", "").replace(" ", "")
    s = s.rstrip(".")
    # Try to convert to float then back to canonical form
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return s


def verify_math_answer(model_answer: Optional[str], ground_truth: str) -> bool:
    """Exact match verification for math answers.

    Handles: integers, decimals, negatives, fractions.
    """
    if model_answer is None:
        return False

    gt = _normalize_number(ground_truth)
    pred = _normalize_number(model_answer)

    if gt == pred:
        return True

    # Float comparison with tolerance
    try:
        gt_val = float(gt)
        pred_val = float(pred)
        if abs(gt_val - pred_val) < 1e-6:
            return True
        # Relative tolerance for large numbers
        if abs(gt_val) > 1 and abs((gt_val - pred_val) / gt_val) < 1e-4:
            return True
    except ValueError:
        pass

    return False


# ═══════════════════════════════════════════════════════════
# GSM8K Dataset Loading
# ═══════════════════════════════════════════════════════════

@dataclass
class MathProblem:
    """A math problem with ground-truth answer."""
    question: str
    ground_truth: str  # Normalized numeric answer
    full_solution: str  # GSM8K chain-of-thought solution


def load_gsm8k(max_problems: int = 1000) -> List[MathProblem]:
    """Load GSM8K training set."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")

    problems: List[MathProblem] = []
    for ex in ds:
        gt = extract_gsm8k_answer(ex["answer"])
        if not gt:
            continue
        problems.append(MathProblem(
            question=ex["question"],
            ground_truth=gt,
            full_solution=ex["answer"],
        ))
        if len(problems) >= max_problems:
            break

    logger.info(f"Loaded {len(problems)} GSM8K problems")
    return problems


# ═══════════════════════════════════════════════════════════
# METIS Math Prompt Format
# ═══════════════════════════════════════════════════════════

_MATH_SYSTEM_PROMPT = (
    "You are a precise mathematical reasoner. "
    "Think step-by-step inside <thinking> tags, then provide your final numeric answer.\n"
    "Format your final answer as: FINAL ANSWER: <number>"
)


def format_math_prompt(question: str) -> str:
    """Format a math question for METIS-style generation."""
    return f"{_MATH_SYSTEM_PROMPT}\n\nProblem: {question}"


# ═══════════════════════════════════════════════════════════
# Scored Sample
# ═══════════════════════════════════════════════════════════

@dataclass
class MathSample:
    """A single generated math solution with all scores."""
    text: str
    extracted_answer: Optional[str]
    is_correct: bool
    r_outcome: float        # +1.0 if correct, -1.0 if wrong
    r_cognitive: float      # METIS cognitive reward
    r_total: float           # Combined reward
    n_tokens: int


# ═══════════════════════════════════════════════════════════
# Phase A: Generation
# ═══════════════════════════════════════════════════════════

def generate_solutions(
    generator: MetisGenerator,
    reward_computer: CognitiveRewardComputer,
    tokenizer: Any,
    problem: MathProblem,
    n_samples: int = 8,
    max_new_tokens: int = 512,
    outcome_weight: float = 0.7,
    cognitive_weight: float = 0.3,
) -> List[MathSample]:
    """Generate N solutions for a math problem, score with outcome + METIS reward."""
    prompt = format_math_prompt(problem.question)
    chat_prompt = format_chat(tokenizer, prompt)

    # Diverse temperature rollout
    temps = [0.5, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9, 1.0][:n_samples]

    try:
        results = generator.generate_batch(
            chat_prompt,
            n_samples=n_samples,
            temperatures=temps,
            max_new_tokens=max_new_tokens,
        )
    except Exception as e:
        logger.warning(f"Generation failed: {e}")
        return []

    samples: List[MathSample] = []
    for text, trace in results:
        # METIS cognitive reward
        reward = reward_computer.compute(trace)
        r_cognitive = reward.total

        # Outcome-based reward: absolute truth verification
        extracted = extract_model_number(text)
        is_correct = verify_math_answer(extracted, problem.ground_truth)
        r_outcome = 1.0 if is_correct else -1.0

        # Combined reward
        r_total = outcome_weight * r_outcome + cognitive_weight * r_cognitive

        samples.append(MathSample(
            text=text,
            extracted_answer=extracted,
            is_correct=is_correct,
            r_outcome=r_outcome,
            r_cognitive=r_cognitive,
            r_total=r_total,
            n_tokens=len(trace.events) if trace.events else 0,
        ))

    return samples


# ═══════════════════════════════════════════════════════════
# Phase B: GRPO Advantage Computation
# ═══════════════════════════════════════════════════════════

@dataclass
class GRPOItem:
    """A single GRPO training item with advantage."""
    prompt: str
    completion: str
    reward: float
    advantage: float
    is_correct: bool
    group_mean: float
    group_std: float


def compute_advantages(
    prompt: str,
    samples: List[MathSample],
    clip: float = 2.0,
) -> List[GRPOItem]:
    """Compute group-relative advantages for GRPO.

    GRPO normalizes rewards within each group (per-prompt):
      A_i = (R_i - mean(R)) / std(R)
    Then clips to [-clip, clip] for training stability.
    """
    if len(samples) < 2:
        return []

    rewards = [s.r_total for s in samples]
    mean_r = sum(rewards) / len(rewards)
    var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
    std_r = math.sqrt(var_r) if var_r > 1e-8 else 1.0

    items: List[GRPOItem] = []
    for s in samples:
        adv = (s.r_total - mean_r) / std_r
        adv = max(-clip, min(clip, adv))
        items.append(GRPOItem(
            prompt=prompt,
            completion=s.text,
            reward=s.r_total,
            advantage=adv,
            is_correct=s.is_correct,
            group_mean=mean_r,
            group_std=std_r,
        ))

    return items


# ═══════════════════════════════════════════════════════════
# Phase C: GRPO Training (Manual Loss)
# ═══════════════════════════════════════════════════════════

def run_grpo_training(
    model: torch.nn.Module,
    tokenizer: Any,
    grpo_items: List[GRPOItem],
    output_dir: str,
    epochs: int = 2,
    batch_size: int = 2,
    grad_accum: int = 8,
    learning_rate: float = 5e-7,
    max_grad_norm: float = 1.0,
    max_length: int = 1536,
    save_steps: int = 100,
    logging_steps: int = 10,
) -> None:
    """Manual GRPO policy gradient training.

    Loss = -E[ A_i * log π(completion_i | prompt_i) ]

    On GB10: batch=2, grad_accum=8, gc=True → ~57GB VRAM
    """
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    # Filter near-zero advantage items (no learning signal)
    active_items = [it for it in grpo_items if abs(it.advantage) > 0.1]
    logger.info(f"Training on {len(active_items)}/{len(grpo_items)} items "
                f"(filtered {len(grpo_items) - len(active_items)} near-zero advantage)")

    if not active_items:
        logger.warning("No items with sufficient advantage — skipping training")
        return

    model.train()
    # Enable gradient checkpointing for memory safety
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
    )
    total_steps = epochs * len(active_items) // grad_accum
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    os.makedirs(output_dir, exist_ok=True)
    global_step = 0
    accum_loss = 0.0
    accum_count = 0

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs} ({len(active_items)} items)")
        import random
        random.shuffle(active_items)

        for i, item in enumerate(active_items):
            # Tokenize prompt + completion
            chat_prompt = format_chat(tokenizer, item.prompt)
            full_text = chat_prompt + item.completion
            encoding = tokenizer(
                full_text, return_tensors="pt", truncation=True,
                max_length=max_length,
            )
            input_ids = encoding["input_ids"].to(next(model.parameters()).device)

            prompt_ids = tokenizer.encode(chat_prompt, add_special_tokens=False)
            prompt_len = len(prompt_ids)

            if input_ids.shape[1] <= prompt_len + 1:
                continue

            # Forward pass
            outputs = model(input_ids)

            # Extract completion-only log-probs
            shift_logits = outputs.logits[0, prompt_len - 1:-1, :]
            shift_labels = input_ids[0, prompt_len:]
            completion_len = min(len(shift_logits), len(shift_labels))

            if completion_len < 1:
                continue

            log_probs = F.log_softmax(shift_logits[:completion_len].float(), dim=-1)
            token_log_probs = log_probs.gather(
                1, shift_labels[:completion_len].unsqueeze(1)
            ).squeeze(1)
            mean_log_prob = token_log_probs.mean()

            # GRPO loss: -advantage * log_prob
            loss = -item.advantage * mean_log_prob
            loss = loss / grad_accum
            loss.backward()
            accum_loss += loss.item()
            accum_count += 1

            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % logging_steps == 0:
                    avg_loss = accum_loss / max(accum_count, 1)
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"  step={global_step} loss={avg_loss:.4f} lr={lr:.2e}"
                    )
                    accum_loss = 0.0
                    accum_count = 0

                if global_step % save_steps == 0:
                    ckpt = os.path.join(output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
                    logger.info(f"  Saved checkpoint: {ckpt}")

            # VRAM hygiene
            del outputs, loss, log_probs
            if (i + 1) % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Final save
    final_path = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Training complete. Model saved to {final_path}")


# ═══════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="METIS × GRPO Math Training (GSM8K)")
    parser.add_argument("--sft-model", default="experiment_output_7B_restructured/metis_sft_base")
    parser.add_argument("--output", default="experiment_output_7B_grpo_math")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--max-prompts", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--outcome-weight", type=float, default=0.7,
                        help="Weight for outcome reward (correct=+1, wrong=-1)")
    parser.add_argument("--cognitive-weight", type=float, default=0.3,
                        help="Weight for METIS cognitive reward")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--resume-from", type=int, default=0)
    parser.add_argument("--phase", choices=["all", "generate", "train"], default="all",
                        help="Run specific phase only")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("METIS × GRPO: Math Reasoning RL (GSM8K)")
    logger.info("=" * 60)
    logger.info(f"  SFT Model:       {args.sft_model}")
    logger.info(f"  Output:          {args.output}")
    logger.info(f"  Samples/Problem: {args.n_samples}")
    logger.info(f"  Max Problems:    {args.max_prompts}")
    logger.info(f"  Reward Mix:      {args.outcome_weight:.0%} outcome + {args.cognitive_weight:.0%} cognitive")
    logger.info(f"  LoRA Rank:       {args.lora_rank}")
    logger.info(f"  Epochs:          {args.epochs}")

    os.makedirs(args.output, exist_ok=True)
    grpo_data_path = os.path.join(args.output, "grpo_data.json")

    # ══════════════════════════════════════════════
    # PHASE A: Generate + Score
    # ══════════════════════════════════════════════
    if args.phase in ("all", "generate"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading SFT model for generation...")
        tokenizer = AutoTokenizer.from_pretrained(args.sft_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.sft_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        tokenizer, model = register_metis_special_tokens(tokenizer, model)
        device = str(next(model.parameters()).device)
        logger.info(f"  Device: {device}")

        # Load GSM8K
        problems = load_gsm8k(args.max_prompts)

        # Initialize METIS generator + reward computer
        generator = MetisGenerator(model, tokenizer)
        reward_computer = CognitiveRewardComputer(RewardConfig())

        all_grpo_items: List[Dict[str, Any]] = []
        stats = {
            "total_problems": 0,
            "all_wrong": 0,
            "all_correct": 0,
            "mixed": 0,
            "total_samples": 0,
            "total_correct": 0,
        }

        checkpoint_path = grpo_data_path + ".checkpoint.json"
        if args.resume_from > 0 and os.path.exists(checkpoint_path):
            with open(checkpoint_path) as f:
                all_grpo_items = json.load(f)
            logger.info(f"Resumed: {len(all_grpo_items)} items loaded")

        t0 = time.time()
        for idx in range(args.resume_from, len(problems)):
            problem = problems[idx]

            # Generate N diverse solutions
            samples = generate_solutions(
                generator, reward_computer, tokenizer, problem,
                n_samples=args.n_samples,
                max_new_tokens=args.max_new_tokens,
                outcome_weight=args.outcome_weight,
                cognitive_weight=args.cognitive_weight,
            )

            n_correct = sum(1 for s in samples if s.is_correct)
            n_total = len(samples)
            stats["total_problems"] += 1
            stats["total_samples"] += n_total
            stats["total_correct"] += n_correct

            if n_correct == 0:
                stats["all_wrong"] += 1
            elif n_correct == n_total:
                stats["all_correct"] += 1
            else:
                stats["mixed"] += 1

            # Compute GRPO advantages
            prompt_text = format_math_prompt(problem.question)
            advantages = compute_advantages(prompt_text, samples)
            for item in advantages:
                all_grpo_items.append({
                    "prompt": item.prompt,
                    "completion": item.completion,
                    "reward": round(item.reward, 4),
                    "advantage": round(item.advantage, 4),
                    "is_correct": item.is_correct,
                    "group_mean": round(item.group_mean, 4),
                    "group_std": round(item.group_std, 4),
                })

            # Progress
            elapsed = time.time() - t0
            rate = stats["total_problems"] / max(elapsed, 1)
            eta = (len(problems) - idx - 1) / max(rate, 0.001)

            if (idx + 1) % 10 == 0 or idx == len(problems) - 1:
                acc = stats["total_correct"] / max(stats["total_samples"], 1)
                logger.info(
                    f"[{idx+1}/{len(problems)}] "
                    f"correct={stats['total_correct']}/{stats['total_samples']} ({acc:.1%}) "
                    f"all_wrong={stats['all_wrong']} all_correct={stats['all_correct']} "
                    f"mixed={stats['mixed']} "
                    f"grpo_items={len(all_grpo_items)} "
                    f"ETA={eta/60:.0f}min"
                )

            # Checkpoint
            if (idx + 1) % args.checkpoint_every == 0:
                with open(checkpoint_path, "w") as f:
                    json.dump(all_grpo_items, f, ensure_ascii=False)
                logger.info(f"  Checkpoint: {len(all_grpo_items)} items")

            # VRAM hygiene
            if (idx + 1) % 20 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Save GRPO data
        with open(grpo_data_path, "w", encoding="utf-8") as f:
            json.dump(all_grpo_items, f, indent=2, ensure_ascii=False)

        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        elapsed = time.time() - t0
        logger.info("=" * 60)
        logger.info("PHASE A COMPLETE: Generation + Scoring")
        logger.info("=" * 60)
        logger.info(f"  Time:         {elapsed/3600:.1f}h")
        logger.info(f"  Problems:     {stats['total_problems']}")
        logger.info(f"  GRPO items:   {len(all_grpo_items)}")
        logger.info(f"  Accuracy:     {stats['total_correct']}/{stats['total_samples']} "
                     f"({stats['total_correct']/max(stats['total_samples'],1):.1%})")
        logger.info(f"  All wrong:    {stats['all_wrong']} (skipped — beyond model capability)")
        logger.info(f"  Mixed:        {stats['mixed']} (best for GRPO learning signal)")
        logger.info(f"  All correct:  {stats['all_correct']} (reward shaping only)")

        # Release generation model
        del model, generator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ══════════════════════════════════════════════
    # PHASE B: GRPO Training
    # ══════════════════════════════════════════════
    if args.phase in ("all", "train"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("=" * 60)
        logger.info("PHASE B: GRPO Policy Gradient Training")
        logger.info("=" * 60)

        # Load GRPO data
        with open(grpo_data_path) as f:
            grpo_data = json.load(f)
        logger.info(f"Loaded {len(grpo_data)} GRPO items")

        # Stats
        n_pos = sum(1 for d in grpo_data if d["advantage"] > 0)
        n_neg = sum(1 for d in grpo_data if d["advantage"] < 0)
        n_correct = sum(1 for d in grpo_data if d["is_correct"])
        logger.info(f"  Positive advantage: {n_pos}")
        logger.info(f"  Negative advantage: {n_neg}")
        logger.info(f"  Correct answers:    {n_correct}")

        # Reconstruct GRPOItem objects
        grpo_items = [
            GRPOItem(
                prompt=d["prompt"],
                completion=d["completion"],
                reward=d["reward"],
                advantage=d["advantage"],
                is_correct=d["is_correct"],
                group_mean=d["group_mean"],
                group_std=d["group_std"],
            )
            for d in grpo_data
        ]

        # Load fresh SFT model for training
        logger.info(f"Loading SFT model for training: {args.sft_model}")
        tokenizer = AutoTokenizer.from_pretrained(args.sft_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.sft_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer, model = register_metis_special_tokens(tokenizer, model)

        # Apply LoRA
        if args.lora_rank and args.lora_rank > 0:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_rank,  # alpha = r → scaling = 1.0
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                task_type="CAUSAL_LM",
                bias="none",
                lora_dropout=0.05,
            )
            model = get_peft_model(model, lora_config)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            logger.info(f"  LoRA: {trainable:,} trainable / {total:,} total "
                        f"({100*trainable/total:.2f}%)")

        # Run GRPO training
        train_output = os.path.join(args.output, "grpo_model")
        run_grpo_training(
            model=model,
            tokenizer=tokenizer,
            grpo_items=grpo_items,
            output_dir=train_output,
            epochs=args.epochs,
            batch_size=2,       # GB10-safe: batch=2
            grad_accum=8,       # Effective batch = 16
            learning_rate=args.lr,
            save_steps=100,
        )

        logger.info("=" * 60)
        logger.info("GRPO MATH TRAINING COMPLETE")
        logger.info(f"  Model saved to: {train_output}/final_model")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
