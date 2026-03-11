#!/usr/bin/env python3
"""
Ground-Truth Gated Reject Sampling DPO Data Generator v2

Architecture (per user spec):
  1. Load SFT-warmed model (Qwen-7B-METIS-SFT) as generation engine
  2. High-temperature rollout: N=8 diverse samples per prompt with METIS cognitive tracing
  3. Ground truth gate: verify each sample against Orca reference answer
     - All wrong → discard prompt (超出认知上限)
  4. Preference pairing:
     Chosen  = correct + highest METIS cognitive reward
     Rejected = incorrect (but deceptively fluent), OR correct + worst cognitive reward
  5. Save high-quality DPO pairs with real thought variance

Usage:
  python tools/generate_dpo_pairs_v2.py \
    --sft-model experiment_output_7B_restructured/metis_sft_base \
    --orca-data data/external_dpo_pairs.json \
    --output data/dpo_pairs_v2_gated.json \
    --n-samples 8 --max-prompts 2000
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

# Add project root to path
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

# ─────────────────────────────────────────────────────
# Ground Truth Verification
# ─────────────────────────────────────────────────────

_STOPWORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and',
    'or', 'if', 'while', 'about', 'that', 'this', 'these', 'those', 'it',
    'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
    'she', 'her', 'they', 'them', 'their', 'what', 'which', 'who', 'whom',
})


def _strip_thinking(text: str) -> str:
    """Remove <thinking>...</thinking> blocks."""
    return re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL).strip()


def verify_against_reference(
    generated: str,
    reference: str,
    threshold: float = 0.25,
) -> Tuple[bool, float]:
    """Verify generated answer against reference using content overlap.

    Multi-signal verification:
      1. Word-level Jaccard on content words (factual alignment)
      2. Key number/entity match (quantitative accuracy)

    Args:
        generated: Model's generated text (thinking blocks stripped)
        reference: Orca reference answer (chosen)
        threshold: Minimum score to pass ground truth gate

    Returns:
        (passed: bool, score: float)
    """
    gen_clean = _strip_thinking(generated).lower()
    ref_clean = reference.lower()

    if not gen_clean or not ref_clean:
        return False, 0.0

    # Signal 1: Content word Jaccard overlap
    gen_words = set(gen_clean.split()) - _STOPWORDS
    ref_words = set(ref_clean.split()) - _STOPWORDS

    # Filter very short words (articles, prepositions in other languages)
    gen_words = {w for w in gen_words if len(w) > 2}
    ref_words = {w for w in ref_words if len(w) > 2}

    if not ref_words:
        return True, 0.5  # Can't verify, be lenient

    intersection = gen_words & ref_words
    union = gen_words | ref_words
    jaccard = len(intersection) / max(len(union), 1)

    # Signal 2: Key number/entity match
    ref_numbers = set(re.findall(r'\b\d+\.?\d*\b', reference))
    gen_numbers = set(re.findall(r'\b\d+\.?\d*\b', generated))
    if ref_numbers:
        number_score = len(ref_numbers & gen_numbers) / len(ref_numbers)
    else:
        number_score = 0.5  # No numbers to check

    # Signal 3: Short answer exact containment
    # If reference is very short (< 50 chars), check direct containment
    containment_bonus = 0.0
    if len(ref_clean) < 100:
        ref_key = ref_clean.strip().rstrip('.')
        if ref_key in gen_clean:
            containment_bonus = 0.3

    score = 0.6 * jaccard + 0.2 * number_score + 0.2 * containment_bonus

    return score >= threshold, score


# ─────────────────────────────────────────────────────
# Sample Container
# ─────────────────────────────────────────────────────

@dataclass
class ScoredSample:
    """A single generated sample with all metadata."""
    text: str
    reward_total: float
    reward_breakdown: Dict[str, float]
    gt_passed: bool
    gt_score: float
    final_answer: Optional[str]
    n_tokens: int


# ─────────────────────────────────────────────────────
# Core Pipeline
# ─────────────────────────────────────────────────────

def generate_and_score_prompt(
    generator: MetisGenerator,
    reward_computer: CognitiveRewardComputer,
    tokenizer: Any,
    prompt: str,
    reference: str,
    n_samples: int = 8,
    max_new_tokens: int = 512,
    gt_threshold: float = 0.25,
) -> List[ScoredSample]:
    """Generate N samples for a prompt, score with METIS rewards + ground truth gate.

    Returns list of ScoredSample (may be empty if generation fails).
    """
    # Apply chat template for proper Instruct model behavior
    chat_prompt = format_chat(tokenizer, prompt)

    # Generate N samples with diverse temperatures
    # KV cache is shared across all samples (MetisGenerator.generate_batch)
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

    samples: List[ScoredSample] = []
    for text, trace in results:
        # Compute METIS cognitive reward (no ground_truth here — that's the gate)
        reward = reward_computer.compute(trace)

        # Ground truth verification
        gt_passed, gt_score = verify_against_reference(
            text, reference, threshold=gt_threshold,
        )

        # Extract FINAL ANSWER if present
        final_answer = extract_final_answer(text)

        samples.append(ScoredSample(
            text=text,
            reward_total=reward.total,
            reward_breakdown=reward.to_dict(),
            gt_passed=gt_passed,
            gt_score=gt_score,
            final_answer=final_answer,
            n_tokens=len(trace.events) if trace.events else 0,
        ))

    return samples


def build_dpo_pair(
    prompt: str,
    samples: List[ScoredSample],
) -> Optional[Dict[str, Any]]:
    """Build a DPO pair from scored samples using ground-truth gated preference.

    Pairing strategy:
      1. Separate samples into correct (gt_passed) and incorrect groups
      2. If no correct samples → skip (prompt too hard)
      3. If all correct → use best vs worst cognitive reward (reward shaping)
      4. Best case: Chosen = correct + best reward, Rejected = incorrect + most deceptive
    """
    correct = [s for s in samples if s.gt_passed]
    incorrect = [s for s in samples if not s.gt_passed]

    if not correct:
        return None  # All wrong → prompt exceeds model capability

    # Sort by METIS reward
    correct.sort(key=lambda s: s.reward_total, reverse=True)
    incorrect.sort(key=lambda s: s.reward_total, reverse=True)

    chosen = correct[0]  # Best reward among correct answers

    if incorrect:
        # Best case: rejected = most deceptive incorrect sample
        # (highest reward but wrong — the "eloquent nonsense" archetype)
        rejected = incorrect[0]
    elif len(correct) >= 2:
        # All correct → reward shaping: best vs worst cognitive quality
        rejected = correct[-1]
        # Skip if reward delta is too small (no signal)
        if chosen.reward_total - rejected.reward_total < 0.05:
            return None
    else:
        return None  # Only 1 correct sample, can't build pair

    return {
        "prompt": prompt,
        "chosen": chosen.text,
        "rejected": rejected.text,
        "chosen_reward": round(chosen.reward_total, 4),
        "rejected_reward": round(rejected.reward_total, 4),
        "reward_delta": round(chosen.reward_total - rejected.reward_total, 4),
        "chosen_gt_score": round(chosen.gt_score, 4),
        "rejected_gt_score": round(rejected.gt_score, 4),
        "n_correct": len(correct),
        "n_incorrect": len(incorrect),
        "pairing_type": "correct_vs_incorrect" if incorrect else "reward_shaping",
    }


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ground-Truth Gated Reject Sampling DPO")
    parser.add_argument("--sft-model", default="experiment_output_7B_restructured/metis_sft_base",
                        help="Path to SFT-warmed model (has METIS special tokens)")
    parser.add_argument("--orca-data", default="data/external_dpo_pairs.json",
                        help="Raw Orca DPO pairs (before template injection)")
    parser.add_argument("--output", default="data/dpo_pairs_v2_gated.json",
                        help="Output path for gated DPO pairs")
    parser.add_argument("--n-samples", type=int, default=8,
                        help="Samples per prompt (diverse rollouts)")
    parser.add_argument("--max-prompts", type=int, default=2000,
                        help="Max prompts to process")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Max generation length per sample")
    parser.add_argument("--gt-threshold", type=float, default=0.25,
                        help="Ground truth verification threshold")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Save checkpoint every N prompts")
    parser.add_argument("--resume-from", type=int, default=0,
                        help="Resume from prompt index N")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("METIS DPO v2: Ground-Truth Gated Reject Sampling")
    logger.info("=" * 60)
    logger.info(f"  SFT Model:      {args.sft_model}")
    logger.info(f"  Orca Data:      {args.orca_data}")
    logger.info(f"  Output:         {args.output}")
    logger.info(f"  Samples/Prompt: {args.n_samples}")
    logger.info(f"  Max Prompts:    {args.max_prompts}")
    logger.info(f"  GT Threshold:   {args.gt_threshold}")

    # ── Load SFT model ──
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading SFT model...")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Ensure METIS special tokens are registered
    tokenizer, model = register_metis_special_tokens(tokenizer, model)
    device = str(next(model.parameters()).device)
    logger.info(f"  Device: {device}, Params: {sum(p.numel() for p in model.parameters())/1e9:.1f}B")

    # ── Load Orca data ──
    logger.info(f"Loading Orca data from {args.orca_data}...")
    with open(args.orca_data, "r", encoding="utf-8") as f:
        orca_data = json.load(f)
    logger.info(f"  Total prompts available: {len(orca_data)}")

    n_prompts = min(args.max_prompts, len(orca_data))
    orca_data = orca_data[:n_prompts]

    # ── Initialize METIS components ──
    generator = MetisGenerator(model, tokenizer)
    reward_computer = CognitiveRewardComputer(RewardConfig())

    # ── Resume support ──
    dpo_pairs: List[Dict[str, Any]] = []
    checkpoint_path = args.output + ".checkpoint.json"
    if args.resume_from > 0 and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            dpo_pairs = json.load(f)
        logger.info(f"Resumed from checkpoint: {len(dpo_pairs)} pairs loaded")

    # ── Stats tracking ──
    stats = {
        "total_prompts": 0,
        "prompts_all_wrong": 0,   # Discarded: all N samples incorrect
        "prompts_no_pair": 0,     # Could not form a valid pair
        "prompts_paired": 0,      # Successfully built DPO pair
        "correct_vs_incorrect": 0,
        "reward_shaping": 0,
        "mean_reward_delta": 0.0,
        "total_samples": 0,
        "total_correct": 0,
    }

    start_time = time.time()

    for idx in range(args.resume_from, n_prompts):
        entry = orca_data[idx]
        prompt = entry["prompt"]
        reference = entry["chosen"]  # Orca's correct answer as ground truth

        # ── Generate N diverse samples with METIS tracing ──
        samples = generate_and_score_prompt(
            generator=generator,
            reward_computer=reward_computer,
            tokenizer=tokenizer,
            prompt=prompt,
            reference=reference,
            n_samples=args.n_samples,
            max_new_tokens=args.max_new_tokens,
            gt_threshold=args.gt_threshold,
        )

        stats["total_prompts"] += 1
        stats["total_samples"] += len(samples)
        n_correct = sum(1 for s in samples if s.gt_passed)
        stats["total_correct"] += n_correct

        # ── Build DPO pair ──
        pair = build_dpo_pair(prompt, samples)

        if pair is None:
            if n_correct == 0:
                stats["prompts_all_wrong"] += 1
            else:
                stats["prompts_no_pair"] += 1
        else:
            dpo_pairs.append(pair)
            stats["prompts_paired"] += 1
            stats[pair["pairing_type"]] += 1
            stats["mean_reward_delta"] = (
                stats["mean_reward_delta"] * (stats["prompts_paired"] - 1)
                + pair["reward_delta"]
            ) / stats["prompts_paired"]

        # ── Progress log ──
        elapsed = time.time() - start_time
        rate = stats["total_prompts"] / max(elapsed, 1)
        eta = (n_prompts - idx - 1) / max(rate, 0.001)

        if (idx + 1) % 10 == 0 or idx == n_prompts - 1:
            logger.info(
                f"[{idx+1}/{n_prompts}] pairs={len(dpo_pairs)} "
                f"correct_rate={stats['total_correct']}/{stats['total_samples']} "
                f"({stats['total_correct']/max(stats['total_samples'],1):.1%}) "
                f"all_wrong={stats['prompts_all_wrong']} "
                f"Δr={stats['mean_reward_delta']:.4f} "
                f"ETA={eta/60:.0f}min"
            )

        # ── Checkpoint ──
        if (idx + 1) % args.checkpoint_every == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(dpo_pairs, f, indent=2, ensure_ascii=False)
            logger.info(f"  Checkpoint saved: {len(dpo_pairs)} pairs")

        # ── VRAM hygiene ──
        if (idx + 1) % 20 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Final save ──
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dpo_pairs, f, indent=2, ensure_ascii=False)

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Time:            {elapsed/3600:.1f}h")
    logger.info(f"  Total prompts:   {stats['total_prompts']}")
    logger.info(f"  DPO pairs built: {len(dpo_pairs)}")
    logger.info(f"  All-wrong skip:  {stats['prompts_all_wrong']}")
    logger.info(f"  No-pair skip:    {stats['prompts_no_pair']}")
    logger.info(f"  Correct vs Incorrect: {stats['correct_vs_incorrect']}")
    logger.info(f"  Reward Shaping:  {stats['reward_shaping']}")
    logger.info(f"  Mean Δ reward:   {stats['mean_reward_delta']:.4f}")
    logger.info(f"  Overall correct rate: "
                f"{stats['total_correct']}/{stats['total_samples']} "
                f"({stats['total_correct']/max(stats['total_samples'],1):.1%})")
    logger.info(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
