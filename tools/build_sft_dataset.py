#!/usr/bin/env python3
"""
METIS SFT Seed Dataset Builder

Constructs a physically-grounded SFT dataset where cognitive states map to
real structural differences in text — fixing the DPO failure caused by
static <thinking> templates (∇L = 0).

Data Sources:
  FAST (System 1): tatsu-lab/alpaca — short-form QA, direct answers
  DEEP (System 2): openai/gsm8k    — multi-step math with real CoT

Output:
  1. metis_sft_seed.jsonl         — ShareGPT-style {"system","user","assistant"}
  2. metis_sft_seed_legacy.json   — Legacy format [{"text": ...}] for trainer_phase.py

Usage:
  python tools/build_sft_dataset.py --n-fast 500 --n-deep 500
  python tools/build_sft_dataset.py --output data/metis_sft_seed.jsonl --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────

METIS_SYSTEM_PROMPT = (
    "You are METIS, an AI with a dynamic cognitive routing layer. "
    "Analyze the complexity of the user's request and allocate compute accordingly."
)

# METIS special tokens (must match metis/training/tokenizer_utils.py)
FAST_PREFIX = "[COGNITIVE_STATE: FAST]\n[ENTROPY: LOW]\n"
DEEP_PREFIX = "[COGNITIVE_STATE: DEEP]\n[ENTROPY: HIGH]\n"
THINKING_OPEN = "<thinking>\n"
THINKING_CLOSE = "</thinking>\n"


# ─────────────────────────────────────────────────────
# Data Download
# ─────────────────────────────────────────────────────

def _is_math_toxic(text: str) -> bool:
    """Detect math/logic content that Alpaca often hallucinates on.

    Alpaca contains mathematically incorrect responses (e.g., 2/4 = 2).
    We must exclude ALL math-adjacent samples to prevent poisoning
    METIS's logic network during FAST-state SFT.
    """
    lowered = text.lower()
    # Keyword blacklist: any match → toxic
    _MATH_KEYWORDS = [
        'math', 'equation', 'calculate', 'number', 'value',
        'solve', 'formula', 'algebra', 'arithmetic', 'geometry',
        'sum', 'difference', 'product', 'quotient', 'remainder',
        'percent', 'fraction', 'decimal', 'integer', 'variable',
        'coefficient', 'exponent', 'logarithm', 'factorial',
        'probability', 'average', 'median', 'ratio', 'proportion',
    ]
    if any(kw in lowered for kw in _MATH_KEYWORDS):
        return True

    # Symbol blacklist: math operators in instruction/input
    # (these catch "find x", "4x + 2y = 10", "3 * 5", etc.)
    _MATH_SYMBOLS = ['+', '-', '*', '/', '=']
    if any(sym in text for sym in _MATH_SYMBOLS):
        return True

    # Single-letter variable patterns: standalone x or y as math variables
    # Exclude common English uses like "x-ray", "year"
    if re.search(r'\b[xy]\b', lowered) and not re.search(r'x-ray|proxy|taxonomy|luxury|galaxy', lowered):
        return True

    return False


def download_alpaca(n_samples: int) -> List[Dict[str, str]]:
    """Download and parse tatsu-lab/alpaca for FAST (System 1) samples.

    Returns list of {"user": ..., "answer": ...} dicts.
    Strict exclusion: filters out ALL math/logic samples to prevent
    hallucinated arithmetic from poisoning METIS's FAST state.
    Only keeps purely linguistic tasks (summarization, translation, QA, etc.).
    """
    from datasets import load_dataset

    logger.info(f"Downloading tatsu-lab/alpaca (target: {n_samples} FAST samples)...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    logger.info(f"  Raw dataset size: {len(ds)}")

    samples: List[Dict[str, str]] = []
    n_math_filtered = 0
    for ex in ds:
        instruction = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        output = (ex.get("output") or "").strip()

        if not instruction or not output:
            continue

        # Skip overly long answers — FAST should be concise
        if len(output) > 500:
            continue

        # CRITICAL: Exclude math/logic samples — Alpaca hallucinates arithmetic
        combined_input = f"{instruction} {inp}"
        if _is_math_toxic(combined_input) or _is_math_toxic(output):
            n_math_filtered += 1
            continue

        # Skip samples that look like multi-step reasoning (not FAST)
        if any(marker in output.lower() for marker in [
            "step 1", "step 2", "first,", "second,", "finally,",
            "let's break", "let me think",
        ]):
            continue

        # Combine instruction + input as user query
        user = instruction
        if inp:
            user = f"{instruction}\n{inp}"

        samples.append({"user": user, "answer": output})

        if len(samples) >= n_samples:
            break

    logger.info(f"  Extracted {len(samples)} FAST samples from Alpaca "
                f"(filtered {n_math_filtered} math/logic toxic samples)")
    if len(samples) < n_samples:
        logger.warning(
            f"  Only {len(samples)} samples passed filtering "
            f"(requested {n_samples}). Using all available."
        )
    return samples


def download_gsm8k(n_samples: int) -> List[Dict[str, str]]:
    """Download and parse openai/gsm8k for DEEP (System 2) samples.

    GSM8K answer format: step-by-step reasoning ending with "#### <number>"
    Returns list of {"user": ..., "reasoning": ..., "final_answer": ...} dicts.
    """
    from datasets import load_dataset

    logger.info(f"Downloading openai/gsm8k (target: {n_samples} DEEP samples)...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    logger.info(f"  Raw dataset size: {len(ds)}")

    samples: List[Dict[str, str]] = []
    for ex in ds:
        question = (ex.get("question") or "").strip()
        answer = (ex.get("answer") or "").strip()

        if not question or not answer:
            continue

        # Extract final numeric answer from "#### N" marker
        match = re.search(r'####\s*(.+)', answer)
        if not match:
            continue

        final_answer = match.group(1).strip()

        # Everything before #### is the reasoning chain
        reasoning = answer[:match.start()].strip()
        if not reasoning:
            continue

        samples.append({
            "user": question,
            "reasoning": reasoning,
            "final_answer": final_answer,
        })

        if len(samples) >= n_samples:
            break

    logger.info(f"  Extracted {len(samples)} DEEP samples from GSM8K")
    if len(samples) < n_samples:
        logger.warning(
            f"  Only {len(samples)} samples passed filtering "
            f"(requested {n_samples}). Using all available."
        )
    return samples


# ─────────────────────────────────────────────────────
# Data Transformation
# ─────────────────────────────────────────────────────

_FAST_THINKING_TEMPLATES = [
    "This is a straightforward request. I can respond directly.\n"
    "[SELF-CRITIQUE: None needed, confidence is high.]",

    "Low complexity query. No multi-step reasoning required.\n"
    "[SELF-CRITIQUE: Direct recall is sufficient here.]",

    "Simple factual or linguistic task. Allocating minimal compute.\n"
    "[SELF-CRITIQUE: High confidence, no ambiguity detected.]",

    "The answer is well within my training distribution. Responding immediately.\n"
    "[SELF-CRITIQUE: No need for extended reasoning.]",

    "Routine request with a clear expected output. Fast path engaged.\n"
    "[SELF-CRITIQUE: Confidence is high, direct answer appropriate.]",

    "This requires recall, not reasoning. Single-step response.\n"
    "[SELF-CRITIQUE: None needed.]",

    "Quick classification or translation task. No decomposition needed.\n"
    "[SELF-CRITIQUE: Straightforward, proceeding with direct answer.]",

    "The user's intent is unambiguous. I will provide a concise response.\n"
    "[SELF-CRITIQUE: No uncertainty, direct path is optimal.]",
]


def transform_fast(sample: Dict[str, str], idx: int = 0) -> Dict[str, str]:
    """Transform an Alpaca sample into METIS FAST format.

    Output assistant field:
      [COGNITIVE_STATE: FAST]
      [ENTROPY: LOW]
      <thinking>
      {varied brief reasoning from template pool}
      </thinking>
      {direct answer}

    v4 FIX: Added <thinking> block to match DEEP structural skeleton.
    Without this, FAST samples had ~2x fewer trainable tokens than DEEP,
    causing gradient signal imbalance → model defaulted to DEEP (v3: 0/3 FAST).
    Uses varied templates to avoid static thinking trap (cf. synthesize_metis_chosen bug).
    """
    thinking = _FAST_THINKING_TEMPLATES[idx % len(_FAST_THINKING_TEMPLATES)]
    assistant = (
        f"{FAST_PREFIX}"
        f"{THINKING_OPEN}"
        f"{thinking}\n"
        f"{THINKING_CLOSE}"
        f"{sample['answer']}"
    )
    return {
        "system": METIS_SYSTEM_PROMPT,
        "user": sample["user"],
        "assistant": assistant,
    }


def _clean_gsm8k_reasoning(text: str) -> str:
    """Strip GSM8K internal calc markers <<3*5=15>> from reasoning text.

    These are dataset annotation artifacts, not natural language.
    Leaving them in would teach METIS to generate <<...>> tokens.
    Example: "14 * 200 = <<14*200=2800>>2800 pages" → "14 * 200 = 2800 pages"
    """
    return re.sub(r'<<[^>]*>>', '', text)


def transform_deep(sample: Dict[str, str]) -> Dict[str, str]:
    """Transform a GSM8K sample into METIS DEEP format.

    Output assistant field:
      [COGNITIVE_STATE: DEEP]
      [ENTROPY: HIGH]
      <thinking>
      {actual step-by-step reasoning from GSM8K, cleaned of calc markers}
      </thinking>
      FINAL ANSWER: {number}
    """
    reasoning = _clean_gsm8k_reasoning(sample['reasoning'])
    assistant = (
        f"{DEEP_PREFIX}"
        f"{THINKING_OPEN}"
        f"{reasoning}\n"
        f"{THINKING_CLOSE}"
        f"FINAL ANSWER: {sample['final_answer']}"
    )
    return {
        "system": METIS_SYSTEM_PROMPT,
        "user": sample["user"],
        "assistant": assistant,
    }


# ─────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────

def export_jsonl(data: List[Dict[str, str]], path: str) -> None:
    """Export to JSONL (one JSON object per line)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Exported {len(data)} samples to {path}")


def export_legacy_json(data: List[Dict[str, str]], path: str) -> None:
    """Export legacy format for trainer_phase.py compatibility.

    Format: [{"text": "System: ...\nUser: ...\nassistant_content"}]
    This matches _run_sft_warmup() which expects {"text": "prompt\\nresponse"}.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    legacy: List[Dict[str, str]] = []
    for item in data:
        text = (
            f"System: {item['system']}\n"
            f"User: {item['user']}\n"
            f"{item['assistant']}"
        )
        legacy.append({"text": text})

    with open(path, "w", encoding="utf-8") as f:
        json.dump(legacy, f, indent=2, ensure_ascii=False)
    logger.info(f"Exported {len(legacy)} samples to {path} (legacy format)")


# ─────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────

def validate_dataset(data: List[Dict[str, str]]) -> None:
    """Validate the dataset for structural correctness."""
    required_keys = {"system", "user", "assistant"}
    n_fast = 0
    n_deep = 0
    errors: List[str] = []

    for i, item in enumerate(data):
        # Check keys
        missing = required_keys - set(item.keys())
        if missing:
            errors.append(f"Sample {i}: missing keys {missing}")
            continue

        # Check non-empty
        for key in required_keys:
            if not item[key].strip():
                errors.append(f"Sample {i}: empty '{key}' field")

        assistant = item["assistant"]

        # Classify and validate format
        if assistant.startswith("[COGNITIVE_STATE: FAST]"):
            n_fast += 1
            if "[ENTROPY: LOW]" not in assistant:
                errors.append(f"Sample {i}: FAST missing [ENTROPY: LOW]")
            # v4: FAST samples now include brief <thinking> blocks for structural parity
            if "<thinking>" not in assistant:
                errors.append(f"Sample {i}: FAST missing <thinking>")
            if "</thinking>" not in assistant:
                errors.append(f"Sample {i}: FAST missing </thinking>")

        elif assistant.startswith("[COGNITIVE_STATE: DEEP]"):
            n_deep += 1
            if "[ENTROPY: HIGH]" not in assistant:
                errors.append(f"Sample {i}: DEEP missing [ENTROPY: HIGH]")
            if "<thinking>" not in assistant:
                errors.append(f"Sample {i}: DEEP missing <thinking>")
            if "</thinking>" not in assistant:
                errors.append(f"Sample {i}: DEEP missing </thinking>")
            if "FINAL ANSWER:" not in assistant:
                errors.append(f"Sample {i}: DEEP missing FINAL ANSWER:")
        else:
            errors.append(f"Sample {i}: unknown cognitive state prefix")

    if errors:
        for e in errors[:10]:
            logger.error(f"  VALIDATION: {e}")
        if len(errors) > 10:
            logger.error(f"  ... and {len(errors) - 10} more errors")
        raise ValueError(f"Dataset validation failed with {len(errors)} errors")

    logger.info(f"Validation passed: {n_fast} FAST + {n_deep} DEEP = {len(data)} total")


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="METIS SFT Seed Dataset Builder — "
                    "constructs physically-grounded cognitive routing data"
    )
    parser.add_argument("--n-fast", type=int, default=500,
                        help="Number of FAST (System 1) samples from Alpaca")
    parser.add_argument("--n-deep", type=int, default=500,
                        help="Number of DEEP (System 2) samples from GSM8K")
    parser.add_argument("--output", type=str, default="data/metis_sft_seed.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--legacy-output", type=str, default=None,
                        help="Legacy JSON output for trainer_phase.py "
                             "(default: auto-derived from --output)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # Derive legacy path if not specified
    if args.legacy_output is None:
        base, _ = os.path.splitext(args.output)
        args.legacy_output = f"{base}_legacy.json"

    random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("METIS SFT Seed Dataset Builder")
    logger.info("=" * 60)
    logger.info(f"  FAST samples: {args.n_fast} (Alpaca)")
    logger.info(f"  DEEP samples: {args.n_deep} (GSM8K)")
    logger.info(f"  Output:       {args.output}")
    logger.info(f"  Legacy:       {args.legacy_output}")
    logger.info(f"  Seed:         {args.seed}")

    # ── Download ──
    alpaca_data = download_alpaca(args.n_fast)
    gsm8k_data = download_gsm8k(args.n_deep)

    # ── Transform ──
    logger.info("Transforming samples...")
    fast_samples = [transform_fast(s, idx=i) for i, s in enumerate(alpaca_data)]
    deep_samples = [transform_deep(s) for s in gsm8k_data]

    # ── Combine & Shuffle ──
    combined = fast_samples + deep_samples
    random.shuffle(combined)
    logger.info(f"Combined dataset: {len(combined)} samples (shuffled)")

    # ── Validate ──
    validate_dataset(combined)

    # ── Export ──
    export_jsonl(combined, args.output)
    export_legacy_json(combined, args.legacy_output)

    # ── Summary stats ──
    fast_lens = [len(s["assistant"]) for s in fast_samples]
    deep_lens = [len(s["assistant"]) for s in deep_samples]
    logger.info("=" * 60)
    logger.info("BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  FAST: {len(fast_samples)} samples, "
                f"avg {sum(fast_lens)/max(len(fast_lens),1):.0f} chars")
    logger.info(f"  DEEP: {len(deep_samples)} samples, "
                f"avg {sum(deep_lens)/max(len(deep_lens),1):.0f} chars")
    logger.info(f"  Total: {len(combined)} samples")
    logger.info(f"  JSONL: {args.output}")
    logger.info(f"  Legacy: {args.legacy_output}")

    # ── Print sample previews ──
    logger.info("\n--- FAST Sample Preview ---")
    for s in combined:
        if "[COGNITIVE_STATE: FAST]" in s["assistant"]:
            logger.info(f"  user: {s['user'][:80]}...")
            logger.info(f"  assistant: {s['assistant'][:150]}...")
            break
    logger.info("\n--- DEEP Sample Preview ---")
    for s in combined:
        if "[COGNITIVE_STATE: DEEP]" in s["assistant"]:
            logger.info(f"  user: {s['user'][:80]}...")
            logger.info(f"  assistant: {s['assistant'][:200]}...")
            break


if __name__ == "__main__":
    main()
