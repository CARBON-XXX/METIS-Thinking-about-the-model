#!/usr/bin/env python3
"""
METIS Phase 23.5 — Production Golden Anchor Dataset Builder

Downloads high-quality samples from HuggingFace datasets and formats
them into the JSONL schema expected by the DreamingDaemon experience
replay system.

Sources:
  - openai/gsm8k          (train split) → math reasoning prompts
  - truthful_qa/generation (validation)  → factual QA prompts

Output: data/production_golden_anchor.jsonl  (exactly 500 prompts)

JSONL format (per line):
    {"prompt": "...", "_meta": {"source": "golden", "dataset": "gsm8k|truthful_qa"}}

Usage:
    python tools/build_golden_dataset.py [--output data/production_golden_anchor.jsonl] [--count 500]
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("metis.build_golden")

SEED = 42


def load_gsm8k(target_count: int) -> List[Dict[str, Any]]:
    """Load GSM8K math problems from HuggingFace."""
    from datasets import load_dataset

    logger.info("Loading openai/gsm8k (train split)...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    logger.info(f"  GSM8K loaded: {len(ds)} examples")

    records: List[Dict[str, Any]] = []
    for item in ds:
        question = item["question"].strip()
        if not question or len(question) < 20:
            continue
        records.append({
            "prompt": question,
            "_meta": {
                "source": "golden",
                "dataset": "gsm8k",
                "category": "math_reasoning",
            },
        })

    random.shuffle(records)
    selected = records[:target_count]
    logger.info(f"  GSM8K selected: {len(selected)}/{len(records)}")
    return selected


def load_truthful_qa(target_count: int) -> List[Dict[str, Any]]:
    """Load TruthfulQA questions from HuggingFace."""
    from datasets import load_dataset

    logger.info("Loading truthful_qa (generation, validation split)...")
    ds = load_dataset("truthful_qa", "generation", split="validation")
    logger.info(f"  TruthfulQA loaded: {len(ds)} examples")

    records: List[Dict[str, Any]] = []
    for item in ds:
        question = item["question"].strip()
        if not question or len(question) < 10:
            continue
        records.append({
            "prompt": question,
            "_meta": {
                "source": "golden",
                "dataset": "truthful_qa",
                "category": "factual_qa",
            },
        })

    random.shuffle(records)
    selected = records[:target_count]
    logger.info(f"  TruthfulQA selected: {len(selected)}/{len(records)}")
    return selected


def build_golden_dataset(output_path: str, total_count: int = 500) -> Path:
    """Build the production golden anchor dataset.

    Split:
      - 60% GSM8K math reasoning  (300 prompts)
      - 40% TruthfulQA factual QA (200 prompts)

    Returns:
        Path to the written JSONL file.
    """
    random.seed(SEED)

    n_math = int(total_count * 0.6)
    n_qa = total_count - n_math

    logger.info(f"Building golden dataset: {total_count} total ({n_math} math + {n_qa} QA)")

    math_records = load_gsm8k(n_math)
    qa_records = load_truthful_qa(n_qa)

    # If either source is short, rebalance
    if len(math_records) < n_math:
        shortfall = n_math - len(math_records)
        logger.warning(f"  GSM8K short by {shortfall}, pulling more from TruthfulQA")
        qa_records = load_truthful_qa(n_qa + shortfall)

    if len(qa_records) < n_qa:
        shortfall = n_qa - len(qa_records)
        logger.warning(f"  TruthfulQA short by {shortfall}, pulling more from GSM8K")
        math_records = load_gsm8k(n_math + shortfall)

    combined = math_records[:n_math] + qa_records[:n_qa]

    # Exact count enforcement — pad or truncate
    if len(combined) > total_count:
        combined = combined[:total_count]
    elif len(combined) < total_count:
        logger.warning(
            f"  Only {len(combined)} records available (target: {total_count}). "
            f"Duplicating random samples to reach target."
        )
        while len(combined) < total_count:
            combined.append(random.choice(combined))

    random.shuffle(combined)

    # Write JSONL
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        for record in combined:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Written {len(combined)} prompts to {out}")

    # Verify
    from collections import Counter
    datasets_dist = Counter(r["_meta"]["dataset"] for r in combined)
    for ds_name, count in datasets_dist.most_common():
        logger.info(f"  {ds_name}: {count} ({count/len(combined)*100:.0f}%)")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="METIS Phase 23.5 — Build Production Golden Anchor Dataset"
    )
    parser.add_argument(
        "--output", default="data/production_golden_anchor.jsonl",
        help="Output JSONL path (default: data/production_golden_anchor.jsonl)",
    )
    parser.add_argument(
        "--count", type=int, default=500,
        help="Total number of prompts (default: 500)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    output = build_golden_dataset(args.output, args.count)

    # Final verification
    with open(output) as f:
        lines = [l for l in f if l.strip()]
    print(f"\n{'=' * 60}")
    print(f"  Production Golden Anchor Dataset — COMPLETE")
    print(f"  Output: {output}")
    print(f"  Records: {len(lines)}")
    print(f"{'=' * 60}")

    if len(lines) != args.count:
        print(f"  WARNING: Expected {args.count}, got {len(lines)}")
        sys.exit(1)
    else:
        print(f"  Exact count verified: {len(lines)} == {args.count} ✅")


if __name__ == "__main__":
    main()
