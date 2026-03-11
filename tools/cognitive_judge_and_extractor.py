#!/usr/bin/env python3
"""
METIS Phase 3 — Cognitive Judge & DPO Pair Extractor
=====================================================

Ingests sampled trajectories from Phase 2, scores each trajectory with
a deterministic reward function R, and extracts strictly contrasting
(Chosen, Rejected) pairs for DPO training.

Reward Function R(y):
  1. Base Format Score:    +2 valid tag, -10 broken/missing
  2. Cognitive Alignment:  +5 correct route, -3/-5 misroute
  3. Correctness Heuristic: +3 math answer extraction, +2 reasonable output

Pair Extraction:
  - Sort 6 trajectories by R
  - chosen = argmax(R), rejected = best "wrong-route" or lowest R
  - Discard if R_chosen - R_rejected < 5.0
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

DEFAULT_INPUT_PATH = "data/metis_sampled_trajectories.jsonl"
DEFAULT_OUTPUT_PATH = "data/metis_dpo_pairs.jsonl"

MIN_SCORE_GAP = 5.0

TASK_COMPLEX = "TASK_COMPLEX"
TASK_SIMPLE = "TASK_SIMPLE"


# ─────────────────────────────────────────────────────
# Task Classification
# ─────────────────────────────────────────────────────

def classify_task(prompt: Dict[str, Any]) -> str:
    """Classify prompt as TASK_COMPLEX (GSM8K math) or TASK_SIMPLE (Alpaca).

    Uses the source field directly — it was tracked during Phase 2 sampling.
    """
    source = prompt.get("source", "")
    if "gsm8k" in source.lower():
        return TASK_COMPLEX
    return TASK_SIMPLE


# ─────────────────────────────────────────────────────
# Tag Extraction
# ─────────────────────────────────────────────────────

TAG_FAST = "FAST"
TAG_DEEP = "DEEP"
TAG_NONE = "NONE"

_COGNITIVE_TAG_RE = re.compile(
    r"\[COGNITIVE_STATE:\s*(FAST|DEEP)\]", re.IGNORECASE
)
_THINKING_OPEN_RE = re.compile(r"<thinking>", re.IGNORECASE)
_THINKING_CLOSE_RE = re.compile(r"</thinking>", re.IGNORECASE)


def extract_cognitive_tag(text: str) -> str:
    """Extract cognitive state tag from trajectory text.

    Returns TAG_FAST, TAG_DEEP, or TAG_NONE.
    """
    match = _COGNITIVE_TAG_RE.search(text)
    if match:
        return match.group(1).upper()
    return TAG_NONE


def has_valid_thinking_block(text: str) -> bool:
    """Check if text has properly paired <thinking>...</thinking> tags."""
    opens = len(_THINKING_OPEN_RE.findall(text))
    closes = len(_THINKING_CLOSE_RE.findall(text))
    return opens >= 1 and closes >= 1 and opens == closes


# ─────────────────────────────────────────────────────
# Correctness Heuristics
# ─────────────────────────────────────────────────────

_FINAL_ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer\s+is|####)\s*[:\s]*\$?\\?boxed\{?(\-?\d[\d,\.]*)\}?"
    r"|(?:####)\s*(\-?\d[\d,\.]*)"
    r"|(?:=\s*)(\-?\d[\d,\.]*)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def extract_numeric_answer(text: str) -> Optional[str]:
    """Try to extract a numeric final answer from a math trajectory."""
    # Try structured patterns first
    match = _FINAL_ANSWER_RE.search(text)
    if match:
        for g in match.groups():
            if g is not None:
                return g.replace(",", "").strip()

    # Fallback: look for #### pattern (GSM8K standard)
    lines = text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("####"):
            num = line.replace("####", "").strip()
            if num:
                return num
        # Also check "The answer is X" at end
        m = re.search(r"(?:the\s+answer\s+is|=)\s*\$?(\-?\d[\d,\.]*)", line, re.IGNORECASE)
        if m:
            return m.group(1).replace(",", "").strip()

    return None


# ─────────────────────────────────────────────────────
# Reward Function R
# ─────────────────────────────────────────────────────

def compute_reward(
    text: str,
    task_type: str,
) -> Tuple[float, Dict[str, Any]]:
    """Compute reward R(y) for a single trajectory.

    Returns (total_score, breakdown_dict).
    """
    breakdown: Dict[str, Any] = {}
    score = 0.0

    # ── 1. Base Format Score ──
    tag = extract_cognitive_tag(text)
    breakdown["tag"] = tag

    starts_with_tag = (
        text.strip().startswith("[COGNITIVE_STATE: FAST]")
        or text.strip().startswith("[COGNITIVE_STATE: DEEP]")
    )
    has_thinking = has_valid_thinking_block(text)

    if tag != TAG_NONE and starts_with_tag:
        score += 2.0
        breakdown["format"] = 2.0
    elif tag != TAG_NONE:
        # Tag exists but doesn't strictly start the text
        score += 1.0
        breakdown["format"] = 1.0
    else:
        # No cognitive tag or broken thinking tags
        score -= 10.0
        breakdown["format"] = -10.0

    # Extra penalty for broken thinking blocks
    if tag != TAG_NONE and not has_thinking:
        score -= 2.0
        breakdown["thinking_broken"] = -2.0

    # ── 2. Cognitive Alignment Score ──
    alignment = 0.0
    if task_type == TASK_COMPLEX:
        if tag == TAG_DEEP:
            alignment = 5.0
        elif tag == TAG_FAST:
            alignment = -5.0
        # NONE: no alignment bonus (already penalized in format)
    else:  # TASK_SIMPLE
        if tag == TAG_FAST:
            alignment = 5.0
        elif tag == TAG_DEEP:
            alignment = -3.0

    score += alignment
    breakdown["alignment"] = alignment

    # ── 3. Correctness Heuristic ──
    correctness = 0.0
    if task_type == TASK_COMPLEX:
        num_answer = extract_numeric_answer(text)
        breakdown["extracted_answer"] = num_answer
        if num_answer is not None:
            correctness = 3.0
    else:  # TASK_SIMPLE
        # Reasonable output length check
        # Strip tags to measure actual content
        content = text.strip()
        if len(content) > 10:
            correctness = 2.0

    score += correctness
    breakdown["correctness"] = correctness

    breakdown["total"] = score
    return score, breakdown


# ─────────────────────────────────────────────────────
# Pair Extraction
# ─────────────────────────────────────────────────────

def extract_dpo_pair(
    prompt: Dict[str, Any],
    task_type: str,
    min_gap: float = MIN_SCORE_GAP,
) -> Optional[Dict[str, Any]]:
    """Extract (chosen, rejected) pair from a prompt's 6 trajectories.

    Strategy:
      - chosen = highest scoring trajectory
      - rejected = prefer a "wrong cognitive route" trajectory over random low scorer
      - Discard if gap < min_gap
    """
    trajectories = prompt["trajectories"]
    if not trajectories:
        return None

    # Score all trajectories
    scored: List[Tuple[int, float, Dict[str, Any], str]] = []
    for traj in trajectories:
        text = traj["text"]
        score, breakdown = compute_reward(text, task_type)
        scored.append((traj["index"], score, breakdown, text))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    chosen_idx, chosen_score, chosen_bd, chosen_text = scored[0]

    # Find the best rejected trajectory:
    # Priority 1: Wrong cognitive state (still has text, not empty)
    # Priority 2: NONE tag (dropped format)
    # Priority 3: Lowest scorer
    rejected_candidate = None

    correct_tag = TAG_DEEP if task_type == TASK_COMPLEX else TAG_FAST

    # Look for wrong-route candidates (not the chosen, has text > 10 chars)
    for idx, sc, bd, text in reversed(scored):
        if idx == chosen_idx:
            continue
        if len(text.strip()) < 10:
            continue
        tag = bd["tag"]
        # Wrong route: opposite tag for this task type
        if tag != TAG_NONE and tag != correct_tag:
            if chosen_score - sc >= min_gap:
                rejected_candidate = (idx, sc, bd, text)
                break

    # If no wrong-route found, look for NONE tag
    if rejected_candidate is None:
        for idx, sc, bd, text in reversed(scored):
            if idx == chosen_idx:
                continue
            if len(text.strip()) < 10:
                continue
            if bd["tag"] == TAG_NONE:
                if chosen_score - sc >= min_gap:
                    rejected_candidate = (idx, sc, bd, text)
                    break

    # Fallback: just take the lowest scorer with text
    if rejected_candidate is None:
        for idx, sc, bd, text in reversed(scored):
            if idx == chosen_idx:
                continue
            if len(text.strip()) < 10:
                continue
            if chosen_score - sc >= min_gap:
                rejected_candidate = (idx, sc, bd, text)
                break

    if rejected_candidate is None:
        return None

    rej_idx, rej_score, rej_bd, rej_text = rejected_candidate

    # Build DPO-compatible output
    dpo_prompt = f"{prompt['system']}\n{prompt['user']}"

    return {
        "prompt": dpo_prompt,
        "chosen": chosen_text,
        "rejected": rej_text,
        # Metadata for analysis (not used by DPO trainer, but useful for debugging)
        "_meta": {
            "prompt_id": prompt["prompt_id"],
            "source": prompt["source"],
            "task_type": task_type,
            "chosen_score": chosen_score,
            "rejected_score": rej_score,
            "score_gap": chosen_score - rej_score,
            "chosen_tag": chosen_bd["tag"],
            "rejected_tag": rej_bd["tag"],
            "chosen_index": chosen_idx,
            "rejected_index": rej_idx,
        },
    }


# ─────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────

def run_pipeline(
    input_path: str,
    output_path: str,
    min_gap: float = MIN_SCORE_GAP,
) -> None:
    """Run full Phase 3 pipeline: load → score → extract → export."""

    # ── Load trajectories ──
    logger.info(f"Loading trajectories from {input_path}...")
    prompts: List[Dict[str, Any]] = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    logger.info(f"  Loaded {len(prompts)} prompts, "
                f"{sum(len(p['trajectories']) for p in prompts)} trajectories")

    # ── Classify tasks ──
    task_counts = {TASK_COMPLEX: 0, TASK_SIMPLE: 0}
    for p in prompts:
        p["_task_type"] = classify_task(p)
        task_counts[p["_task_type"]] += 1

    logger.info(f"  Task classification: {task_counts}")

    # ── Score all trajectories ──
    logger.info("Scoring trajectories with reward function R...")

    all_scores: List[float] = []
    tag_score_map: Dict[str, List[float]] = {TAG_FAST: [], TAG_DEEP: [], TAG_NONE: []}
    task_score_map: Dict[str, List[float]] = {TASK_COMPLEX: [], TASK_SIMPLE: []}

    for p in prompts:
        task_type = p["_task_type"]
        for traj in p["trajectories"]:
            score, breakdown = compute_reward(traj["text"], task_type)
            traj["_score"] = score
            traj["_breakdown"] = breakdown
            all_scores.append(score)
            tag_score_map[breakdown["tag"]].append(score)
            task_score_map[task_type].append(score)

    # Score distribution stats
    logger.info(f"  Score stats: avg={sum(all_scores)/len(all_scores):.2f}, "
                f"min={min(all_scores):.1f}, max={max(all_scores):.1f}")
    for tag in [TAG_FAST, TAG_DEEP, TAG_NONE]:
        scores = tag_score_map[tag]
        if scores:
            logger.info(f"    {tag}: n={len(scores)}, "
                        f"avg={sum(scores)/len(scores):.2f}")
    for task in [TASK_SIMPLE, TASK_COMPLEX]:
        scores = task_score_map[task]
        if scores:
            logger.info(f"    {task}: n={len(scores)}, "
                        f"avg={sum(scores)/len(scores):.2f}")

    # ── Extract DPO pairs ──
    logger.info(f"Extracting DPO pairs (min_gap={min_gap})...")

    pairs: List[Dict[str, Any]] = []
    discarded = 0
    discard_reasons: Dict[str, int] = {"gap_too_small": 0, "no_valid_rejected": 0}

    for p in prompts:
        result = extract_dpo_pair(p, p["_task_type"], min_gap)
        if result is not None:
            pairs.append(result)
        else:
            discarded += 1

    # Analyze discard reasons more precisely
    for p in prompts:
        task_type = p["_task_type"]
        trajectories = p["trajectories"]
        scored = []
        for traj in trajectories:
            score, bd = compute_reward(traj["text"], task_type)
            scored.append((score, bd, traj["text"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][0]
        worst_valid = None
        for sc, bd, text in reversed(scored):
            if len(text.strip()) >= 10:
                worst_valid = sc
                break
        if worst_valid is not None and best - worst_valid < min_gap:
            discard_reasons["gap_too_small"] += 1
        elif worst_valid is None:
            discard_reasons["no_valid_rejected"] += 1

    logger.info(f"  Valid pairs: {len(pairs)}")
    logger.info(f"  Discarded: {discarded} prompts")
    logger.info(f"  Discard reasons: {discard_reasons}")

    # ── Pair quality stats ──
    if pairs:
        gaps = [p["_meta"]["score_gap"] for p in pairs]
        chosen_tags = [p["_meta"]["chosen_tag"] for p in pairs]
        rejected_tags = [p["_meta"]["rejected_tag"] for p in pairs]

        from collections import Counter
        logger.info(f"  Score gap: avg={sum(gaps)/len(gaps):.2f}, "
                    f"min={min(gaps):.1f}, max={max(gaps):.1f}")
        logger.info(f"  Chosen tags:   {dict(Counter(chosen_tags))}")
        logger.info(f"  Rejected tags: {dict(Counter(rejected_tags))}")

        # Source breakdown
        src_counts = Counter(p["_meta"]["source"] for p in pairs)
        logger.info(f"  Source: {dict(src_counts)}")

    # ── Export ──
    logger.info(f"Exporting {len(pairs)} DPO pairs to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info(f"  Written {len(pairs)} pairs to {output_path}")

    # ── Final Report ──
    logger.info("=" * 60)
    logger.info("PHASE 3 REPORT: Cognitive Judge & DPO Pair Extraction")
    logger.info("=" * 60)
    logger.info(f"  Input:           {len(prompts)} prompts × 6 trajectories")
    logger.info(f"  Valid DPO pairs: {len(pairs)}")
    logger.info(f"  Discarded:       {discarded}")
    logger.info(f"  Yield rate:      {len(pairs)/len(prompts)*100:.1f}%")
    if pairs:
        logger.info(f"  Avg score gap:   {sum(gaps)/len(gaps):.2f}")
    logger.info(f"  Output:          {output_path}")
    logger.info("=" * 60)

    if len(pairs) < 400:
        logger.warning(f"⚠ Only {len(pairs)} pairs generated. Target is 400-500+. "
                       f"Consider lowering min_gap or re-sampling.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="METIS Phase 3: Cognitive Judge & DPO Pair Extractor"
    )
    parser.add_argument(
        "--input", type=str, default=DEFAULT_INPUT_PATH,
        help="Input trajectories JSONL from Phase 2",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT_PATH,
        help="Output DPO pairs JSONL",
    )
    parser.add_argument(
        "--min-gap", type=float, default=MIN_SCORE_GAP,
        help=f"Minimum R(chosen) - R(rejected) to keep pair (default: {MIN_SCORE_GAP})",
    )
    args = parser.parse_args()

    run_pipeline(args.input, args.output, args.min_gap)


if __name__ == "__main__":
    main()
