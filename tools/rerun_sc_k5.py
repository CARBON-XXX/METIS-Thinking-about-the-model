#!/usr/bin/env python3
"""
Re-run ONLY the Self-Consistency k=5 baseline with semantic equivalence
clustering for majority voting. Updates pareto.json in place and
re-renders Figure 1.

Usage:
    python tools/rerun_sc_k5.py [--model PATH]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse utilities from phase24
from tools.phase24_academic_benchmarks import (
    _build_mixed_dataset,
    _fmt,
    semantic_majority_vote,
    SEED,
)

OUTPUT_DIR = PROJECT_ROOT / "paper" / "data"
PARETO_JSON = OUTPUT_DIR / "pareto.json"

DEFAULT_MODEL = str(
    PROJECT_ROOT / "experiment_output_dpo_balanced" / "metis_dpo_cognitive"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run SC k=5 with semantic voting")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model path")
    args = parser.parse_args()

    # ── Fix Blackwell compatibility ──
    os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda-13.0/bin/ptxas")

    from vllm import LLM, SamplingParams  # type: ignore

    print("=" * 60)
    print("  SC k=5 RE-EVALUATION — Semantic Equivalence Clustering")
    print("=" * 60)

    # Build the SAME dataset (deterministic seed)
    random.seed(SEED)
    dataset = _build_mixed_dataset(100)

    # Load model
    print(f"Loading model: {args.model}")
    gc.collect()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        max_model_len=2048,
        seed=SEED,
    )

    # Run SC k=5
    SC_K = 5
    print(f"Running Self-Consistency k={SC_K} on {len(dataset)} questions...")
    t0 = time.time()
    sc_sys = "You are a helpful assistant. Think step by step and give your final answer."
    outputs = llm.generate(
        [_fmt(it["question"], sys=sc_sys) for it in dataset],
        SamplingParams(n=SC_K, max_tokens=512, temperature=0.7, top_p=0.95),
    )
    wall_time = time.time() - t0

    correct, total_tok = 0, 0
    per_question: List[Dict[str, Any]] = []
    for o, it in zip(outputs, dataset):
        completions: List[str] = []
        q_tok = 0
        for comp in o.outputs:
            q_tok += len(comp.token_ids)
            completions.append(comp.text)
        total_tok += q_tok
        winner, ok = semantic_majority_vote(completions, it["type"], it["answer"])
        correct += int(ok)
        per_question.append({
            "question": it["question"],
            "type": it["type"],
            "gold": it["answer"],
            "winner": winner,
            "correct": ok,
            "tokens": q_tok,
        })

    acc = correct / len(dataset)
    avg_tok = total_tok / len(dataset)
    throughput = total_tok / max(0.01, wall_time)

    print(f"\n{'=' * 60}")
    print(f"  SC k=5 RESULT: {correct}/{len(dataset)} = {acc:.1%}")
    print(f"  Avg tokens: {avg_tok:.1f}  Wall: {wall_time:.1f}s  Throughput: {throughput:.1f} tok/s")
    print(f"{'=' * 60}")

    # Breakdown by type
    math_q = [q for q in per_question if q["type"] == "math"]
    qa_q = [q for q in per_question if q["type"] == "qa"]
    math_acc = sum(q["correct"] for q in math_q) / max(1, len(math_q))
    qa_acc = sum(q["correct"] for q in qa_q) / max(1, len(qa_q))
    print(f"  Math: {sum(q['correct'] for q in math_q)}/{len(math_q)} = {math_acc:.1%}")
    print(f"  QA:   {sum(q['correct'] for q in qa_q)}/{len(qa_q)} = {qa_acc:.1%}")

    # Show QA misses
    qa_misses = [q for q in qa_q if not q["correct"]]
    if qa_misses:
        print(f"\n  QA misses ({len(qa_misses)}):")
        for q in qa_misses[:10]:
            print(f"    Q: {q['question'][:60]}  Gold: {q['gold']}  Got: {q['winner'][:40]}")

    # ── Update pareto.json ──
    if PARETO_JSON.exists():
        pareto = json.load(open(PARETO_JSON))
    else:
        print("WARNING: pareto.json not found, creating minimal structure")
        pareto = {"strategies": {}}

    old_acc = pareto.get("strategies", {}).get("self_consistency_k5", {}).get("accuracy", "N/A")
    pareto["strategies"]["self_consistency_k5"] = {
        "accuracy": round(acc, 4),
        "correct": correct,
        "total": len(dataset),
        "avg_tokens": round(avg_tok, 1),
        "total_tokens": total_tok,
        "k_samples": SC_K,
        "wall_time_s": round(wall_time, 2),
        "throughput_tok_s": round(throughput, 1),
        "note": f"Semantic equivalence clustering for majority vote (n={SC_K})",
    }

    with open(PARETO_JSON, "w") as f:
        json.dump(pareto, f, indent=2, ensure_ascii=False)
    print(f"\n  Updated: {PARETO_JSON}")
    print(f"  Old accuracy: {old_acc}  →  New: {acc:.4f}")

    # ── Re-render Figure 1 ──
    print("\n  Re-rendering Figure 1...")
    from tools.render_pareto_frontier import main as render_fig1
    render_fig1()

    # Cleanup
    del llm
    gc.collect()

    print(f"\n{'=' * 60}")
    print(f"  DONE. SC k=5 accuracy: {old_acc} → {acc:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
