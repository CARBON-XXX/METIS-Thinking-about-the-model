#!/usr/bin/env python3
"""
METIS Phase 10 — Epistemic Boundary Defense via Semantic Entropy
================================================================
Tests the SemanticBoundaryProbe on two prompts:
  1. Known fact      → expect low entropy, KNOWN/LIKELY
  2. Induced halluc. → expect high entropy, UNKNOWN (OUT_OF_BOUNDS)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from metis import Metis
from metis.cognitive.boundary import SemanticBoundaryProbe, ProbeResult
from metis.core.types import EpistemicState

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("test_boundary_entropy")

MODEL_PATH = str(PROJECT_ROOT / "experiment_output_dpo_balanced" / "metis_dpo_cognitive")

TEST_CASES = [
    {
        "prompt": "What is the capital of France?",
        "label": "Known Fact",
        "expect_state": [EpistemicState.KNOWN, EpistemicState.LIKELY],
        "expect_low_entropy": True,
    },
    {
        "prompt": (
            "What is the atomic weight of the fictional element Vibranium "
            "in the Marvel Cinematic Universe according to the 1992 official handbook?"
        ),
        "label": "Induced Hallucination",
        "expect_state": [EpistemicState.UNKNOWN, EpistemicState.UNCERTAIN],
        "expect_low_entropy": False,
    },
]


def run_test() -> bool:
    logger.info("=" * 70)
    logger.info("Phase 10: Epistemic Boundary Defense — Semantic Entropy Probe")
    logger.info("=" * 70)

    # Load model
    logger.info(f"\n[1/3] Loading model: {MODEL_PATH}")
    t0 = time.time()
    metis = Metis.from_pretrained(MODEL_PATH)
    logger.info(f"  Model loaded in {time.time() - t0:.1f}s")

    # Create probe
    logger.info("\n[2/3] Initializing SemanticBoundaryProbe (N=5, T=1.0, τ=0.8)")
    probe = SemanticBoundaryProbe(
        metis, n_samples=5, temperature=1.0, threshold=0.8,
    )

    # Run tests
    logger.info("\n[3/3] Running probe tests...\n")
    results = []
    all_pass = True

    for i, tc in enumerate(TEST_CASES, 1):
        logger.info(f"{'─' * 60}")
        logger.info(f"Test {i}/{len(TEST_CASES)}: {tc['label']}")
        logger.info(f"  Prompt: \"{tc['prompt'][:90]}{'...' if len(tc['prompt']) > 90 else ''}\"")

        t1 = time.time()
        pr: ProbeResult = probe.evaluate_uncertainty(tc["prompt"])
        elapsed = time.time() - t1

        state_ok = pr.epistemic_state in tc["expect_state"]
        entropy_ok = (
            (pr.semantic_entropy < 0.8) if tc["expect_low_entropy"]
            else (pr.semantic_entropy >= 0.4)
        )
        passed = state_ok and entropy_ok
        if not passed:
            all_pass = False

        status = "PASS ✓" if passed else "FAIL ✗"
        logger.info(f"  Semantic Entropy H = {pr.semantic_entropy:.4f}")
        logger.info(f"  Clusters: {pr.n_clusters} (sizes={pr.cluster_sizes})")
        logger.info(f"  Epistemic State: {pr.epistemic_state.value} "
                     f"{'✓' if state_ok else '✗ expected: ' + str([s.value for s in tc['expect_state']])}")
        logger.info(f"  Samples:")
        for j, s in enumerate(pr.samples, 1):
            logger.info(f"    [{j}] {s[:120]}{'...' if len(s) > 120 else ''}")
        logger.info(f"  Elapsed: {elapsed:.1f}s")
        logger.info(f"  [{status}]")

        results.append({
            "label": tc["label"],
            "prompt": tc["prompt"],
            "semantic_entropy": round(pr.semantic_entropy, 4),
            "n_clusters": pr.n_clusters,
            "cluster_sizes": pr.cluster_sizes,
            "epistemic_state": pr.epistemic_state.value,
            "samples": pr.samples,
            "elapsed_s": round(elapsed, 1),
            "passed": passed,
        })

    # Summary
    print(f"\n{'=' * 70}")
    print("SEMANTIC ENTROPY PROBE — SUMMARY")
    print(f"{'=' * 70}")
    for r in results:
        tag = "PASS ✓" if r["passed"] else "FAIL ✗"
        print(f"  [{tag}] {r['label']}: H={r['semantic_entropy']:.4f}, "
              f"clusters={r['n_clusters']}, state={r['epistemic_state']}")
    n_pass = sum(1 for r in results if r["passed"])
    print(f"\n  Result: {n_pass}/{len(results)} passed")

    print(f"\n{'─' * 70}")
    print("STRUCTURED RESULTS (JSON)")
    print(f"{'─' * 70}")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print("=" * 70)

    return all_pass


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
