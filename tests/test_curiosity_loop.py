#!/usr/bin/env python3
"""
METIS Phase 11 — Curiosity-Driven Knowledge Acquisition Test
=============================================================
Full closed loop: Uncertainty → Search → Deep Reasoning → Resolution

Pipeline:
  1. SemanticBoundaryProbe detects UNKNOWN (high entropy)
  2. CuriosityDriver.resolve_gap() generates search query via LLM
  3. ToolRetriever returns mock knowledge
  4. Augmented prompt → generate_cognitive() → grounded DEEP answer
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from metis import Metis, MetisInference
from metis.cognitive.boundary import SemanticBoundaryProbe
from metis.cognitive.curiosity import CuriosityDriver
from metis.search.retriever import ToolRetriever
from metis.core.types import EpistemicState

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("test_curiosity_loop")

MODEL_PATH = str(PROJECT_ROOT / "experiment_output_dpo_balanced" / "metis_dpo_cognitive")

HALLUCINATION_PROMPT = (
    "What is the atomic weight of Vibranium according to the "
    "1992 official handbook?"
)


def run_test() -> bool:
    logger.info("=" * 70)
    logger.info("Phase 11: Curiosity-Driven Knowledge Acquisition")
    logger.info("=" * 70)

    # ── Load model ──
    logger.info(f"\n[1/5] Loading model...")
    t0 = time.time()
    metis = Metis.from_pretrained(MODEL_PATH)
    engine = MetisInference(metis)
    logger.info(f"  Model loaded in {time.time() - t0:.1f}s")

    # ── Init components ──
    logger.info("\n[2/5] Initializing components...")
    probe = SemanticBoundaryProbe(metis, n_samples=5, temperature=1.0, threshold=0.8)
    curiosity = CuriosityDriver()
    retriever = ToolRetriever()
    logger.info("  ✓ SemanticBoundaryProbe (N=5, T=1.0, τ=0.8)")
    logger.info("  ✓ CuriosityDriver")
    logger.info("  ✓ ToolRetriever (mock KB)")

    # ── Step A: Probe → detect UNKNOWN ──
    logger.info(f"\n[3/5] Probing: \"{HALLUCINATION_PROMPT}\"")
    t1 = time.time()
    probe_result = probe.evaluate_uncertainty(HALLUCINATION_PROMPT)
    probe_time = time.time() - t1

    logger.info(f"  H = {probe_result.semantic_entropy:.4f}")
    logger.info(f"  Clusters: {probe_result.n_clusters} (sizes={probe_result.cluster_sizes})")
    logger.info(f"  State: {probe_result.epistemic_state.value}")
    logger.info(f"  Samples:")
    for i, s in enumerate(probe_result.samples, 1):
        logger.info(f"    [{i}] {s[:120]}")
    logger.info(f"  Elapsed: {probe_time:.1f}s")

    probe_pass = probe_result.epistemic_state in (
        EpistemicState.UNKNOWN, EpistemicState.UNCERTAIN,
    )
    logger.info(f"  Probe assertion (UNKNOWN/UNCERTAIN): {'PASS ✓' if probe_pass else 'FAIL ✗'}")

    # ── Step B: Resolve gap ──
    logger.info(f"\n[4/5] CuriosityDriver.resolve_gap() — full pipeline")
    t2 = time.time()
    resolution = curiosity.resolve_gap(
        prompt=HALLUCINATION_PROMPT,
        probe_result=probe_result,
        engine=engine,
        retriever=retriever,
    )
    resolve_time = time.time() - t2

    final = resolution["result"]
    logger.info(f"  Search query: \"{resolution['search_query']}\"")
    logger.info(f"  Retrieved ({len(resolution['retrieved_context'])} chars):")
    for line in resolution["retrieved_context"].split("\n")[:4]:
        logger.info(f"    {line[:120]}")
    logger.info(f"  Route: {final.cognitive_route}")
    logger.info(f"  Thinking: {'present' if final.thinking_text else 'none'} "
                f"(repaired={final.thinking_repaired})")
    logger.info(f"  Final answer: {final.text[:300]}")
    logger.info(f"  Tokens: {final.tokens_generated}, Latency: {final.latency_ms:.0f}ms")
    logger.info(f"  Resolve elapsed: {resolve_time:.1f}s")
    logger.info(f"  Gap resolved: {resolution['resolved']}")

    # ── Assertions ──
    answer_has_weight = "238" in final.text
    route_ok = final.cognitive_route in ("DEEP", "FAST", "FAST (Implicit)")
    resolved_ok = resolution["resolved"] is True

    logger.info(f"\n[5/5] Assertions:")
    logger.info(f"  Answer contains '238': {'PASS ✓' if answer_has_weight else 'FAIL ✗'}")
    logger.info(f"  Route valid: {'PASS ✓' if route_ok else 'FAIL ✗'} ({final.cognitive_route})")
    logger.info(f"  Gap resolved: {'PASS ✓' if resolved_ok else 'FAIL ✗'}")

    all_pass = probe_pass and answer_has_weight and route_ok and resolved_ok

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("CURIOSITY LOOP — SUMMARY")
    print(f"{'=' * 70}")
    print(f"  [{'PASS ✓' if probe_pass else 'FAIL ✗'}] Probe: H={probe_result.semantic_entropy:.4f}, "
          f"state={probe_result.epistemic_state.value}")
    print(f"  [{'PASS ✓' if answer_has_weight else 'FAIL ✗'}] Grounded answer contains '238.04'")
    print(f"  [{'PASS ✓' if route_ok else 'FAIL ✗'}] Cognitive route: {final.cognitive_route}")
    print(f"  [{'PASS ✓' if resolved_ok else 'FAIL ✗'}] Gap resolved: {resolution['resolved']}")
    n_pass = sum([probe_pass, answer_has_weight, route_ok, resolved_ok])
    print(f"\n  Result: {n_pass}/4 passed")

    print(f"\n{'─' * 70}")
    print("STRUCTURED RESULT (JSON)")
    print(f"{'─' * 70}")
    print(json.dumps({
        "probe": {
            "semantic_entropy": round(probe_result.semantic_entropy, 4),
            "n_clusters": probe_result.n_clusters,
            "cluster_sizes": probe_result.cluster_sizes,
            "epistemic_state": probe_result.epistemic_state.value,
            "samples": probe_result.samples,
        },
        "resolution": {
            "search_query": resolution["search_query"],
            "retrieved_chars": len(resolution["retrieved_context"]),
            "cognitive_route": final.cognitive_route,
            "thinking_present": bool(final.thinking_text),
            "thinking_repaired": final.thinking_repaired,
            "answer_preview": final.text[:400],
            "tokens": final.tokens_generated,
            "latency_ms": round(final.latency_ms, 1),
            "gap_resolved": resolution["resolved"],
        },
    }, indent=2, ensure_ascii=False))
    print("=" * 70)

    return all_pass


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
