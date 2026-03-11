#!/usr/bin/env python3
"""
METIS Phase 9 — Core Infrastructure Integration Test
=====================================================
Validates that the METIS core engine (Metis.from_pretrained → MetisInference
→ generate_cognitive) correctly loads the DPO model and applies the
deterministic cognitive routing state machine.

Test Matrix:
  1. "What is 5 + 7?"                     → FAST (Implicit) — simple arithmetic
  2. "Solve: 3x + 2y = 16, x - y = 2"    → DEEP            — system of equations
  3. "What is the capital of France?"      → FAST             — factual recall
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from metis import Metis, MetisInference, Decision, InferenceResult

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("test_core_integration")

# ─────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────

MODEL_PATH = str(PROJECT_ROOT / "experiment_output_dpo_balanced" / "metis_dpo_cognitive")

TEST_QUERIES = [
    {
        "prompt": "What is 5 + 7?",
        "expected_routes": ["FAST", "FAST (Implicit)"],
        "expected_decision": Decision.FAST,
        "expect_thinking": False,
        "description": "Simple arithmetic → FAST / Implicit FAST",
    },
    {
        "prompt": "Solve: 3x + 2y = 16, x - y = 2",
        "expected_routes": ["DEEP"],
        "expected_decision": Decision.DEEP,
        "expect_thinking": True,
        "description": "System of equations → DEEP with thinking",
    },
    {
        "prompt": "What is the capital of France?",
        "expected_routes": ["FAST", "FAST (Implicit)"],
        "expected_decision": Decision.FAST,
        "expect_thinking": False,
        "description": "Factual recall → FAST / Implicit FAST",
    },
]


# ─────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────


def run_integration_test() -> bool:
    """Run the full integration test suite.

    Returns:
        True if all critical assertions pass.
    """
    logger.info("=" * 70)
    logger.info("METIS Phase 9 — Core Infrastructure Integration Test")
    logger.info("=" * 70)

    # ── Step 1: Load model via Metis.from_pretrained() ──
    logger.info(f"\n[1/4] Loading model via Metis.from_pretrained('{MODEL_PATH}')...")
    t0 = time.time()
    metis = Metis.from_pretrained(MODEL_PATH)
    load_time = time.time() - t0
    logger.info(f"  ✓ Model loaded in {load_time:.1f}s")

    # ── Step 2: Verify Metis state ──
    logger.info("\n[2/4] Verifying Metis internal state...")
    assert metis.model is not None, "Model must be attached"
    assert metis.tokenizer is not None, "Tokenizer must be attached"
    logger.info(f"  ✓ model type: {type(metis.model).__name__}")
    logger.info(f"  ✓ tokenizer type: {type(metis.tokenizer).__name__}")
    logger.info(f"  ✓ stats: {metis.stats}")

    # ── Step 3: Create MetisInference engine ──
    logger.info("\n[3/4] Creating MetisInference engine...")
    engine = MetisInference(metis)
    logger.info("  ✓ MetisInference initialized")

    # ── Step 4: Run test queries ──
    logger.info("\n[4/4] Running cognitive generation tests...\n")

    results: list[dict] = []
    all_pass = True

    for i, tq in enumerate(TEST_QUERIES, 1):
        logger.info(f"─── Test {i}/{len(TEST_QUERIES)}: {tq['description']} ───")
        logger.info(f"  Prompt: \"{tq['prompt']}\"")

        result: InferenceResult = engine.generate_cognitive(tq["prompt"])

        # ── Assertions ──
        route_ok = result.cognitive_route in tq["expected_routes"]
        decision_ok = result.final_decision == tq["expected_decision"]
        thinking_ok = (
            (bool(result.thinking_text) == tq["expect_thinking"])
            if tq["expect_thinking"]
            else True  # FAST queries: don't penalize if model emits no thinking
        )
        has_answer = len(result.text.strip()) > 0

        passed = route_ok and decision_ok and has_answer
        if not passed:
            all_pass = False

        status = "PASS ✓" if passed else "FAIL ✗"
        logger.info(f"  Route:     {result.cognitive_route} {'✓' if route_ok else '✗ expected: ' + str(tq['expected_routes'])}")
        logger.info(f"  Decision:  {result.final_decision} {'✓' if decision_ok else '✗ expected: ' + str(tq['expected_decision'])}")
        logger.info(f"  Thinking:  {'present' if result.thinking_text else 'none'} (repaired={result.thinking_repaired})")
        logger.info(f"  Answer:    {result.text[:200]}{'...' if len(result.text) > 200 else ''}")
        logger.info(f"  Latency:   {result.latency_ms:.0f}ms")
        logger.info(f"  Tokens:    {result.tokens_generated}")
        logger.info(f"  [{status}]\n")

        results.append({
            "prompt": tq["prompt"],
            "route": result.cognitive_route,
            "decision": str(result.final_decision),
            "thinking_present": bool(result.thinking_text),
            "thinking_repaired": result.thinking_repaired,
            "answer_preview": result.text[:300],
            "latency_ms": round(result.latency_ms, 1),
            "tokens": result.tokens_generated,
            "passed": passed,
        })

    # ── Summary ──
    print("\n" + "=" * 70)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 70)
    for i, r in enumerate(results, 1):
        status = "PASS ✓" if r["passed"] else "FAIL ✗"
        print(f"  [{status}] Query {i}: route={r['route']}, "
              f"decision={r['decision']}, "
              f"repaired={r['thinking_repaired']}, "
              f"latency={r['latency_ms']:.0f}ms")

    n_pass = sum(1 for r in results if r["passed"])
    print(f"\n  Result: {n_pass}/{len(results)} passed")

    # Pretty JSON
    print(f"\n{'─' * 70}")
    print("STRUCTURED RESULTS (JSON)")
    print(f"{'─' * 70}")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print("=" * 70)

    return all_pass


if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)
