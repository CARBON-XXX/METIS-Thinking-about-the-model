#!/usr/bin/env python3
"""Phase 14 Diagnostic — Prove short-circuit works + dump raw output for failed GSM8K.

Usage:
    python tools/phase14_diagnostic.py
"""
import sys
import os
import time
import logging

# Force offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("diag")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

METIS_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "experiment_output_dpo_balanced", "metis_dpo_cognitive"
)

# ── Load METIS model ──
logger.info(f"Loading METIS model: {METIS_MODEL_PATH}")
from metis import Metis
from metis.cognitive.metacognition import MetacognitiveOrchestrator
from metis.search.retriever import ToolRetriever

t0 = time.time()
metis = Metis.from_pretrained(METIS_MODEL_PATH)
retriever = ToolRetriever(force_mock=False)
orch = MetacognitiveOrchestrator(metis, retriever=retriever)
logger.info(f"Loaded in {time.time() - t0:.1f}s")


def run_test(label: str, prompt: str, expected_answer: str = "") -> None:
    """Run a single prompt and dump full diagnostics."""
    print(f"\n{'='*72}")
    print(f"  TEST: {label}")
    print(f"  PROMPT: {prompt}")
    print(f"{'='*72}")

    t0 = time.perf_counter()
    resp = orch.process_query(prompt)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"  cognitive_route:   {resp.cognitive_route}")
    print(f"  semantic_entropy:  {resp.semantic_entropy}")
    print(f"  epistemic_state:   {resp.epistemic_state}")
    print(f"  searched:          {resp.searched}")
    print(f"  tokens_generated:  {resp.tokens_generated}")
    print(f"  latency_ms:        {elapsed:.0f}")
    print(f"  trajectory:        {resp.trajectory}")
    print(f"  thinking_repaired: {resp.thinking_repaired}")
    print(f"  ---")
    print(f"  FINAL ANSWER (first 500 chars):")
    print(f"  {repr(resp.final_answer[:500])}")
    print(f"  ---")
    print(f"  THINKING TEXT (first 500 chars):")
    print(f"  {repr(resp.thinking_text[:500])}")
    print(f"  ---")
    print(f"  RAW OUTPUT (first 800 chars):")
    print(f"  {repr(resp.raw_output[:800])}")

    # Assertions for FAST short-circuit
    if "FAST" in resp.cognitive_route:
        assert resp.semantic_entropy is None, \
            f"FAIL: FAST route has semantic_entropy={resp.semantic_entropy} (expected None)"
        assert resp.epistemic_state == "short_circuit", \
            f"FAIL: FAST route has epistemic_state={resp.epistemic_state!r} (expected 'short_circuit')"
        assert resp.searched is False, \
            f"FAIL: FAST route has searched=True"
        print(f"\n  ✅ FAST SHORT-CIRCUIT VERIFIED: entropy=None, state=short_circuit, searched=False")
    else:
        print(f"\n  ℹ️  DEEP route — probe ran (H={resp.semantic_entropy})")

    if expected_answer:
        print(f"\n  EXPECTED ANSWER: {expected_answer}")
        # Check if expected answer appears in output
        full = (resp.final_answer + " " + resp.thinking_text).lower()
        if expected_answer.lower() in full:
            print(f"  ✅ Gold answer found in output")
        else:
            print(f"  ❌ Gold answer NOT found in output")


# ── TEST 1: Simple factual (should be FAST, no probe) ──
run_test(
    "SIMPLE — Capital of France (expect FAST short-circuit)",
    "What is the capital of France?",
    expected_answer="Paris",
)

# ── TEST 2: Complex GSM8K math (should be DEEP) ──
# This is gsm8k#1309 which failed in Phase 13 benchmark
run_test(
    "COMPLEX — GSM8K math (expect DEEP route)",
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    expected_answer="72",
)

print(f"\n{'='*72}")
print("  DIAGNOSTIC COMPLETE")
print(f"{'='*72}")
