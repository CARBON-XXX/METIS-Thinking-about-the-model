#!/usr/bin/env python3
"""
Phase 24.20: Qwen-32B Cross-Model Entropy Scaling Probe

Runs 20 questions (10 simple + 10 complex) through Qwen2.5-32B-Instruct,
extracts token-level entropy from logprobs, computes EWMA + CUSUM,
and validates that routing bifurcation remains stable at 32B parameter scale.

Usage:
    HF_HUB_OFFLINE=1 TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas \
        python tools/qwen32b_probe.py
"""

import asyncio
import json
import math
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

# ── Configuration ──────────────────────────────────────────────────────
MODEL: str = "/home/metis/.cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct"
GPU_MEM_UTIL: float = 0.85
OUTPUT_PATH: Path = Path(__file__).parent / "32b_scaling_results.json"

# METIS signal processing parameters (must match metis/core/controller.py)
EWMA_ALPHA: float = 0.3
CUSUM_DRIFT: float = 0.5
CUSUM_THRESHOLD: float = 2.0   # Siegmund critical value

# Routing thresholds (calibrated from 7B benchmarks)
FAST_CEILING: float = 0.8      # mean EWMA < this → FAST
DEEP_FLOOR: float = 1.2        # mean EWMA > this → DEEP

# ── Prompt Sets ────────────────────────────────────────────────────────
SIMPLE_QUESTIONS: list[str] = [
    "What is the capital of France?",
    "Who invented the telephone?",
    "What is the chemical symbol for gold?",
    "How many continents are there?",
    "What is the boiling point of water in Celsius?",
    "Who wrote Romeo and Juliet?",
    "What is the square root of 144?",
    "Name the largest ocean in the world.",
    "What is the atomic number of carbon?",
    "What language is spoken in Brazil?",
]

COMPLEX_QUESTIONS: list[str] = [
    "Explain step by step how to solve a system of two linear equations using substitution.",
    "Derive the quadratic formula from ax^2 + bx + c = 0.",
    "Explain how transformers work in deep learning, including the self-attention mechanism.",
    "Describe the process of photosynthesis at the molecular level.",
    "Compare the time complexity of quicksort, mergesort, and heapsort with proofs.",
    "Explain the concept of quantum entanglement and its implications for information theory.",
    "Derive Bayes' theorem from first principles and provide three real-world applications.",
    "Explain the mathematics behind principal component analysis (PCA).",
    "Design an algorithm to efficiently find the median of a data stream in O(log n).",
    "Explain the halting problem and prove why it is undecidable using diagonalization.",
]


# ── Entropy Computation ────────────────────────────────────────────────

def compute_entropy_from_logprobs(logprobs_dict: dict[int, Any]) -> float:
    """Compute Shannon entropy H(p) from vLLM's top-k logprob dict."""
    if not logprobs_dict:
        return 0.0
    probs: list[float] = []
    for lp in logprobs_dict.values():
        val = lp.logprob if hasattr(lp, "logprob") else lp
        if val is not None:
            probs.append(math.exp(val))
    if not probs:
        return 0.0
    total = sum(probs)
    if total <= 0:
        return 0.0
    probs = [p / total for p in probs]
    return -sum(p * math.log(p + 1e-15) for p in probs if p > 0)


# ── EWMA + CUSUM Signal Processing ────────────────────────────────────

def compute_ewma_cusum(
    entropies: list[float],
) -> dict[str, Any]:
    """Replicate METIS EWMA low-pass filter + Siegmund CUSUM detector."""
    if not entropies:
        return {"ewma": [], "cusum_pos": [], "cusum_neg": [],
                "cusum_triggered": False, "mean_ewma": 0.0, "max_cusum": 0.0}

    ewma: list[float] = []
    cusum_pos: list[float] = []
    cusum_neg: list[float] = []

    e: float = entropies[0]
    cp: float = 0.0
    cn: float = 0.0

    for h in entropies:
        e = EWMA_ALPHA * h + (1 - EWMA_ALPHA) * e
        ewma.append(round(e, 6))

        # Siegmund CUSUM on first derivative of EWMA
        delta = h - e
        cp = max(0.0, cp + delta - CUSUM_DRIFT)
        cn = max(0.0, cn - delta - CUSUM_DRIFT)
        cusum_pos.append(round(cp, 6))
        cusum_neg.append(round(cn, 6))

    triggered = any(c > CUSUM_THRESHOLD for c in cusum_pos) or \
                any(c > CUSUM_THRESHOLD for c in cusum_neg)

    return {
        "ewma": ewma,
        "cusum_pos": cusum_pos,
        "cusum_neg": cusum_neg,
        "cusum_triggered": triggered,
        "mean_ewma": round(float(np.mean(ewma)), 6),
        "std_ewma": round(float(np.std(ewma)), 6),
        "max_cusum": round(float(max(max(cusum_pos), max(cusum_neg))), 6),
    }


def classify_route(mean_ewma: float, cusum_triggered: bool) -> str:
    """Determine cognitive route from EWMA mean and CUSUM state."""
    if mean_ewma < FAST_CEILING and not cusum_triggered:
        return "FAST"
    elif mean_ewma > DEEP_FLOOR or cusum_triggered:
        return "DEEP"
    else:
        return "NORMAL"


# ── Request Processing ─────────────────────────────────────────────────

async def probe_question(
    engine: AsyncLLMEngine,
    prompt: str,
    request_id: str,
) -> dict[str, Any]:
    """Run a single question, extract per-token entropy, compute signals."""
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
        logprobs=20,  # top-20 logprobs per token for entropy estimation
    )

    token_entropies: list[float] = []
    generated_text: str = ""
    total_tokens: int = 0
    seen_tokens: int = 0

    start = time.perf_counter()
    async for output in engine.generate(prompt, sampling_params, request_id):
        if output.outputs:
            out = output.outputs[0]
            generated_text = out.text
            total_tokens = len(out.token_ids)
            if out.logprobs:
                for lp_dict in out.logprobs[seen_tokens:]:
                    if lp_dict:
                        h = compute_entropy_from_logprobs(lp_dict)
                        token_entropies.append(round(h, 6))
                seen_tokens = len(out.logprobs)
    latency = time.perf_counter() - start

    # Compute EWMA + CUSUM
    signals = compute_ewma_cusum(token_entropies)
    route = classify_route(signals["mean_ewma"], signals["cusum_triggered"])

    return {
        "prompt": prompt,
        "generated_tokens": total_tokens,
        "latency_s": round(latency, 3),
        "raw_entropies": token_entropies[:50],  # first 50 for compact JSON
        "mean_raw_entropy": round(float(np.mean(token_entropies)), 6) if token_entropies else 0.0,
        "signals": {
            "mean_ewma": signals["mean_ewma"],
            "std_ewma": signals["std_ewma"],
            "max_cusum": signals["max_cusum"],
            "cusum_triggered": signals["cusum_triggered"],
        },
        "route": route,
        "generated_text_preview": generated_text[:200],
    }


# ── Warmup ─────────────────────────────────────────────────────────────

async def warmup(engine: AsyncLLMEngine) -> None:
    """Warmup with 3 requests to stabilize GPU clocks."""
    print("  Warming up...", end="", flush=True)
    params = SamplingParams(temperature=0.0, max_tokens=16)
    for i in range(3):
        async for _ in engine.generate("Hello", params, f"warmup-{i}"):
            pass
    print(" done.")


# ── Main ───────────────────────────────────────────────────────────────

async def main() -> None:
    print("=" * 70)
    print("Phase 24.20: Qwen-32B Cross-Model Entropy Scaling Probe")
    print(f"  Model: {MODEL}")
    print(f"  Questions: 10 simple + 10 complex = 20")
    print(f"  EWMA α={EWMA_ALPHA}, CUSUM drift={CUSUM_DRIFT}, τ={CUSUM_THRESHOLD}")
    print("=" * 70)

    # Initialize engine
    print("\n[1/3] Initializing vLLM AsyncLLMEngine for 32B model...")
    engine_args = AsyncEngineArgs(
        model=MODEL,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_num_seqs=64,
        trust_remote_code=True,
        disable_log_stats=True,
        max_model_len=2048,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Warmup
    print("[2/3] Warmup phase...")
    await warmup(engine)

    # Probe
    print("[3/3] Running 20-question probe...\n")
    all_results: list[dict[str, Any]] = []

    for i, q in enumerate(SIMPLE_QUESTIONS):
        rid = f"simple-{i}-{uuid.uuid4().hex[:6]}"
        print(f"  [{i+1:2d}/20] SIMPLE: {q[:60]}...", end="", flush=True)
        result = await probe_question(engine, q, rid)
        result["category"] = "simple"
        all_results.append(result)
        print(f" → {result['route']} (EWMA={result['signals']['mean_ewma']:.4f})")

    for i, q in enumerate(COMPLEX_QUESTIONS):
        rid = f"complex-{i}-{uuid.uuid4().hex[:6]}"
        print(f"  [{i+11:2d}/20] COMPLEX: {q[:58]}...", end="", flush=True)
        result = await probe_question(engine, q, rid)
        result["category"] = "complex"
        all_results.append(result)
        print(f" → {result['route']} (EWMA={result['signals']['mean_ewma']:.4f})")

    # Aggregate statistics
    simple_results = [r for r in all_results if r["category"] == "simple"]
    complex_results = [r for r in all_results if r["category"] == "complex"]

    simple_ewma = [r["signals"]["mean_ewma"] for r in simple_results]
    complex_ewma = [r["signals"]["mean_ewma"] for r in complex_results]
    simple_routes = [r["route"] for r in simple_results]
    complex_routes = [r["route"] for r in complex_results]

    summary: dict[str, Any] = {
        "simple": {
            "mean_ewma": round(float(np.mean(simple_ewma)), 6),
            "std_ewma": round(float(np.std(simple_ewma)), 6),
            "route_distribution": {
                "FAST": simple_routes.count("FAST"),
                "NORMAL": simple_routes.count("NORMAL"),
                "DEEP": simple_routes.count("DEEP"),
            },
        },
        "complex": {
            "mean_ewma": round(float(np.mean(complex_ewma)), 6),
            "std_ewma": round(float(np.std(complex_ewma)), 6),
            "route_distribution": {
                "FAST": complex_routes.count("FAST"),
                "NORMAL": complex_routes.count("NORMAL"),
                "DEEP": complex_routes.count("DEEP"),
            },
        },
        "bifurcation_stable": (
            float(np.mean(simple_ewma)) < FAST_CEILING and
            float(np.mean(complex_ewma)) > DEEP_FLOOR
        ),
        "ewma_gap": round(float(np.mean(complex_ewma)) - float(np.mean(simple_ewma)), 6),
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Simple  → mean EWMA: {summary['simple']['mean_ewma']:.4f} ± {summary['simple']['std_ewma']:.4f}")
    print(f"            Routes: {summary['simple']['route_distribution']}")
    print(f"  Complex → mean EWMA: {summary['complex']['mean_ewma']:.4f} ± {summary['complex']['std_ewma']:.4f}")
    print(f"            Routes: {summary['complex']['route_distribution']}")
    print(f"  EWMA gap (complex - simple): {summary['ewma_gap']:.4f}")
    print(f"  Bifurcation stable: {summary['bifurcation_stable']}")

    # Save
    output = {
        "metadata": {
            "model": "Qwen/Qwen2.5-32B-Instruct",
            "parameter_count": "32B",
            "gpu_memory_utilization": GPU_MEM_UTIL,
            "ewma_alpha": EWMA_ALPHA,
            "cusum_drift": CUSUM_DRIFT,
            "cusum_threshold": CUSUM_THRESHOLD,
            "fast_ceiling": FAST_CEILING,
            "deep_floor": DEEP_FLOOR,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "summary": summary,
        "results": all_results,
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
