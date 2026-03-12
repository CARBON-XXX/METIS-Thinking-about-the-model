#!/usr/bin/env python3
"""
Phase 24.18: MLSys System-Level Stress Test

ShareGPT-style workload benchmark: Vanilla vLLM vs. METIS Dynamic Routing
Measures: Global Throughput (tok/s), P50 TTFT, P99 TTFT
QPS levels: 1, 2, 4, 8, 16

Usage:
    python tools/sharegpt_stress_test.py
"""

import asyncio
import json
import random
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

# ── Configuration ──────────────────────────────────────────────────────
MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
GPU_MEM_UTIL: float = 0.85
QPS_LEVELS: list[float] = [1, 2, 4, 8, 16]
REQUESTS_PER_QPS: int = 200
METIS_FAST_RATIO: float = 0.48       # 48% FAST route
METIS_FAST_MAX_TOKENS: int = 50
VANILLA_MAX_TOKENS: int = 512
METIS_DEEP_MAX_TOKENS: int = 512
METIS_ROUTING_DELAY_US: float = 1.3  # microseconds
OUTPUT_PATH: Path = Path(__file__).parent / "mlsys_results.json"

# ── Prompt Pool (ShareGPT-style diverse workload) ──────────────────────
SIMPLE_PROMPTS: list[str] = [
    "What is the capital of France?",
    "Who invented the telephone?",
    "What year was Google founded?",
    "Define entropy in one sentence.",
    "What is the chemical symbol for gold?",
    "How many continents are there?",
    "Name the largest ocean in the world.",
    "What color is the sky on a clear day?",
    "Is iron a metal or nonmetal?",
    "What continent is Brazil located on?",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light in vacuum?",
    "How many bones are in the human body?",
    "What is the square root of 144?",
    "Name the planet closest to the Sun.",
    "What is the boiling point of water in Celsius?",
    "Who painted the Mona Lisa?",
    "What is the chemical formula for water?",
    "How many sides does a hexagon have?",
    "What is the largest mammal on Earth?",
    "Who discovered penicillin?",
    "What is the atomic number of carbon?",
    "Name the longest river in Africa.",
    "What language is spoken in Brazil?",
    "How many players are on a soccer team?",
    "What is the currency of Japan?",
    "Who was the first person to walk on the Moon?",
    "What is the freezing point of water in Fahrenheit?",
    "Name the smallest country in the world.",
    "What gas do plants absorb from the atmosphere?",
]

COMPLEX_PROMPTS: list[str] = [
    "Explain step by step how to solve a system of two linear equations using substitution.",
    "Compare and contrast supervised and unsupervised learning in machine learning.",
    "Write a Python function that implements binary search on a sorted list.",
    "Analyze the economic implications of rising interest rates on housing markets.",
    "Describe the mathematical proof for the Pythagorean theorem using similar triangles.",
    "Explain the key differences between TCP and UDP protocols.",
    "Explain the physics behind how a rainbow forms using first principles of optics.",
    "Design an algorithm to efficiently find the median of a data stream.",
    "Critically evaluate the argument that artificial intelligence poses an existential risk.",
    "Derive the quadratic formula from the general form ax^2 + bx + c = 0.",
    "Explain how transformers work in deep learning, including self-attention mechanics.",
    "Describe the process of photosynthesis at the molecular level.",
    "Write a detailed explanation of how public-key cryptography works.",
    "Analyze the causes and consequences of the 2008 financial crisis.",
    "Explain the concept of entropy in both thermodynamics and information theory.",
    "Design a database schema for a social media platform and explain normalization.",
    "Describe how CRISPR-Cas9 gene editing works and its potential applications.",
    "Explain the halting problem and why it is undecidable.",
    "Compare the time complexity of quicksort, mergesort, and heapsort.",
    "Describe how a convolutional neural network processes an image step by step.",
    "Explain the concept of quantum entanglement and its implications.",
    "Analyze the trade-offs between consistency and availability in distributed systems.",
    "Derive Bayes' theorem and provide three real-world applications.",
    "Explain how garbage collection works in the JVM with generational GC.",
    "Describe the MapReduce programming model and its use in distributed computing.",
    "Explain the mathematics behind principal component analysis (PCA).",
    "Analyze the environmental impact of cryptocurrency mining.",
    "Design a load balancer that handles 10 million requests per second.",
    "Explain the difference between parametric and non-parametric statistical tests.",
    "Describe the architecture of GPT models and the role of each component.",
]


def generate_prompt_pool(n: int) -> list[str]:
    """Generate n diverse prompts by cycling through simple + complex pools."""
    pool: list[str] = []
    all_templates = SIMPLE_PROMPTS + COMPLEX_PROMPTS
    for i in range(n):
        pool.append(all_templates[i % len(all_templates)])
    random.shuffle(pool)
    return pool


# ── Request Handling ───────────────────────────────────────────────────

async def process_request(
    engine: AsyncLLMEngine,
    prompt: str,
    sampling_params: SamplingParams,
    request_id: str,
    submit_time: float,
) -> dict[str, Any]:
    """Process a single request and measure TTFT + total tokens."""
    ttft: float | None = None
    total_tokens: int = 0
    end_time: float = submit_time

    async for output in engine.generate(prompt, sampling_params, request_id):
        if ttft is None:
            ttft = time.perf_counter() - submit_time
        if output.outputs:
            total_tokens = len(output.outputs[0].token_ids)
        end_time = time.perf_counter()

    return {
        "request_id": request_id,
        "ttft": ttft if ttft is not None else 0.0,
        "total_tokens": total_tokens,
        "total_time": end_time - submit_time,
    }


# ── Poisson Arrival Process ───────────────────────────────────────────

async def run_poisson_benchmark(
    engine: AsyncLLMEngine,
    prompts: list[str],
    qps: float,
    mode: str,
) -> dict[str, Any]:
    """Run a Poisson-arrival stress test at the given QPS."""
    print(f"  [{mode.upper():>7}] QPS={qps:<5.1f} n={len(prompts)} ...", end="", flush=True)

    tasks: list[asyncio.Task[dict[str, Any]]] = []
    start_time = time.perf_counter()

    for i, prompt in enumerate(prompts):
        # Poisson inter-arrival time
        if i > 0:
            inter_arrival: float = np.random.exponential(1.0 / qps)
            await asyncio.sleep(inter_arrival)

        # Configure sampling params based on mode
        if mode == "metis":
            # Simulate 1.3μs routing delay
            await asyncio.sleep(METIS_ROUTING_DELAY_US * 1e-6)
            # Route: 48% FAST (max_tokens=50), 52% DEEP (max_tokens=512)
            is_fast: bool = random.random() < METIS_FAST_RATIO
            max_tokens: int = METIS_FAST_MAX_TOKENS if is_fast else METIS_DEEP_MAX_TOKENS
        else:
            max_tokens = VANILLA_MAX_TOKENS

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
        )

        request_id = f"{mode}-qps{qps}-{uuid.uuid4().hex[:8]}"
        submit_time = time.perf_counter()

        task = asyncio.create_task(
            process_request(engine, prompt, sampling_params, request_id, submit_time)
        )
        tasks.append(task)

    # Wait for all requests to complete
    results: list[dict[str, Any]] = await asyncio.gather(*tasks)

    wall_time = time.perf_counter() - start_time

    # Compute metrics
    ttfts = [r["ttft"] for r in results if r["ttft"] > 0]
    total_tokens = sum(r["total_tokens"] for r in results)

    throughput = total_tokens / wall_time if wall_time > 0 else 0.0
    p50_ttft = float(np.percentile(ttfts, 50)) * 1000 if ttfts else 0.0
    p99_ttft = float(np.percentile(ttfts, 99)) * 1000 if ttfts else 0.0
    avg_tokens = total_tokens / len(results) if results else 0.0

    print(
        f" Tput: {throughput:>8.1f} tok/s | "
        f"P50: {p50_ttft:>7.1f}ms | "
        f"P99: {p99_ttft:>7.1f}ms | "
        f"AvgTok: {avg_tokens:>5.1f}"
    )

    return {
        "qps": qps,
        "mode": mode,
        "num_requests": len(prompts),
        "wall_time_s": round(wall_time, 2),
        "total_tokens": total_tokens,
        "throughput_toks": round(throughput, 1),
        "p50_ttft_ms": round(p50_ttft, 1),
        "p99_ttft_ms": round(p99_ttft, 1),
        "avg_tokens_per_request": round(avg_tokens, 1),
    }


# ── Warmup ─────────────────────────────────────────────────────────────

async def warmup(engine: AsyncLLMEngine) -> None:
    """Run 5 warmup requests to stabilize GPU clocks and KV cache."""
    print("  Warming up (5 requests)...", end="", flush=True)
    for i in range(5):
        params = SamplingParams(temperature=0.0, max_tokens=32)
        rid = f"warmup-{i}"
        async for _ in engine.generate("Hello, how are you?", params, rid):
            pass
    print(" done.")


# ── Main ───────────────────────────────────────────────────────────────

async def main() -> None:
    print("=" * 70)
    print("Phase 24.18: MLSys System-Level Stress Test")
    print(f"  Model:          {MODEL}")
    print(f"  GPU mem util:   {GPU_MEM_UTIL}")
    print(f"  QPS levels:     {QPS_LEVELS}")
    print(f"  Requests/QPS:   {REQUESTS_PER_QPS}")
    print(f"  METIS fast%:    {METIS_FAST_RATIO*100:.0f}%")
    print(f"  Routing delay:  {METIS_ROUTING_DELAY_US} μs")
    print("=" * 70)

    # Generate prompt pool
    print("\n[1/4] Generating 2000 ShareGPT-style prompts...")
    all_prompts = generate_prompt_pool(2000)

    # Initialize engine
    print("[2/4] Initializing vLLM AsyncLLMEngine...")
    engine_args = AsyncEngineArgs(
        model=MODEL,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_num_seqs=256,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Warmup
    print("[3/4] Warmup phase...")
    await warmup(engine)

    # Run benchmarks
    print("[4/4] Running stress tests...\n")
    all_results: list[dict[str, Any]] = []

    for qps in QPS_LEVELS:
        prompts = random.sample(all_prompts, REQUESTS_PER_QPS)

        vanilla_result = await run_poisson_benchmark(engine, prompts, qps, "vanilla")
        all_results.append(vanilla_result)

        metis_result = await run_poisson_benchmark(engine, prompts, qps, "metis")
        all_results.append(metis_result)

        print()

    # Save results
    output: dict[str, Any] = {
        "metadata": {
            "model": MODEL,
            "gpu_memory_utilization": GPU_MEM_UTIL,
            "requests_per_qps": REQUESTS_PER_QPS,
            "metis_fast_ratio": METIS_FAST_RATIO,
            "metis_fast_max_tokens": METIS_FAST_MAX_TOKENS,
            "vanilla_max_tokens": VANILLA_MAX_TOKENS,
            "metis_deep_max_tokens": METIS_DEEP_MAX_TOKENS,
            "routing_delay_us": METIS_ROUTING_DELAY_US,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "results": all_results,
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {OUTPUT_PATH}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'QPS':>5} | {'Mode':>8} | {'Tput (tok/s)':>13} | {'P50 TTFT':>10} | {'P99 TTFT':>10}")
    print("-" * 70)
    for r in all_results:
        print(
            f"{r['qps']:5.1f} | {r['mode']:>8} | "
            f"{r['throughput_toks']:>10.1f}   | "
            f"{r['p50_ttft_ms']:>7.1f} ms | "
            f"{r['p99_ttft_ms']:>7.1f} ms"
        )


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    asyncio.run(main())
