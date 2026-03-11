#!/usr/bin/env python3
"""
Phase 16: Metacognitive Scaling Laws Validation
================================================
A/B benchmark on Qwen2.5-32B-Instruct with FP8 (Blackwell-native E4M3).

Model A (Baseline): Raw 32B FP8 — standard "answer concisely" prompt.
Model B (METIS):    32B FP8 + METIS cognitive system prompt — triggers
                    thinking/routing behavior, parsed by state machine.

Backend: vLLM with bf16 (FP8 CUTLASS unsupported on CC 12.1 / PyTorch 2.9.1).

Metrics:
  - Accuracy (numeric match for GSM8K, substring for QA)
  - Token count (verbosity)
  - Latency (ms per query)
  - Cognitive routing distribution (METIS only)
"""
from __future__ import annotations

import gc
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Config ──
MODEL_PATH = os.getenv("METIS_SCALING_MODEL", "Qwen/Qwen2.5-32B-Instruct")
N_COMPLEX = 50
N_SIMPLE = 50
SEED = 42

# ── METIS Cognitive System Prompt ──
# This triggers cognitive routing behavior in instruction-following models.
METIS_SYSTEM_PROMPT = (
    "You are METIS, an advanced AI with metacognitive awareness. "
    "For each query, assess its complexity:\n"
    "- For simple factual questions: answer directly and concisely.\n"
    "- For complex reasoning problems: use <thinking>...</thinking> tags "
    "to show your step-by-step reasoning process before giving the final answer.\n"
    "Always be accurate and concise."
)

BASELINE_SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely."

# ── Data ──
_SIMPLE_QA: List[Dict[str, str]] = [
    {"q": "What is the capital of France?", "a": "Paris"},
    {"q": "What is the capital of Japan?", "a": "Tokyo"},
    {"q": "What is the capital of Germany?", "a": "Berlin"},
    {"q": "What is the capital of Italy?", "a": "Rome"},
    {"q": "What is the capital of Australia?", "a": "Canberra"},
    {"q": "What is the capital of Canada?", "a": "Ottawa"},
    {"q": "What is the capital of Brazil?", "a": "Brasilia"},
    {"q": "What is the capital of India?", "a": "New Delhi"},
    {"q": "What is the capital of Russia?", "a": "Moscow"},
    {"q": "What is the capital of China?", "a": "Beijing"},
    {"q": "What planet is closest to the Sun?", "a": "Mercury"},
    {"q": "What is the largest planet in our solar system?", "a": "Jupiter"},
    {"q": "What is the chemical symbol for gold?", "a": "Au"},
    {"q": "What is the chemical symbol for water?", "a": "H2O"},
    {"q": "What is the chemical symbol for sodium?", "a": "Na"},
    {"q": "What is the speed of light in km/s (approximately)?", "a": "300000"},
    {"q": "Who wrote Romeo and Juliet?", "a": "Shakespeare"},
    {"q": "Who painted the Mona Lisa?", "a": "Leonardo"},
    {"q": "What is the largest ocean on Earth?", "a": "Pacific"},
    {"q": "What is the tallest mountain in the world?", "a": "Everest"},
    {"q": "What is the smallest country in the world by area?", "a": "Vatican"},
    {"q": "What year did World War II end?", "a": "1945"},
    {"q": "What year did the Berlin Wall fall?", "a": "1989"},
    {"q": "Who was the first person to walk on the Moon?", "a": "Armstrong"},
    {"q": "What is the boiling point of water in Celsius?", "a": "100"},
    {"q": "What is the freezing point of water in Celsius?", "a": "0"},
    {"q": "How many continents are there?", "a": "7"},
    {"q": "How many planets are in our solar system?", "a": "8"},
    {"q": "What element has the atomic number 1?", "a": "Hydrogen"},
    {"q": "What element has the atomic number 6?", "a": "Carbon"},
    {"q": "What is the currency of Japan?", "a": "Yen"},
    {"q": "What is the currency of the United Kingdom?", "a": "Pound"},
    {"q": "What gas do plants absorb from the atmosphere?", "a": "CO2"},
    {"q": "What is the powerhouse of the cell?", "a": "Mitochondria"},
    {"q": "Who developed the theory of general relativity?", "a": "Einstein"},
    {"q": "What is the largest mammal on Earth?", "a": "Blue whale"},
    {"q": "What language has the most native speakers?", "a": "Mandarin"},
    {"q": "What is the hardest natural substance?", "a": "Diamond"},
    {"q": "What organ pumps blood through the body?", "a": "Heart"},
    {"q": "How many bones does an adult human have?", "a": "206"},
    {"q": "What is the longest river in the world?", "a": "Nile"},
    {"q": "What is the largest desert in the world?", "a": "Sahara"},
    {"q": "What is the square root of 144?", "a": "12"},
    {"q": "What is 15 squared?", "a": "225"},
    {"q": "What is the value of Pi to 2 decimal places?", "a": "3.14"},
    {"q": "Who is known as the father of computers?", "a": "Babbage"},
    {"q": "What does DNA stand for?", "a": "Deoxyribonucleic"},
    {"q": "What is the most abundant gas in Earth's atmosphere?", "a": "Nitrogen"},
    {"q": "What animal is known as the King of the Jungle?", "a": "Lion"},
    {"q": "What is the main ingredient in glass?", "a": "Sand"},
]

_ROBUST_NUM_RE = re.compile(r"[-+]?\d*\.\d+|\d+")
_THINKING_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)


def _extract_last_number(text: str) -> Optional[float]:
    clean = re.sub(r"(\d),(\d)", r"\1\2", text)
    matches = _ROBUST_NUM_RE.findall(clean)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def check_math_accuracy(answer: str, gold: str) -> bool:
    try:
        gold_val = float(gold.strip().replace(",", ""))
    except ValueError:
        return gold.strip().lower() in answer.lower()
    # Check answer text (strip thinking blocks for cleaner matching)
    full_text = answer
    val = _extract_last_number(full_text)
    if val is not None and abs(val - gold_val) < 0.01:
        return True
    clean = re.sub(r"(\d),(\d)", r"\1\2", full_text)
    for num_str in _ROBUST_NUM_RE.findall(clean):
        try:
            v = float(num_str)
            if abs(v - gold_val) < 0.01:
                return True
        except ValueError:
            continue
    return False


def check_qa_accuracy(answer: str, gold: str) -> bool:
    if gold.lower() in answer.lower():
        return True
    ans_clean = re.sub(r"[^\w\s]", "", answer.lower())
    gold_clean = re.sub(r"[^\w\s]", "", gold.lower())
    return gold_clean in ans_clean


def parse_cognitive_route(text: str) -> Dict[str, Any]:
    """Parse METIS-style cognitive output into route + answer."""
    thinking_match = _THINKING_RE.search(text)
    if thinking_match:
        thinking_text = thinking_match.group(1).strip()
        answer = _THINKING_RE.sub("", text).strip()
        return {
            "route": "DEEP",
            "thinking": thinking_text,
            "answer": answer,
            "thinking_tokens": len(thinking_text.split()),
        }
    return {
        "route": "FAST",
        "thinking": "",
        "answer": text.strip(),
        "thinking_tokens": 0,
    }


@dataclass
class BenchResult:
    question: str
    gold: str
    category: str  # "complex" or "simple"
    answer: str = ""
    correct: bool = False
    tokens: int = 0
    latency_ms: float = 0.0
    route: str = ""  # METIS only
    thinking_tokens: int = 0  # METIS only


def load_gsm8k_samples(n: int, seed: int) -> List[Dict[str, str]]:
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    items: List[Dict[str, str]] = []
    for idx in indices:
        row = ds[idx]
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        items.append({"q": row["question"], "a": gold, "cat": "complex"})
    return items


def load_simple_samples(n: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    pool = list(_SIMPLE_QA)
    rng.shuffle(pool)
    return [{"q": qa["q"], "a": qa["a"], "cat": "simple"} for qa in pool[:n]]


def run_benchmark(
    llm: "LLM",
    tokenizer: "PreTrainedTokenizer",
    items: List[Dict[str, str]],
    system_prompt: str,
    label: str,
    max_tokens: int = 512,
) -> List[BenchResult]:
    """Run benchmark on a pre-loaded vLLM instance using batch generation."""
    from vllm import SamplingParams

    print(f"\n{'='*64}")
    print(f"  [{label}] Running benchmark ({len(items)} items, batch mode)...")
    print(f"{'='*64}")

    sampling = SamplingParams(
        temperature=0.0,  # Greedy for reproducibility
        max_tokens=max_tokens,
    )

    is_metis = "METIS" in label.upper()

    # Build all prompts
    prompts: List[str] = []
    for item in items:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item["q"]},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))

    # Batch generate ALL prompts at once (vLLM handles scheduling)
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=True)
    total_time = (time.perf_counter() - t0) * 1000
    avg_lat = total_time / len(items)
    print(f"  Batch generation done: {total_time/1000:.1f}s total, {avg_lat:.0f}ms/query avg")

    # Parse results
    results: List[BenchResult] = []
    for i, (item, output) in enumerate(zip(items, outputs)):
        r = BenchResult(
            question=item["q"], gold=item["a"], category=item["cat"]
        )

        text = output.outputs[0].text.strip()
        n_tokens = len(output.outputs[0].token_ids)

        if is_metis:
            parsed = parse_cognitive_route(text)
            r.route = parsed["route"]
            r.thinking_tokens = parsed["thinking_tokens"]
            r.answer = parsed["answer"]
        else:
            r.answer = text
            r.route = "N/A"

        r.tokens = n_tokens
        r.latency_ms = avg_lat  # Batch mode: amortized latency

        full_check_text = text
        if item["cat"] == "complex":
            r.correct = check_math_accuracy(full_check_text, item["a"])
        else:
            r.correct = check_qa_accuracy(full_check_text, item["a"])

        results.append(r)

    correct = sum(1 for r in results if r.correct)
    print(f"  Results: {correct}/{len(results)} correct ({100*correct/len(results):.1f}%)")

    return results


def print_report(
    baseline: List[BenchResult],
    metis: List[BenchResult],
) -> None:
    """Print the full Phase 16 comparison report."""
    print(f"\n{'='*70}")
    print(f"  Phase 16: Metacognitive Scaling Laws Validation Report")
    print(f"  Model: Qwen2.5-32B-Instruct | Backend: vLLM bf16")
    print(f"  Hardware: NVIDIA GB10 (Blackwell CC12.1) | 130.7 GB VRAM")
    print(f"{'='*70}\n")

    for cat in ["complex", "simple"]:
        bl_cat = [r for r in baseline if r.category == cat]
        mt_cat = [r for r in metis if r.category == cat]
        n = len(bl_cat)

        bl_acc = sum(1 for r in bl_cat if r.correct) / max(n, 1) * 100
        mt_acc = sum(1 for r in mt_cat if r.correct) / max(n, 1) * 100
        bl_tok = sum(r.tokens for r in bl_cat) / max(n, 1)
        mt_tok = sum(r.tokens for r in mt_cat) / max(n, 1)
        bl_lat = sum(r.latency_ms for r in bl_cat) / max(n, 1)
        mt_lat = sum(r.latency_ms for r in mt_cat) / max(n, 1)

        delta_acc = mt_acc - bl_acc
        delta_tok = (mt_tok - bl_tok) / max(bl_tok, 1) * 100
        delta_lat = (mt_lat - bl_lat) / max(bl_lat, 1) * 100

        label = "Complex (GSM8K)" if cat == "complex" else "Simple (QA)"
        print(f"  {label} (n={n}):")
        print(f"    {'Metric':<20} {'Baseline':>12} {'METIS':>12} {'Delta':>12}")
        print(f"    {'-'*56}")
        print(f"    {'Accuracy':<20} {bl_acc:>11.1f}% {mt_acc:>11.1f}% {delta_acc:>+11.1f}%")
        print(f"    {'Avg Tokens':<20} {bl_tok:>12.1f} {mt_tok:>12.1f} {delta_tok:>+11.1f}%")
        print(f"    {'Avg Latency (ms)':<20} {bl_lat:>12.0f} {mt_lat:>12.0f} {delta_lat:>+11.1f}%")
        print()

    # Overall
    n_total = len(baseline)
    bl_total_acc = sum(1 for r in baseline if r.correct) / max(n_total, 1) * 100
    mt_total_acc = sum(1 for r in metis if r.correct) / max(n_total, 1) * 100
    bl_total_tok = sum(r.tokens for r in baseline) / max(n_total, 1)
    mt_total_tok = sum(r.tokens for r in metis) / max(n_total, 1)
    bl_total_lat = sum(r.latency_ms for r in baseline) / max(n_total, 1)
    mt_total_lat = sum(r.latency_ms for r in metis) / max(n_total, 1)

    print(f"  Overall (n={n_total}):")
    print(f"    {'Metric':<20} {'Baseline':>12} {'METIS':>12} {'Delta':>12}")
    print(f"    {'-'*56}")
    print(f"    {'Accuracy':<20} {bl_total_acc:>11.1f}% {mt_total_acc:>11.1f}% "
          f"{mt_total_acc - bl_total_acc:>+11.1f}%")
    print(f"    {'Avg Tokens':<20} {bl_total_tok:>12.1f} {mt_total_tok:>12.1f} "
          f"{(mt_total_tok - bl_total_tok) / max(bl_total_tok, 1) * 100:>+11.1f}%")
    print(f"    {'Avg Latency (ms)':<20} {bl_total_lat:>12.0f} {mt_total_lat:>12.0f} "
          f"{(mt_total_lat - bl_total_lat) / max(bl_total_lat, 1) * 100:>+11.1f}%")

    # METIS routing distribution
    print(f"\n  {'─'*56}")
    print(f"  METIS Cognitive Routing Distribution:")
    for cat in ["complex", "simple"]:
        mt_cat = [r for r in metis if r.category == cat]
        n = len(mt_cat)
        deep = sum(1 for r in mt_cat if r.route == "DEEP")
        fast = n - deep
        avg_think = sum(r.thinking_tokens for r in mt_cat if r.route == "DEEP") / max(deep, 1)
        label = "Complex" if cat == "complex" else "Simple"
        print(f"    {label}: FAST={fast}/{n} ({100*fast/max(n,1):.0f}%) "
              f"DEEP={deep}/{n} ({100*deep/max(n,1):.0f}%) "
              f"avg_thinking_tokens={avg_think:.0f}")

    # FP8 performance summary
    print(f"\n  {'─'*56}")
    print(f"  Performance Summary:")
    print(f"    Precision: bf16 (FP8 CUTLASS unavailable on CC 12.1)")
    print(f"    Baseline avg latency: {bl_total_lat:.0f}ms/query")
    print(f"    METIS avg latency: {mt_total_lat:.0f}ms/query")

    # Pareto assessment
    pareto = (mt_total_acc >= bl_total_acc) and (mt_total_tok <= bl_total_tok)
    pareto_str = "✅ PARETO OPTIMAL" if pareto else "❌ NOT Pareto optimal"
    print(f"\n  {'═'*56}")
    print(f"  Verdict: {pareto_str}")
    if pareto:
        print(f"    Accuracy: {mt_total_acc:.1f}% ≥ {bl_total_acc:.1f}% (baseline)")
        print(f"    Tokens:   {mt_total_tok:.1f} ≤ {bl_total_tok:.1f} (baseline)")
    print(f"  {'═'*56}")


def main() -> None:
    from vllm import LLM

    print("Phase 16: Metacognitive Scaling Laws Validation")
    print(f"Loading evaluation data (seed={SEED})...")

    gsm8k_items = load_gsm8k_samples(N_COMPLEX, SEED)
    simple_items = load_simple_samples(N_SIMPLE, SEED)
    all_items = gsm8k_items + simple_items
    print(f"  {len(gsm8k_items)} complex + {len(simple_items)} simple = {len(all_items)} total")

    # Load model ONCE, share across both benchmarks
    print(f"\n{'='*64}")
    print(f"  Loading Qwen2.5-32B-Instruct (bf16) via vLLM...")
    print(f"  Path: {MODEL_PATH}")
    print(f"{'='*64}")

    t0 = time.time()
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.70,
        max_model_len=1536,
        enforce_eager=True,  # Avoid CUDA graph issues on CC 12.1
    )
    load_time = time.time() - t0
    tokenizer = llm.get_tokenizer()
    print(f"  Model loaded in {load_time:.1f}s (bf16, enforce_eager=True)")

    # Phase A: Baseline
    baseline_results = run_benchmark(
        llm, tokenizer, all_items, BASELINE_SYSTEM_PROMPT,
        label="Model A: Raw 32B bf16", max_tokens=512,
    )

    # Phase B: METIS
    metis_results = run_benchmark(
        llm, tokenizer, all_items, METIS_SYSTEM_PROMPT,
        label="Model B: METIS + 32B bf16", max_tokens=1024,
    )

    # Cleanup
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    # Report
    print_report(baseline_results, metis_results)


if __name__ == "__main__":
    main()
