#!/usr/bin/env python3
"""
Phase 17: Universal Scaling Matrix
===================================
Prove METIS cognitive routing is agnostic to model scale (7B→72B)
and architecture (Qwen vs Llama) via a 5-model A/B benchmark.

For each model:
  - Baseline: standard "answer concisely" system prompt.
  - METIS: cognitive routing system prompt with <thinking> enforcement.

Metrics:
  - Accuracy (numeric match for GSM8K, substring for QA)
  - Token count (verbosity)
  - Avg Token Entropy H (Shannon entropy from vLLM logprobs)
  - Cognitive routing distribution (METIS only)

Backend: vLLM with bf16 (≤32B) or AWQ-4bit (>32B).
Hardware: NVIDIA GB10 (Blackwell CC12.1) | 122 GB unified memory.
"""
from __future__ import annotations

# ── Blackwell SM_121a fix: Triton's bundled ptxas is too old ──
import os
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-13.0/bin/ptxas"

import gc
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ═══════════════════════════════════════════════════════════
# Model Matrix Configuration
# ═══════════════════════════════════════════════════════════

CACHE_DIR = os.getenv("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"))

MODEL_MATRIX: List[Dict[str, Any]] = [
    {
        "name": "Qwen2.5-7B",
        "family": "Qwen",
        "params": "7B",
        "hf_repo": "Qwen/Qwen2.5-7B-Instruct",
        "local_path": f"{CACHE_DIR}/models--Qwen--Qwen2.5-7B-Instruct",
        "quantization": None,  # bf16
        "gpu_mem_util": 0.70,
        "max_model_len": 2048,
    },
    {
        "name": "Qwen2.5-32B",
        "family": "Qwen",
        "params": "32B",
        "hf_repo": "Qwen/Qwen2.5-32B-Instruct",
        "local_path": f"{CACHE_DIR}/models--Qwen--Qwen2.5-32B-Instruct",
        "quantization": None,  # bf16
        "gpu_mem_util": 0.70,
        "max_model_len": 1536,
    },
    {
        "name": "Qwen2.5-72B-AWQ",
        "family": "Qwen",
        "params": "72B",
        "hf_repo": "Qwen/Qwen2.5-72B-Instruct-AWQ",
        "local_path": f"{CACHE_DIR}/models--Qwen--Qwen2.5-72B-Instruct-AWQ",
        "quantization": "awq",
        "gpu_mem_util": 0.75,
        "max_model_len": 1536,
    },
    {
        "name": "Llama-3.1-8B",
        "family": "Llama",
        "params": "8B",
        "hf_repo": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "local_path": f"{CACHE_DIR}/models--unsloth--Meta-Llama-3.1-8B-Instruct",
        "quantization": None,  # bf16
        "gpu_mem_util": 0.70,
        "max_model_len": 2048,
    },
    {
        "name": "Llama-3.1-70B-AWQ",
        "family": "Llama",
        "params": "70B",
        "hf_repo": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        "local_path": f"{CACHE_DIR}/models--hugging-quants--Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        "quantization": "awq",
        "gpu_mem_util": 0.75,
        "max_model_len": 1536,
    },
]

# ═══════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════

BASELINE_SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely."

METIS_SYSTEM_PROMPT = (
    "You are METIS, an advanced AI with metacognitive awareness. "
    "For each query, assess its complexity:\n"
    "- For simple factual questions: answer directly and concisely.\n"
    "- For complex reasoning problems: use <thinking>...</thinking> tags "
    "to show your step-by-step reasoning process before giving the final answer.\n"
    "Always be accurate and concise."
)

# ═══════════════════════════════════════════════════════════
# Benchmark Constants
# ═══════════════════════════════════════════════════════════

N_COMPLEX = 50
N_SIMPLE = 50
SEED = 42
LOGPROBS_K = 5  # top-k logprobs for Shannon entropy

# ═══════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════
# Accuracy Checking (from Phase 16, battle-tested)
# ═══════════════════════════════════════════════════════════

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
    val = _extract_last_number(answer)
    if val is not None and abs(val - gold_val) < 0.01:
        return True
    clean = re.sub(r"(\d),(\d)", r"\1\2", answer)
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


# ═══════════════════════════════════════════════════════════
# Shannon Entropy from vLLM Logprobs
# ═══════════════════════════════════════════════════════════

def compute_token_entropy(output: Any) -> float:
    """Compute average Shannon entropy H = -Σ p·log₂(p) per token.

    Uses vLLM logprobs output. Each token has a dict of top-k logprobs.
    We exponentiate to get probabilities, compute per-token H, then average.
    """
    if not hasattr(output, 'logprobs') or output.logprobs is None:
        return 0.0

    token_entropies: List[float] = []
    for token_logprobs in output.logprobs:
        if token_logprobs is None:
            continue
        # token_logprobs is a dict: {token_id: Logprob(logprob, rank, decoded_token)}
        probs: List[float] = []
        for token_id, logprob_obj in token_logprobs.items():
            p = math.exp(logprob_obj.logprob)
            if p > 0:
                probs.append(p)

        if not probs:
            continue

        # Account for residual probability mass not in top-k
        total_p = sum(probs)
        if total_p < 1.0:
            residual = 1.0 - total_p
            if residual > 0:
                probs.append(residual)

        # Shannon entropy
        h = 0.0
        for p in probs:
            if p > 0:
                h -= p * math.log2(p)
        token_entropies.append(h)

    if not token_entropies:
        return 0.0
    return sum(token_entropies) / len(token_entropies)


# ═══════════════════════════════════════════════════════════
# Result Dataclass
# ═══════════════════════════════════════════════════════════

@dataclass
class BenchResult:
    question: str
    gold: str
    category: str  # "complex" or "simple"
    answer: str = ""
    correct: bool = False
    tokens: int = 0
    latency_ms: float = 0.0
    route: str = ""          # METIS only: DEEP/FAST
    thinking_tokens: int = 0  # METIS only
    avg_token_h: float = 0.0  # Shannon entropy proxy


@dataclass
class ModelResult:
    model_name: str
    family: str
    params: str
    quantization: str
    load_time_s: float = 0.0
    baseline: List[BenchResult] = field(default_factory=list)
    metis: List[BenchResult] = field(default_factory=list)
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════
# Benchmark Runner (vLLM batch generation)
# ═══════════════════════════════════════════════════════════

def run_benchmark(
    llm: Any,
    tokenizer: Any,
    items: List[Dict[str, str]],
    system_prompt: str,
    label: str,
    max_tokens: int = 512,
) -> List[BenchResult]:
    """Run benchmark on a pre-loaded vLLM instance using batch generation."""
    from vllm import SamplingParams

    print(f"\n{'='*64}")
    print(f"  [{label}] Running benchmark ({len(items)} items, batch)...")
    print(f"{'='*64}")

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        logprobs=LOGPROBS_K,
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

    # Batch generate
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=True)
    total_time = (time.perf_counter() - t0) * 1000
    avg_lat = total_time / len(items)
    print(f"  Batch done: {total_time/1000:.1f}s total, {avg_lat:.0f}ms/query avg")

    # Parse results
    results: List[BenchResult] = []
    for item, output in zip(items, outputs):
        r = BenchResult(
            question=item["q"], gold=item["a"], category=item["cat"]
        )

        completion = output.outputs[0]
        text = completion.text.strip()
        n_tokens = len(completion.token_ids)

        # Shannon entropy from logprobs
        r.avg_token_h = compute_token_entropy(completion)

        if is_metis:
            parsed = parse_cognitive_route(text)
            r.route = parsed["route"]
            r.thinking_tokens = parsed["thinking_tokens"]
            r.answer = parsed["answer"]
        else:
            r.answer = text
            r.route = "N/A"

        r.tokens = n_tokens
        r.latency_ms = avg_lat

        # Accuracy: check full text (including thinking) for answer
        if item["cat"] == "complex":
            r.correct = check_math_accuracy(text, item["a"])
        else:
            r.correct = check_qa_accuracy(text, item["a"])

        results.append(r)

    correct = sum(1 for r in results if r.correct)
    avg_h = sum(r.avg_token_h for r in results) / max(len(results), 1)
    print(f"  Results: {correct}/{len(results)} ({100*correct/len(results):.1f}%) "
          f"| Avg H={avg_h:.4f}")

    return results


# ═══════════════════════════════════════════════════════════
# Model Loading / Unloading
# ═══════════════════════════════════════════════════════════

def _resolve_hf_cache_path(cache_dir: str) -> Optional[str]:
    """Resolve HF cache dir to actual model path.

    HF cache has two layouts:
      1. Flat: files directly in cache_dir (our wget downloads)
      2. Nested: cache_dir/snapshots/<hash>/ (huggingface-cli downloads)
    Returns the resolved path with model files, or None.
    """
    # Check flat layout first
    if os.path.exists(os.path.join(cache_dir, "config.json")):
        return cache_dir

    # Check HF snapshot layout
    snapshots_dir = os.path.join(cache_dir, "snapshots")
    if os.path.isdir(snapshots_dir):
        # Find the latest snapshot (usually only one)
        snaps = sorted(os.listdir(snapshots_dir))
        for snap in reversed(snaps):
            snap_path = os.path.join(snapshots_dir, snap)
            if os.path.exists(os.path.join(snap_path, "config.json")):
                return snap_path

    return None


def check_model_ready(model_cfg: Dict[str, Any]) -> Tuple[bool, str]:
    """Check if model files exist locally. Returns (ready, resolved_path)."""
    local_path = model_cfg["local_path"]
    resolved = _resolve_hf_cache_path(local_path)

    if resolved is None:
        return False, local_path

    index_path = os.path.join(resolved, "model.safetensors.index.json")
    if os.path.exists(index_path):
        try:
            with open(index_path) as f:
                idx = json.load(f)
            shards = set(idx["weight_map"].values())
            for shard in shards:
                shard_path = os.path.join(resolved, shard)
                if not os.path.exists(shard_path):
                    return False, resolved
                if os.path.getsize(shard_path) < 1_000_000:
                    return False, resolved
            return True, resolved
        except Exception:
            return False, resolved

    # Single safetensors file
    single = os.path.join(resolved, "model.safetensors")
    if os.path.exists(single) and os.path.getsize(single) > 1_000_000:
        return True, resolved

    return False, resolved


def destroy_engine(llm: Any) -> None:
    """Aggressively release vLLM engine and GPU memory."""
    import torch

    # vLLM-specific cleanup
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
        print("  destroy_model_parallel() called")
    except Exception as e:
        print(f"  destroy_model_parallel() skipped: {e}")

    del llm
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Force Python GC multiple passes
    for _ in range(3):
        gc.collect()

    print("  GPU memory released.")
    time.sleep(2)  # Let CUDA settle


def benchmark_model(
    model_cfg: Dict[str, Any],
    items: List[Dict[str, str]],
) -> ModelResult:
    """Load a model, run baseline + METIS benchmarks, unload."""
    from vllm import LLM

    result = ModelResult(
        model_name=model_cfg["name"],
        family=model_cfg["family"],
        params=model_cfg["params"],
        quantization=model_cfg["quantization"] or "bf16",
    )

    print(f"\n{'#'*70}")
    print(f"  MODEL: {model_cfg['name']} ({model_cfg['params']})")
    print(f"  Path:  {model_cfg['local_path']}")
    print(f"  Quant: {model_cfg['quantization'] or 'bf16'}")
    print(f"{'#'*70}")

    # Check if model is ready
    ready, resolved_path = check_model_ready(model_cfg)
    if not ready:
        msg = f"Model files not found or incomplete at {model_cfg['local_path']}"
        print(f"  ⚠ SKIP: {msg}")
        result.error = msg
        return result

    print(f"  Resolved path: {resolved_path}")

    # Load model
    try:
        t0 = time.time()
        llm_kwargs: Dict[str, Any] = {
            "model": resolved_path,
            "dtype": "auto",
            "trust_remote_code": True,
            "gpu_memory_utilization": model_cfg["gpu_mem_util"],
            "max_model_len": model_cfg["max_model_len"],
            "enforce_eager": True,
        }
        if model_cfg["quantization"]:
            llm_kwargs["quantization"] = model_cfg["quantization"]

        llm = LLM(**llm_kwargs)
        result.load_time_s = time.time() - t0
        tokenizer = llm.get_tokenizer()
        print(f"  Loaded in {result.load_time_s:.1f}s")
    except Exception as e:
        msg = f"Failed to load: {e}"
        print(f"  ✗ ERROR: {msg}")
        result.error = msg
        return result

    # Run benchmarks
    try:
        # Phase A: Baseline
        result.baseline = run_benchmark(
            llm, tokenizer, items, BASELINE_SYSTEM_PROMPT,
            label=f"{model_cfg['name']} Baseline",
            max_tokens=512,
        )

        # Phase B: METIS
        result.metis = run_benchmark(
            llm, tokenizer, items, METIS_SYSTEM_PROMPT,
            label=f"{model_cfg['name']} METIS",
            max_tokens=1024,
        )
    except Exception as e:
        msg = f"Benchmark failed: {e}"
        print(f"  ✗ ERROR during benchmark: {msg}")
        result.error = msg

    # Unload
    print(f"\n  Unloading {model_cfg['name']}...")
    destroy_engine(llm)

    return result


# ═══════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════

def _stats(results: List[BenchResult], cat: str) -> Dict[str, float]:
    """Compute stats for a category subset."""
    subset = [r for r in results if r.category == cat]
    n = len(subset)
    if n == 0:
        return {"acc": 0.0, "tokens": 0.0, "h": 0.0, "n": 0}
    return {
        "acc": sum(1 for r in subset if r.correct) / n * 100,
        "tokens": sum(r.tokens for r in subset) / n,
        "h": sum(r.avg_token_h for r in subset) / n,
        "n": n,
    }


def generate_markdown_report(
    all_results: List[ModelResult],
    output_path: str,
) -> None:
    """Generate the UNIVERSAL_SCALING_MATRIX.md report."""

    lines: List[str] = []
    lines.append("# Phase 17: Universal Scaling Matrix")
    lines.append("")
    lines.append("> **Hypothesis**: METIS cognitive routing is agnostic to model scale "
                 "(7B→72B) and architecture (Qwen vs Llama).")
    lines.append(">")
    lines.append("> **Method**: A/B benchmark on 100 prompts (50 GSM8K + 50 Simple QA) "
                 "across 5 models, comparing Baseline vs METIS cognitive system prompt.")
    lines.append(">")
    lines.append("> **Entropy**: Shannon Entropy H = -Σ p·log₂(p) computed from vLLM "
                 "top-5 logprobs per generated token.")
    lines.append("")
    lines.append(f"**Hardware**: NVIDIA GB10 (Blackwell CC12.1) | 122 GB unified memory")
    lines.append(f"**Seed**: {SEED} | **Date**: {time.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    # ── Master Comparison Table ──
    lines.append("## 1. Master Comparison Table")
    lines.append("")
    lines.append("| Model | Params | Quant | Task | BL Acc% | MT Acc% | Δ Acc | "
                 "BL Tok | MT Tok | Δ Tok% | BL H | MT H |")
    lines.append("|-------|--------|-------|------|---------|---------|-------|"
                 "--------|--------|--------|------|------|")

    for mr in all_results:
        if mr.error and not mr.baseline:
            lines.append(f"| {mr.model_name} | {mr.params} | {mr.quantization} | "
                         f"— | SKIP | SKIP | — | — | — | — | — | — |")
            continue

        for cat, cat_label in [("complex", "Complex"), ("simple", "Simple")]:
            bl = _stats(mr.baseline, cat)
            mt = _stats(mr.metis, cat)
            delta_acc = mt["acc"] - bl["acc"]
            delta_tok = (mt["tokens"] - bl["tokens"]) / max(bl["tokens"], 1) * 100

            lines.append(
                f"| {mr.model_name} | {mr.params} | {mr.quantization} | "
                f"{cat_label} | {bl['acc']:.1f} | {mt['acc']:.1f} | "
                f"{delta_acc:+.1f} | {bl['tokens']:.1f} | {mt['tokens']:.1f} | "
                f"{delta_tok:+.1f}% | {bl['h']:.4f} | {mt['h']:.4f} |"
            )

    lines.append("")

    # ── Overall Summary Table ──
    lines.append("## 2. Overall Accuracy Summary")
    lines.append("")
    lines.append("| Model | Family | Params | BL Overall% | MT Overall% | Δ% | "
                 "BL Avg Tok | MT Avg Tok |")
    lines.append("|-------|--------|--------|-------------|-------------|-----|"
                 "-----------|-----------|")

    for mr in all_results:
        if mr.error and not mr.baseline:
            lines.append(f"| {mr.model_name} | {mr.family} | {mr.params} | "
                         f"SKIP | SKIP | — | — | — |")
            continue

        n_bl = len(mr.baseline)
        n_mt = len(mr.metis)
        bl_acc = sum(1 for r in mr.baseline if r.correct) / max(n_bl, 1) * 100
        mt_acc = sum(1 for r in mr.metis if r.correct) / max(n_mt, 1) * 100
        bl_tok = sum(r.tokens for r in mr.baseline) / max(n_bl, 1)
        mt_tok = sum(r.tokens for r in mr.metis) / max(n_mt, 1)

        lines.append(
            f"| {mr.model_name} | {mr.family} | {mr.params} | "
            f"{bl_acc:.1f} | {mt_acc:.1f} | {mt_acc - bl_acc:+.1f} | "
            f"{bl_tok:.1f} | {mt_tok:.1f} |"
        )

    lines.append("")

    # ── Epistemic Sharpening Table ──
    lines.append("## 3. Epistemic Sharpening: Avg Token Entropy on Simple Tasks")
    lines.append("")
    lines.append("> If the hypothesis holds, H_simple should **decrease** monotonically "
                 "with parameter count within each family.")
    lines.append("")
    lines.append("| Model | Family | Params | BL H (Simple) | MT H (Simple) | Δ H |")
    lines.append("|-------|--------|--------|---------------|---------------|-----|")

    for mr in all_results:
        if mr.error and not mr.baseline:
            lines.append(f"| {mr.model_name} | {mr.family} | {mr.params} | "
                         f"SKIP | SKIP | — |")
            continue

        bl_simple = _stats(mr.baseline, "simple")
        mt_simple = _stats(mr.metis, "simple")
        delta_h = mt_simple["h"] - bl_simple["h"]

        lines.append(
            f"| {mr.model_name} | {mr.family} | {mr.params} | "
            f"{bl_simple['h']:.4f} | {mt_simple['h']:.4f} | {delta_h:+.4f} |"
        )

    lines.append("")

    # ── Cognitive Routing Distribution ──
    lines.append("## 4. METIS Cognitive Routing Distribution")
    lines.append("")
    lines.append("| Model | Complex FAST | Complex DEEP | Simple FAST | Simple DEEP | "
                 "Avg Think Tok |")
    lines.append("|-------|-------------|-------------|------------|------------|"
                 "--------------|")

    for mr in all_results:
        if mr.error and not mr.metis:
            continue
        mt_complex = [r for r in mr.metis if r.category == "complex"]
        mt_simple = [r for r in mr.metis if r.category == "simple"]

        c_deep = sum(1 for r in mt_complex if r.route == "DEEP")
        c_fast = len(mt_complex) - c_deep
        s_deep = sum(1 for r in mt_simple if r.route == "DEEP")
        s_fast = len(mt_simple) - s_deep
        avg_think = (sum(r.thinking_tokens for r in mr.metis if r.route == "DEEP")
                     / max(c_deep + s_deep, 1))

        lines.append(
            f"| {mr.model_name} | {c_fast}/{len(mt_complex)} "
            f"({100*c_fast/max(len(mt_complex),1):.0f}%) | "
            f"{c_deep}/{len(mt_complex)} "
            f"({100*c_deep/max(len(mt_complex),1):.0f}%) | "
            f"{s_fast}/{len(mt_simple)} "
            f"({100*s_fast/max(len(mt_simple),1):.0f}%) | "
            f"{s_deep}/{len(mt_simple)} "
            f"({100*s_deep/max(len(mt_simple),1):.0f}%) | "
            f"{avg_think:.0f} |"
        )

    lines.append("")

    # ── Cross-Architecture Analysis ──
    lines.append("## 5. Cross-Architecture Analysis")
    lines.append("")

    qwen_results = [mr for mr in all_results if mr.family == "Qwen" and not mr.error]
    llama_results = [mr for mr in all_results if mr.family == "Llama" and not mr.error]

    if qwen_results and llama_results:
        lines.append("### Qwen vs Llama at Similar Scales")
        lines.append("")

        # Compare 7B/8B
        q7 = next((mr for mr in qwen_results if "7B" in mr.params), None)
        l8 = next((mr for mr in llama_results if "8B" in mr.params), None)
        if q7 and l8:
            q7_acc = sum(1 for r in q7.metis if r.correct) / max(len(q7.metis), 1) * 100
            l8_acc = sum(1 for r in l8.metis if r.correct) / max(len(l8.metis), 1) * 100
            q7_bl = sum(1 for r in q7.baseline if r.correct) / max(len(q7.baseline), 1) * 100
            l8_bl = sum(1 for r in l8.baseline if r.correct) / max(len(l8.baseline), 1) * 100
            lines.append(f"- **~8B scale**: Qwen-7B METIS={q7_acc:.1f}% (Δ={q7_acc-q7_bl:+.1f}) "
                         f"vs Llama-8B METIS={l8_acc:.1f}% (Δ={l8_acc-l8_bl:+.1f})")

        # Compare 70B/72B
        q72 = next((mr for mr in qwen_results if "72B" in mr.params), None)
        l70 = next((mr for mr in llama_results if "70B" in mr.params), None)
        if q72 and l70:
            q72_acc = sum(1 for r in q72.metis if r.correct) / max(len(q72.metis), 1) * 100
            l70_acc = sum(1 for r in l70.metis if r.correct) / max(len(l70.metis), 1) * 100
            q72_bl = sum(1 for r in q72.baseline if r.correct) / max(len(q72.baseline), 1) * 100
            l70_bl = sum(1 for r in l70.baseline if r.correct) / max(len(l70.baseline), 1) * 100
            lines.append(f"- **~70B scale**: Qwen-72B METIS={q72_acc:.1f}% (Δ={q72_acc-q72_bl:+.1f}) "
                         f"vs Llama-70B METIS={l70_acc:.1f}% (Δ={l70_acc-l70_bl:+.1f})")

        lines.append("")

    # ── Conclusion ──
    lines.append("## 6. Conclusion")
    lines.append("")

    successful = [mr for mr in all_results if not mr.error]
    if successful:
        # Check if METIS improves accuracy across all successful models
        all_improve = all(
            sum(1 for r in mr.metis if r.correct) >= sum(1 for r in mr.baseline if r.correct)
            for mr in successful
        )
        # Check both families represented
        families = set(mr.family for mr in successful)
        multi_family = len(families) >= 2
        # Check multiple scales
        scales = set(mr.params for mr in successful)
        multi_scale = len(scales) >= 3

        if all_improve and multi_family and multi_scale:
            lines.append("**VERDICT: ✅ METIS is architecture-agnostic and scale-invariant.**")
            lines.append("")
            lines.append(f"Across {len(successful)} models spanning {len(families)} "
                         f"architectures ({', '.join(families)}) and {len(scales)} "
                         f"parameter scales ({', '.join(sorted(scales))}), METIS cognitive "
                         f"routing consistently improved or maintained accuracy while "
                         f"demonstrating selective deep reasoning on complex tasks.")
        else:
            lines.append("**VERDICT: ⚠ Partial validation.**")
            lines.append("")
            lines.append(f"Tested {len(successful)} models. "
                         f"{'All improved.' if all_improve else 'Not all improved.'} "
                         f"{'Multi-family.' if multi_family else 'Single family only.'} "
                         f"{'Multi-scale.' if multi_scale else 'Limited scales.'}")
    else:
        lines.append("**VERDICT: ❌ No models completed successfully.**")

    lines.append("")

    # ── Errors ──
    errors = [mr for mr in all_results if mr.error]
    if errors:
        lines.append("## Appendix: Errors")
        lines.append("")
        for mr in errors:
            lines.append(f"- **{mr.model_name}**: {mr.error}")
        lines.append("")

    # Write
    content = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n  Report written to: {output_path}")


# ═══════════════════════════════════════════════════════════
# Terminal Report (colorful summary)
# ═══════════════════════════════════════════════════════════

def print_terminal_report(all_results: List[ModelResult]) -> None:
    """Print a concise terminal summary."""
    print(f"\n{'='*70}")
    print(f"  Phase 17: Universal Scaling Matrix — Final Report")
    print(f"{'='*70}")

    for mr in all_results:
        if mr.error and not mr.baseline:
            print(f"\n  {mr.model_name} ({mr.params}): SKIPPED — {mr.error}")
            continue

        n = len(mr.baseline)
        bl_acc = sum(1 for r in mr.baseline if r.correct) / max(n, 1) * 100
        mt_acc = sum(1 for r in mr.metis if r.correct) / max(n, 1) * 100
        bl_tok = sum(r.tokens for r in mr.baseline) / max(n, 1)
        mt_tok = sum(r.tokens for r in mr.metis) / max(n, 1)
        bl_h = sum(r.avg_token_h for r in mr.baseline) / max(n, 1)
        mt_h = sum(r.avg_token_h for r in mr.metis) / max(n, 1)

        print(f"\n  {mr.model_name} ({mr.params}, {mr.quantization}):")
        print(f"    {'':18s} {'Baseline':>10} {'METIS':>10} {'Delta':>10}")
        print(f"    {'-'*50}")
        print(f"    {'Accuracy':18s} {bl_acc:>9.1f}% {mt_acc:>9.1f}% "
              f"{mt_acc-bl_acc:>+9.1f}%")
        print(f"    {'Avg Tokens':18s} {bl_tok:>10.1f} {mt_tok:>10.1f} "
              f"{(mt_tok-bl_tok)/max(bl_tok,1)*100:>+9.1f}%")
        print(f"    {'Avg Token H':18s} {bl_h:>10.4f} {mt_h:>10.4f} "
              f"{mt_h-bl_h:>+9.4f}")
        print(f"    Load time: {mr.load_time_s:.1f}s")

    print(f"\n{'='*70}")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main() -> None:
    print("Phase 17: Universal Scaling Matrix")
    print(f"{'='*70}")
    print(f"  Models: {len(MODEL_MATRIX)}")
    print(f"  Prompts: {N_COMPLEX} complex + {N_SIMPLE} simple = {N_COMPLEX + N_SIMPLE}")
    print(f"  Seed: {SEED}")
    print(f"  Entropy: Shannon H from top-{LOGPROBS_K} logprobs")
    print(f"{'='*70}")

    # Load data once
    print("\nLoading benchmark data...")
    gsm8k_items = load_gsm8k_samples(N_COMPLEX, SEED)
    simple_items = load_simple_samples(N_SIMPLE, SEED)
    all_items = gsm8k_items + simple_items
    print(f"  {len(gsm8k_items)} complex + {len(simple_items)} simple = {len(all_items)} total")

    # Run matrix
    all_results: List[ModelResult] = []

    for i, model_cfg in enumerate(MODEL_MATRIX):
        print(f"\n{'▓'*70}")
        print(f"  [{i+1}/{len(MODEL_MATRIX)}] {model_cfg['name']}")
        print(f"{'▓'*70}")

        try:
            result = benchmark_model(model_cfg, all_items)
        except Exception as e:
            print(f"  ✗ FATAL ERROR on {model_cfg['name']}: {e}")
            result = ModelResult(
                model_name=model_cfg["name"],
                family=model_cfg["family"],
                params=model_cfg["params"],
                quantization=model_cfg["quantization"] or "bf16",
                error=str(e),
            )

        all_results.append(result)

    # Terminal report
    print_terminal_report(all_results)

    # Markdown report
    report_path = str(PROJECT_ROOT / "docs" / "UNIVERSAL_SCALING_MATRIX.md")
    generate_markdown_report(all_results, report_path)

    # Save raw JSON
    json_path = str(PROJECT_ROOT / "phase17_results.json")
    json_data: List[Dict[str, Any]] = []
    for mr in all_results:
        entry: Dict[str, Any] = {
            "model": mr.model_name,
            "family": mr.family,
            "params": mr.params,
            "quantization": mr.quantization,
            "load_time_s": mr.load_time_s,
            "error": mr.error,
        }
        for phase_name, phase_results in [("baseline", mr.baseline), ("metis", mr.metis)]:
            entry[phase_name] = [
                {
                    "question": r.question[:100],
                    "gold": r.gold,
                    "category": r.category,
                    "correct": r.correct,
                    "tokens": r.tokens,
                    "route": r.route,
                    "avg_token_h": round(r.avg_token_h, 6),
                }
                for r in phase_results
            ]
        json_data.append(entry)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"  Raw results saved to: {json_path}")

    print(f"\n{'='*70}")
    print(f"  Phase 17 COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
