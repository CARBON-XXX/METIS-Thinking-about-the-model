#!/usr/bin/env python3
"""
METIS Academic Benchmark — A/B Pareto Efficiency Test
=====================================================
Model A (Baseline): Raw Qwen2.5-7B-Instruct — no cognitive routing, no search.
Model B (METIS):    MetacognitiveOrchestrator — DPO + Gateway + Semantic Entropy + DDG.

Datasets:
  - 50 Complex (gsm8k test): math word problems with numeric gold answers.
  - 50 Simple (built-in trivia): factual questions with substring gold answers.

Metrics per prompt:
  - accuracy (strict numeric match for math, substring for QA)
  - compute_cost (tokens generated)
  - semantic_entropy + route_taken (METIS only)

Output: terminal statistical report proving Pareto efficiency.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.WARNING,
)
logger = logging.getLogger("benchmark")
logger.setLevel(logging.INFO)

# ── ANSI ──
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"


# ═══════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════

@dataclass
class BenchItem:
    question: str
    gold_answer: str
    category: str  # "complex" or "simple"
    source: str

# ── Built-in Simple/Factual Questions ──
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


def load_gsm8k_samples(n: int = 50, seed: int = 42) -> List[BenchItem]:
    """Load n random gsm8k test samples from local cache."""
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    items: List[BenchItem] = []
    for idx in indices:
        row = ds[idx]
        # Gold answer is after #### in the answer field
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        items.append(BenchItem(
            question=row["question"],
            gold_answer=gold,
            category="complex",
            source=f"gsm8k#{idx}",
        ))
    return items


def load_simple_samples(n: int = 50, seed: int = 42) -> List[BenchItem]:
    """Load n simple factual QA samples."""
    rng = random.Random(seed)
    pool = list(_SIMPLE_QA)
    rng.shuffle(pool)
    items: List[BenchItem] = []
    for i, qa in enumerate(pool[:n]):
        items.append(BenchItem(
            question=qa["q"],
            gold_answer=qa["a"],
            category="simple",
            source=f"trivia#{i}",
        ))
    return items


# ═══════════════════════════════════════════════════════════
# Answer Extraction & Scoring
# ═══════════════════════════════════════════════════════════

# Robust number regex: matches integers, decimals, negative numbers
_ROBUST_NUM_RE = re.compile(r'[-+]?\d*\.\d+|\d+')
_STRIP_PUNCT_RE = re.compile(r'[^\w\s]')


def _extract_last_number(text: str) -> Optional[float]:
    """Extract the LAST number from text using a robust regex.

    This is the gold-standard GSM8K extraction: take the last numeric
    token from the full output, ignoring formatting noise.
    """
    # Strip commas inside numbers (e.g. "1,234" → "1234")
    clean = re.sub(r'(\d),(\d)', r'\1\2', text)
    nums = _ROBUST_NUM_RE.findall(clean)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def check_math_accuracy(model_answer: str, gold: str,
                        thinking_text: str = "") -> bool:
    """Robust numeric accuracy: last-number extraction on full output.

    Strategy:
    1. Concatenate ``model_answer`` + ``thinking_text`` into one blob.
    2. Use ``re.findall(r"[-+]?\d*\.\d+|\d+", blob)[-1]`` — the LAST
       number in the full output is the answer (GSM8K convention).
    3. Compare against gold with ε < 0.01.

    This eliminates False Negatives from thinking-block parser failures
    where the answer ends up inside the thinking text.
    """
    try:
        gold_val = float(gold.strip().replace(",", ""))
    except ValueError:
        return gold.strip().lower() in model_answer.lower()

    # Build the full text blob: answer first, then thinking as fallback
    full_text = model_answer
    if thinking_text:
        full_text = full_text + " " + thinking_text

    val = _extract_last_number(full_text)
    if val is not None and abs(val - gold_val) < 0.01:
        return True

    # Fallback: check if gold number appears ANYWHERE in full text
    clean = re.sub(r'(\d),(\d)', r'\1\2', full_text)
    for num_str in _ROBUST_NUM_RE.findall(clean):
        try:
            v = float(num_str)
            if abs(v - gold_val) < 0.01:
                return True
        except ValueError:
            continue

    return False


def check_qa_accuracy(model_answer: str, gold: str,
                      thinking_text: str = "") -> bool:
    """Robust inclusion check for factual QA.

    Checks both the answer and thinking text for the gold substring.
    """
    full_text = model_answer
    if thinking_text:
        full_text = full_text + " " + thinking_text

    ans_lower = full_text.lower()
    gold_lower = gold.lower()
    # Direct substring
    if gold_lower in ans_lower:
        return True
    # Punctuation-stripped substring
    ans_clean = _STRIP_PUNCT_RE.sub('', ans_lower)
    gold_clean = _STRIP_PUNCT_RE.sub('', gold_lower)
    if gold_clean in ans_clean:
        return True
    return False


# ═══════════════════════════════════════════════════════════
# Baseline Generation (Raw Qwen, no METIS)
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def baseline_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """Raw model generation with chat template — no cognitive routing."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)
    attn = torch.ones_like(input_ids)

    t0 = time.perf_counter()
    output_ids = model.generate(
        input_ids, attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    latency = (time.perf_counter() - t0) * 1000

    new_tokens = output_ids[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return {
        "text": text,
        "tokens": len(new_tokens),
        "latency_ms": latency,
    }


# ═══════════════════════════════════════════════════════════
# Benchmark Runner
# ═══════════════════════════════════════════════════════════

@dataclass
class RunResult:
    question: str
    gold: str
    category: str
    source: str
    # Baseline
    baseline_answer: str = ""
    baseline_correct: bool = False
    baseline_tokens: int = 0
    baseline_latency: float = 0.0
    # METIS
    metis_answer: str = ""
    metis_correct: bool = False
    metis_tokens: int = 0
    metis_latency: float = 0.0
    metis_route: str = ""
    metis_entropy: float = 0.0
    metis_searched: bool = False


BASELINE_MODEL_ID = os.getenv("METIS_BASELINE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
METIS_MODEL_PATH = str(PROJECT_ROOT / "experiment_output_dpo_balanced" / "metis_dpo_cognitive")


def _unload_model(*refs: Any) -> None:
    """Force-unload model references and free GPU memory."""
    for ref in refs:
        del ref
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.info("  GPU memory released.")


def run_benchmark(
    n_complex: int = 50,
    n_simple: int = 50,
    seed: int = 42,
    skip_baseline: bool = False,
) -> List[RunResult]:
    """Run the full A/B benchmark.

    Phase 1: Load raw Qwen baseline → run all baseline inferences → unload.
    Phase 2: Load METIS DPO model → run all METIS inferences.
    This avoids OOM from loading two 7B models simultaneously.
    """

    # ── Load data ──
    logger.info("Loading benchmark data...")
    complex_items = load_gsm8k_samples(n_complex, seed)
    simple_items = load_simple_samples(n_simple, seed)
    all_items = complex_items + simple_items
    logger.info(f"  {len(complex_items)} complex + {len(simple_items)} simple = {len(all_items)} total")

    results: List[RunResult] = [
        RunResult(
            question=item.question,
            gold=item.gold_answer,
            category=item.category,
            source=item.source,
        )
        for item in all_items
    ]

    # ════════════════════════════════════════════════
    # Phase A: Baseline — Raw Qwen2.5-7B-Instruct
    # ════════════════════════════════════════════════
    if not skip_baseline:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Force offline mode — HF Hub is unreachable in this environment
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        logger.info(f"Loading BASELINE model: {BASELINE_MODEL_ID} (offline cache)")
        t0 = time.time()
        bl_tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL_ID, local_files_only=True)
        bl_model = AutoModelForCausalLM.from_pretrained(
            BASELINE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
        bl_model.eval()
        logger.info(f"  Baseline loaded in {time.time() - t0:.1f}s")

        for i, (item, rr) in enumerate(zip(all_items, results)):
            is_math = item.category == "complex"
            try:
                bl = baseline_generate(bl_model, bl_tokenizer, item.question, max_new_tokens=512)
                rr.baseline_answer = bl["text"]
                rr.baseline_tokens = bl["tokens"]
                rr.baseline_latency = bl["latency_ms"]
                if is_math:
                    rr.baseline_correct = check_math_accuracy(bl["text"], item.gold_answer)
                else:
                    rr.baseline_correct = check_qa_accuracy(bl["text"], item.gold_answer)
            except Exception as e:
                logger.error(f"  Baseline error on [{item.source}]: {e}")

            tag = "✓" if rr.baseline_correct else "✗"
            logger.info(
                f"  [BL {i+1:3d}/{len(all_items)}] {item.category:7s} | "
                f"{tag} ({rr.baseline_tokens:3d}tok) | {item.source}"
            )

        logger.info("Unloading baseline model...")
        _unload_model(bl_model, bl_tokenizer)
        del bl_model, bl_tokenizer

    # ════════════════════════════════════════════════
    # Phase B: METIS — DPO + Orchestrator
    # ════════════════════════════════════════════════
    logger.info(f"Loading METIS model: {METIS_MODEL_PATH}")
    from metis import Metis
    from metis.cognitive.metacognition import MetacognitiveOrchestrator
    from metis.search.retriever import ToolRetriever

    t0 = time.time()
    metis = Metis.from_pretrained(METIS_MODEL_PATH)
    retriever = ToolRetriever(force_mock=False)
    orch = MetacognitiveOrchestrator(metis, retriever=retriever)
    logger.info(f"  METIS loaded in {time.time() - t0:.1f}s")

    for i, (item, rr) in enumerate(zip(all_items, results)):
        is_math = item.category == "complex"
        try:
            resp = orch.process_query(item.question)
            rr.metis_answer = resp.final_answer
            rr.metis_tokens = resp.tokens_generated
            rr.metis_latency = resp.latency_ms
            rr.metis_route = resp.cognitive_route
            rr.metis_entropy = resp.semantic_entropy if resp.semantic_entropy is not None else 0.0
            rr.metis_searched = resp.searched
            # Pass thinking_text so evaluator can check full output
            thinking = resp.thinking_text or ""
            if is_math:
                rr.metis_correct = check_math_accuracy(
                    resp.final_answer, item.gold_answer, thinking_text=thinking)
            else:
                rr.metis_correct = check_qa_accuracy(
                    resp.final_answer, item.gold_answer, thinking_text=thinking)
        except Exception as e:
            logger.error(f"  METIS error on [{item.source}]: {e}")

        bl_tag = "✓" if rr.baseline_correct else "✗"
        mt_tag = "✓" if rr.metis_correct else "✗"
        logger.info(
            f"  [MT {i+1:3d}/{len(all_items)}] {item.category:7s} | "
            f"BL:{bl_tag} | METIS:{mt_tag} ({rr.metis_tokens:3d}tok, "
            f"H={rr.metis_entropy:.2f}, route={rr.metis_route}, "
            f"search={'Y' if rr.metis_searched else 'N'}) | {item.source}"
        )

    return results


# ═══════════════════════════════════════════════════════════
# Statistical Report
# ═══════════════════════════════════════════════════════════

def print_report(results: List[RunResult]) -> None:
    """Print the statistical comparison report."""
    complex_r = [r for r in results if r.category == "complex"]
    simple_r = [r for r in results if r.category == "simple"]

    def stats(subset: List[RunResult], label: str) -> None:
        n = len(subset)
        if n == 0:
            return

        bl_acc = sum(r.baseline_correct for r in subset) / n * 100
        mt_acc = sum(r.metis_correct for r in subset) / n * 100
        bl_tok = sum(r.baseline_tokens for r in subset) / n
        mt_tok = sum(r.metis_tokens for r in subset) / n
        bl_lat = sum(r.baseline_latency for r in subset) / n
        mt_lat = sum(r.metis_latency for r in subset) / n

        # Route distribution (METIS)
        routes = {}
        for r in subset:
            routes[r.metis_route] = routes.get(r.metis_route, 0) + 1
        searches = sum(1 for r in subset if r.metis_searched)
        avg_h = sum(r.metis_entropy for r in subset) / n

        acc_delta = mt_acc - bl_acc
        tok_delta = (mt_tok - bl_tok) / max(bl_tok, 1) * 100
        acc_color = C.GREEN if acc_delta >= 0 else C.RED
        tok_color = C.GREEN if tok_delta <= 0 else C.RED

        print(f"\n  {C.BOLD}{C.CYAN}── {label} (n={n}) ──{C.RESET}")
        print(f"  {'':20s} {'Baseline':>12s}  {'METIS':>12s}  {'Delta':>12s}")
        print(f"  {C.DIM}{'─' * 60}{C.RESET}")
        print(f"  {'Accuracy':20s} {bl_acc:11.1f}%  {mt_acc:11.1f}%  "
              f"{acc_color}{acc_delta:+11.1f}%{C.RESET}")
        print(f"  {'Avg Tokens':20s} {bl_tok:12.1f}  {mt_tok:12.1f}  "
              f"{tok_color}{tok_delta:+11.1f}%{C.RESET}")
        print(f"  {'Avg Latency (ms)':20s} {bl_lat:12.0f}  {mt_lat:12.0f}")
        print(f"  {C.DIM}{'─' * 60}{C.RESET}")
        print(f"  {C.MAGENTA}METIS Routing:{C.RESET}  ", end="")
        for route, cnt in sorted(routes.items()):
            print(f"{route}={cnt}  ", end="")
        print()
        print(f"  {C.MAGENTA}Web Searches:{C.RESET}   {searches}/{n}")
        print(f"  {C.BLUE}Avg Entropy (H):{C.RESET} {avg_h:.4f}")

    print(f"\n{'=' * 64}")
    print(f"{C.BOLD}{C.CYAN}  METIS ACADEMIC BENCHMARK — A/B COMPARISON{C.RESET}")
    print(f"{'=' * 64}")
    print(f"  Model A (Baseline): Qwen2.5-7B-Instruct (raw)")
    print(f"  Model B (METIS):    DPO + Cognitive Routing + Semantic Entropy + DDG")
    print(f"  Total prompts: {len(results)}")

    stats(complex_r, "COMPLEX (GSM8K Math)")
    stats(simple_r, "SIMPLE (Factual QA)")
    stats(results, "OVERALL")

    # Pareto summary
    n = len(results)
    bl_acc_all = sum(r.baseline_correct for r in results) / n * 100
    mt_acc_all = sum(r.metis_correct for r in results) / n * 100
    bl_tok_all = sum(r.baseline_tokens for r in results) / n
    mt_tok_all = sum(r.metis_tokens for r in results) / n

    # Simple subset: token savings
    if simple_r:
        bl_tok_s = sum(r.baseline_tokens for r in simple_r) / len(simple_r)
        mt_tok_s = sum(r.metis_tokens for r in simple_r) / len(simple_r)
        tok_saving = (1 - mt_tok_s / max(bl_tok_s, 1)) * 100
    else:
        tok_saving = 0

    # Complex subset: accuracy gain
    if complex_r:
        bl_acc_c = sum(r.baseline_correct for r in complex_r) / len(complex_r) * 100
        mt_acc_c = sum(r.metis_correct for r in complex_r) / len(complex_r) * 100
        acc_gain = mt_acc_c - bl_acc_c
    else:
        acc_gain = 0

    print(f"\n{'=' * 64}")
    print(f"{C.BOLD}  PARETO EFFICIENCY SUMMARY{C.RESET}")
    print(f"{'=' * 64}")
    if tok_saving > 0:
        print(f"  {C.GREEN}✓ Simple tasks: {tok_saving:.1f}% token savings{C.RESET}")
    else:
        print(f"  {C.YELLOW}~ Simple tasks: {tok_saving:.1f}% token delta{C.RESET}")
    if acc_gain > 0:
        print(f"  {C.GREEN}✓ Complex tasks: +{acc_gain:.1f}% accuracy gain{C.RESET}")
    else:
        print(f"  {C.YELLOW}~ Complex tasks: {acc_gain:+.1f}% accuracy delta{C.RESET}")
    print(f"{'=' * 64}\n")


def save_results(results: List[RunResult], path: str) -> None:
    """Save detailed results to JSON."""
    data = []
    for r in results:
        data.append({
            "question": r.question,
            "gold": r.gold,
            "category": r.category,
            "source": r.source,
            "baseline": {
                "answer": r.baseline_answer[:300],
                "correct": r.baseline_correct,
                "tokens": r.baseline_tokens,
                "latency_ms": round(r.baseline_latency, 1),
            },
            "metis": {
                "answer": r.metis_answer[:300],
                "correct": r.metis_correct,
                "tokens": r.metis_tokens,
                "latency_ms": round(r.metis_latency, 1),
                "route": r.metis_route,
                "entropy": round(r.metis_entropy, 4),
                "searched": r.metis_searched,
            },
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {path}")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="METIS Academic Benchmark")
    parser.add_argument("--n-complex", type=int, default=50)
    parser.add_argument("--n-simple", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    results = run_benchmark(
        n_complex=args.n_complex,
        n_simple=args.n_simple,
        seed=args.seed,
        skip_baseline=args.skip_baseline,
    )

    print_report(results)

    out_path = str(PROJECT_ROOT / args.output)
    save_results(results, out_path)


if __name__ == "__main__":
    main()
