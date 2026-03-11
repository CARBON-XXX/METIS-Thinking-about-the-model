#!/usr/bin/env python3
"""
GRPO vs DPO Quick Evaluation
=============================
Compare GRPO merged model against DPO baseline on:
  - GSM8K accuracy (complex math)
  - Simple QA accuracy + verbosity (token count)

Loads models sequentially to avoid OOM.
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
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ── Paths ──
DPO_MODEL = str(PROJECT_ROOT / "experiment_output_dpo_balanced" / "metis_dpo_cognitive")
GRPO_MODEL = str(PROJECT_ROOT / "experiment_output_grpo_final" / "metis_grpo_merged")

# ── Simple QA pool ──
_SIMPLE_QA = [
    {"q": "What is the capital of France?", "a": "Paris"},
    {"q": "What is the capital of Japan?", "a": "Tokyo"},
    {"q": "What is the capital of Germany?", "a": "Berlin"},
    {"q": "What is the capital of Italy?", "a": "Rome"},
    {"q": "What is the capital of Australia?", "a": "Canberra"},
    {"q": "What is the capital of Canada?", "a": "Ottawa"},
    {"q": "What is the capital of Brazil?", "a": "Brasilia"},
    {"q": "What is the boiling point of water in Celsius?", "a": "100"},
    {"q": "What is the chemical symbol for gold?", "a": "Au"},
    {"q": "What is the chemical symbol for water?", "a": "H2O"},
    {"q": "Who wrote Romeo and Juliet?", "a": "Shakespeare"},
    {"q": "What planet is closest to the Sun?", "a": "Mercury"},
    {"q": "What is the largest ocean on Earth?", "a": "Pacific"},
    {"q": "How many continents are there?", "a": "7"},
    {"q": "What is the speed of light in km/s (approx)?", "a": "300000"},
    {"q": "What gas do plants absorb from the atmosphere?", "a": "carbon dioxide"},
    {"q": "Who painted the Mona Lisa?", "a": "Leonardo da Vinci"},
    {"q": "What is the square root of 144?", "a": "12"},
    {"q": "What is the largest planet in our solar system?", "a": "Jupiter"},
    {"q": "What year did World War II end?", "a": "1945"},
    {"q": "What is 15 multiplied by 3?", "a": "45"},
    {"q": "What is the freezing point of water in Fahrenheit?", "a": "32"},
    {"q": "Who discovered penicillin?", "a": "Fleming"},
    {"q": "What is the capital of Spain?", "a": "Madrid"},
    {"q": "How many sides does a hexagon have?", "a": "6"},
    {"q": "What is the chemical formula for table salt?", "a": "NaCl"},
    {"q": "Who was the first person to walk on the Moon?", "a": "Armstrong"},
    {"q": "What is the smallest prime number?", "a": "2"},
    {"q": "What is the capital of Egypt?", "a": "Cairo"},
    {"q": "What element has atomic number 1?", "a": "Hydrogen"},
]

_ROBUST_NUM_RE = re.compile(r"[-+]?\d*\.\d+|\d+")


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


@dataclass
class EvalResult:
    question: str
    gold: str
    category: str
    answer: str = ""
    correct: bool = False
    tokens: int = 0
    latency_ms: float = 0.0


@torch.no_grad()
def generate_answer(
    model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 512
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)
    attn = torch.ones_like(input_ids)
    t0 = time.perf_counter()
    output_ids = model.generate(
        input_ids, attention_mask=attn,
        max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    latency = (time.perf_counter() - t0) * 1000
    new_tokens = output_ids[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return {"text": text, "tokens": len(new_tokens), "latency_ms": latency}


def load_gsm8k_samples(n: int, seed: int) -> List[Dict[str, str]]:
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    items = []
    for idx in indices:
        row = ds[idx]
        answer_text = row["answer"].split("####")[-1].strip()
        items.append({"q": row["question"], "a": answer_text, "cat": "complex"})
    return items


def evaluate_model(
    model_path: str, model_name: str, items: List[Dict[str, str]]
) -> List[EvalResult]:
    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name}")
    print(f"  Path: {model_path}")
    print(f"{'='*60}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Loaded in {time.time() - t0:.1f}s")

    results: List[EvalResult] = []
    for i, item in enumerate(items):
        r = EvalResult(question=item["q"], gold=item["a"], category=item["cat"])
        try:
            out = generate_answer(model, tokenizer, item["q"])
            r.answer = out["text"]
            r.tokens = out["tokens"]
            r.latency_ms = out["latency_ms"]
            if item["cat"] == "complex":
                r.correct = check_math_accuracy(out["text"], item["a"])
            else:
                r.correct = check_qa_accuracy(out["text"], item["a"])
        except Exception as e:
            print(f"  ERROR on item {i}: {e}")
        results.append(r)
        tag = "✓" if r.correct else "✗"
        if (i + 1) % 10 == 0 or i == len(items) - 1:
            done = sum(1 for rr in results if rr.correct)
            print(f"  [{i+1}/{len(items)}] acc={done}/{i+1} ({100*done/(i+1):.1f}%)")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  GPU memory released.")

    return results


def print_comparison(
    dpo_results: List[EvalResult], grpo_results: List[EvalResult]
) -> None:
    print(f"\n{'='*64}")
    print(f"  GRPO vs DPO Comparison Report")
    print(f"{'='*64}\n")

    for cat in ["complex", "simple"]:
        dpo_cat = [r for r in dpo_results if r.category == cat]
        grpo_cat = [r for r in grpo_results if r.category == cat]
        n = len(dpo_cat)

        dpo_acc = sum(1 for r in dpo_cat if r.correct) / max(n, 1) * 100
        grpo_acc = sum(1 for r in grpo_cat if r.correct) / max(n, 1) * 100
        dpo_tokens = sum(r.tokens for r in dpo_cat) / max(n, 1)
        grpo_tokens = sum(r.tokens for r in grpo_cat) / max(n, 1)

        delta_acc = grpo_acc - dpo_acc
        delta_tokens = (grpo_tokens - dpo_tokens) / max(dpo_tokens, 1) * 100

        label = "Complex (GSM8K)" if cat == "complex" else "Simple (QA)"
        print(f"  {label} (n={n}):")
        print(f"    {'Metric':<20} {'DPO':>10} {'GRPO':>10} {'Delta':>10}")
        print(f"    {'-'*50}")
        print(f"    {'Accuracy':<20} {dpo_acc:>9.1f}% {grpo_acc:>9.1f}% {delta_acc:>+9.1f}%")
        print(f"    {'Avg Tokens':<20} {dpo_tokens:>10.1f} {grpo_tokens:>10.1f} {delta_tokens:>+9.1f}%")
        print()

    # Overall
    dpo_total_acc = sum(1 for r in dpo_results if r.correct) / max(len(dpo_results), 1) * 100
    grpo_total_acc = sum(1 for r in grpo_results if r.correct) / max(len(grpo_results), 1) * 100
    dpo_total_tokens = sum(r.tokens for r in dpo_results) / max(len(dpo_results), 1)
    grpo_total_tokens = sum(r.tokens for r in grpo_results) / max(len(grpo_results), 1)

    print(f"  Overall (n={len(dpo_results)}):")
    print(f"    {'Metric':<20} {'DPO':>10} {'GRPO':>10} {'Delta':>10}")
    print(f"    {'-'*50}")
    print(f"    {'Accuracy':<20} {dpo_total_acc:>9.1f}% {grpo_total_acc:>9.1f}% {grpo_total_acc - dpo_total_acc:>+9.1f}%")
    print(f"    {'Avg Tokens':<20} {dpo_total_tokens:>10.1f} {grpo_total_tokens:>10.1f} {(grpo_total_tokens - dpo_total_tokens) / max(dpo_total_tokens, 1) * 100:>+9.1f}%")
    print(f"\n{'='*64}")


def main() -> None:
    seed = 42
    n_complex = 50
    n_simple = 30

    # Load data
    print("Loading evaluation data...")
    gsm8k_items = load_gsm8k_samples(n_complex, seed)
    rng = random.Random(seed)
    simple_items = [{"q": q["q"], "a": q["a"], "cat": "simple"} for q in rng.sample(_SIMPLE_QA, min(n_simple, len(_SIMPLE_QA)))]
    all_items = gsm8k_items + simple_items
    print(f"  {len(gsm8k_items)} complex + {len(simple_items)} simple = {len(all_items)} total")

    # Phase 1: DPO baseline
    dpo_results = evaluate_model(DPO_MODEL, "DPO Baseline", all_items)

    # Phase 2: GRPO merged
    grpo_results = evaluate_model(GRPO_MODEL, "GRPO Merged", all_items)

    # Print comparison
    print_comparison(dpo_results, grpo_results)


if __name__ == "__main__":
    main()
