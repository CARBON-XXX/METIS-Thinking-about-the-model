#!/usr/bin/env python3
"""
METIS Phase 5 — Orthogonal Cognitive Routing Benchmark
=======================================================

Diagnoses "Spurious Correlation" and "Feature Entanglement" in the
DPO-trained model by testing routing accuracy across a 2×2 matrix:

              Simple          Complex
  Math     Quadrant A       Quadrant B
  Text     Quadrant C       Quadrant D

Each quadrant has 5 hardcoded prompts with known expected routing.
Near-greedy decoding (temperature=0.1, do_sample=False) to observe
the model's default policy without sampling noise.
"""

import re
import sys
import torch
from typing import Any, Dict, List, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────

MODEL_PATH = "experiment_output_dpo_balanced/metis_dpo_cognitive"

SYSTEM_PROMPT = (
    "You are METIS, an AI with a dynamic cognitive routing layer. "
    "Analyze the complexity of the user's request and allocate compute accordingly."
)

TAG_FAST = "FAST"
TAG_DEEP = "DEEP"
TAG_NONE = "NONE"

_COGNITIVE_TAG_RE = re.compile(r"\[COGNITIVE_STATE:\s*(FAST|DEEP)\]", re.IGNORECASE)
_THINKING_OPEN_RE = re.compile(r"<thinking>", re.IGNORECASE)
_THINKING_CLOSE_RE = re.compile(r"</thinking>", re.IGNORECASE)

# ─────────────────────────────────────────────────────
# Orthogonal Test Matrix (5 prompts × 4 quadrants)
# ─────────────────────────────────────────────────────

QUADRANTS: Dict[str, Dict[str, Any]] = {
    "A: Simple Math": {
        "expected_tag": TAG_FAST,
        "expect_thinking": False,
        "prompts": [
            "What is 5 + 7?",
            "If I have 3 apples and eat 1, how many are left?",
            "What is 10 × 4?",
            "How much is 100 divided by 5?",
            "What is 15 - 8?",
        ],
    },
    "B: Complex Math": {
        "expected_tag": TAG_DEEP,
        "expect_thinking": True,
        "prompts": [
            "Solve the system of equations: 3x + 2y = 16 and x - y = 2.",
            "A train travels 60mph for 2 hours, then 80mph for 1.5 hours. What is the average speed for the entire trip?",
            "A store offers a 20% discount on a $150 item, then charges 8% sales tax. A second store sells the same item for $130 with 8% tax and no discount. Which store offers the better deal and by how much?",
            "If the probability of rain on any given day is 0.3, what is the probability that it rains on exactly 2 out of 5 days?",
            "A cylindrical tank has a radius of 3 meters and a height of 10 meters. How many liters of water can it hold? (1 cubic meter = 1000 liters)",
        ],
    },
    "C: Simple Text": {
        "expected_tag": TAG_FAST,
        "expect_thinking": False,
        "prompts": [
            "What is the capital of Japan?",
            "Translate 'hello' to French.",
            "What color do you get when you mix red and blue?",
            "Name the largest planet in our solar system.",
            "What does the abbreviation 'NASA' stand for?",
        ],
    },
    "D: Complex Text/Logic": {
        "expected_tag": TAG_DEEP,
        "expect_thinking": True,
        "prompts": [
            "There are 3 boxes. One contains only apples, one only oranges, and one both. All labels are wrong. You can pick one fruit from one box. How do you correctly label all boxes?",
            "Analyze the philosophical implications of the Ship of Theseus paradox.",
            "A farmer needs to cross a river with a wolf, a goat, and a cabbage. The boat can only carry the farmer and one item. The wolf will eat the goat if left alone, and the goat will eat the cabbage. How does the farmer get everything across safely?",
            "Compare and contrast the economic philosophies of Keynesian economics and Austrian economics. What are the key points of disagreement?",
            "You have 12 identical-looking balls. One is a different weight (heavier or lighter). Using a balance scale only 3 times, how do you find the odd ball and determine if it's heavier or lighter?",
        ],
    },
}

# ─────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────

def parse_response(text: str) -> Dict[str, Any]:
    """Parse cognitive tag, thinking presence, and format integrity."""
    tag_match = _COGNITIVE_TAG_RE.search(text)
    tag = tag_match.group(1).upper() if tag_match else TAG_NONE

    has_thinking_open = bool(_THINKING_OPEN_RE.search(text))
    has_thinking_close = bool(_THINKING_CLOSE_RE.search(text))
    has_thinking = has_thinking_open and has_thinking_close
    thinking_broken = has_thinking_open != has_thinking_close

    return {
        "tag": tag,
        "has_thinking": has_thinking,
        "thinking_broken": thinking_broken,
        "has_thinking_open": has_thinking_open,
        "has_thinking_close": has_thinking_close,
    }


# ─────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────

def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
) -> str:
    """Generate near-greedy response for a single prompt."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=False)
    if tokenizer.eos_token:
        response = response.split(tokenizer.eos_token)[0]
    return response


# ─────────────────────────────────────────────────────
# Evaluation & Reporting
# ─────────────────────────────────────────────────────

def evaluate_quadrant(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    quadrant_name: str,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Run and evaluate all prompts in a quadrant."""
    results: List[Dict[str, Any]] = []
    expected_tag = config["expected_tag"]
    expect_thinking = config["expect_thinking"]

    for i, prompt in enumerate(config["prompts"]):
        response = generate_response(model, tokenizer, prompt)
        parsed = parse_response(response)

        # Routing accuracy
        routing_correct = parsed["tag"] == expected_tag

        # Format integrity
        format_ok = True
        format_issue = ""
        if expected_tag == TAG_FAST:
            if parsed["has_thinking"]:
                format_ok = False
                format_issue = "LEAKAGE: FAST but has <thinking>"
        elif expected_tag == TAG_DEEP:
            if not parsed["has_thinking"]:
                format_ok = False
                format_issue = "MISSING: DEEP but no <thinking>"
        if parsed["thinking_broken"]:
            format_ok = False
            format_issue += " BROKEN_TAGS"

        results.append({
            "quadrant": quadrant_name,
            "prompt": prompt,
            "expected_tag": expected_tag,
            "actual_tag": parsed["tag"],
            "routing_correct": routing_correct,
            "expect_thinking": expect_thinking,
            "has_thinking": parsed["has_thinking"],
            "format_ok": format_ok,
            "format_issue": format_issue,
            "response_preview": response[:200],
        })

    return results


def print_report(all_results: List[Dict[str, Any]]) -> None:
    """Print formatted orthogonal benchmark report."""
    W = 72

    print("\n" + "=" * W)
    print("  METIS ORTHOGONAL COGNITIVE ROUTING BENCHMARK")
    print("=" * W)

    # Per-quadrant detail
    quadrant_stats: Dict[str, Dict[str, Any]] = {}

    current_quad = None
    for r in all_results:
        q = r["quadrant"]
        if q != current_quad:
            current_quad = q
            print(f"\n{'─' * W}")
            print(f"  {q}  |  Expected: [{r['expected_tag']}]  "
                  f"{'+ <thinking>' if r['expect_thinking'] else '(no thinking)'}")
            print(f"{'─' * W}")

        if q not in quadrant_stats:
            quadrant_stats[q] = {
                "total": 0, "routing_ok": 0, "format_ok": 0,
                "tags": {TAG_FAST: 0, TAG_DEEP: 0, TAG_NONE: 0},
                "expected": r["expected_tag"],
                "leakage": 0,
            }

        stats = quadrant_stats[q]
        stats["total"] += 1
        if r["routing_correct"]:
            stats["routing_ok"] += 1
        if r["format_ok"]:
            stats["format_ok"] += 1
        stats["tags"][r["actual_tag"]] += 1

        # Detect leakage
        if r["actual_tag"] == TAG_FAST and r["has_thinking"]:
            stats["leakage"] += 1

        route_icon = "✓" if r["routing_correct"] else "✗"
        fmt_icon = "✓" if r["format_ok"] else "⚠"
        prompt_short = r["prompt"][:50] + ("..." if len(r["prompt"]) > 50 else "")
        tag_str = r["actual_tag"]
        think_str = "T" if r["has_thinking"] else "-"

        print(f"  {route_icon} {fmt_icon}  [{tag_str:4s}|{think_str}]  {prompt_short}")
        if r["format_issue"]:
            print(f"         └─ {r['format_issue']}")

    # Summary matrix
    print(f"\n{'=' * W}")
    print("  SUMMARY MATRIX")
    print(f"{'=' * W}")
    print(f"{'':20s} {'Routing Acc':>12s} {'Format OK':>12s} {'Leakage':>10s} {'Tag Distribution':>20s}")
    print(f"{'─' * W}")

    total_routing = 0
    total_format = 0
    total_n = 0

    for q_name, stats in quadrant_stats.items():
        n = stats["total"]
        r_acc = stats["routing_ok"] / n * 100
        f_acc = stats["format_ok"] / n * 100
        tag_dist = f"F={stats['tags']['FAST']} D={stats['tags']['DEEP']} N={stats['tags']['NONE']}"
        leak = stats["leakage"]

        total_routing += stats["routing_ok"]
        total_format += stats["format_ok"]
        total_n += n

        print(f"  {q_name:18s} {r_acc:10.0f}% {f_acc:10.0f}% {leak:8d}   {tag_dist}")

    print(f"{'─' * W}")
    overall_r = total_routing / total_n * 100
    overall_f = total_format / total_n * 100
    print(f"  {'OVERALL':18s} {overall_r:10.1f}% {overall_f:10.1f}%")

    # Domain bias analysis
    print(f"\n{'=' * W}")
    print("  DOMAIN BIAS ANALYSIS")
    print(f"{'=' * W}")

    # Math domain: Quadrant A + B
    math_quads = [q for q in quadrant_stats if "Math" in q]
    text_quads = [q for q in quadrant_stats if "Text" in q or "Logic" in q]

    math_deep = sum(quadrant_stats[q]["tags"][TAG_DEEP] for q in math_quads)
    math_fast = sum(quadrant_stats[q]["tags"][TAG_FAST] for q in math_quads)
    math_none = sum(quadrant_stats[q]["tags"][TAG_NONE] for q in math_quads)
    math_total = math_deep + math_fast + math_none

    text_deep = sum(quadrant_stats[q]["tags"][TAG_DEEP] for q in text_quads)
    text_fast = sum(quadrant_stats[q]["tags"][TAG_FAST] for q in text_quads)
    text_none = sum(quadrant_stats[q]["tags"][TAG_NONE] for q in text_quads)
    text_total = text_deep + text_fast + text_none

    print(f"  Math domain (n={math_total}):  DEEP={math_deep} ({math_deep/math_total*100:.0f}%)  "
          f"FAST={math_fast} ({math_fast/math_total*100:.0f}%)  NONE={math_none}")
    print(f"  Text domain (n={text_total}):  DEEP={text_deep} ({text_deep/text_total*100:.0f}%)  "
          f"FAST={text_fast} ({text_fast/text_total*100:.0f}%)  NONE={text_none}")

    # Complexity axis
    simple_quads = [q for q in quadrant_stats if "Simple" in q]
    complex_quads = [q for q in quadrant_stats if "Complex" in q]

    simple_deep = sum(quadrant_stats[q]["tags"][TAG_DEEP] for q in simple_quads)
    simple_fast = sum(quadrant_stats[q]["tags"][TAG_FAST] for q in simple_quads)
    simple_none = sum(quadrant_stats[q]["tags"][TAG_NONE] for q in simple_quads)
    simple_total = simple_deep + simple_fast + simple_none

    complex_deep = sum(quadrant_stats[q]["tags"][TAG_DEEP] for q in complex_quads)
    complex_fast = sum(quadrant_stats[q]["tags"][TAG_FAST] for q in complex_quads)
    complex_none = sum(quadrant_stats[q]["tags"][TAG_NONE] for q in complex_quads)
    complex_total = complex_deep + complex_fast + complex_none

    print(f"\n  Simple tasks (n={simple_total}): DEEP={simple_deep} ({simple_deep/simple_total*100:.0f}%)  "
          f"FAST={simple_fast} ({simple_fast/simple_total*100:.0f}%)  NONE={simple_none}")
    print(f"  Complex tasks (n={complex_total}): DEEP={complex_deep} ({complex_deep/complex_total*100:.0f}%)  "
          f"FAST={complex_fast} ({complex_fast/complex_total*100:.0f}%)  NONE={complex_none}")

    # Bias verdict
    print(f"\n  {'─' * (W - 4)}")
    math_bias = math_deep / math_total if math_total else 0
    text_bias = text_fast / text_total if text_total else 0
    complexity_discrimination = (complex_deep / complex_total if complex_total else 0) - \
                                (simple_deep / simple_total if simple_total else 0)

    if math_bias > 0.8:
        print(f"  ⚠ MATH→DEEP BIAS: {math_bias*100:.0f}% of math routed to DEEP (including simple)")
    if text_bias > 0.8:
        print(f"  ⚠ TEXT→FAST BIAS: {text_bias*100:.0f}% of text routed to FAST (including complex)")
    if abs(complexity_discrimination) < 0.2:
        print(f"  ⚠ WEAK COMPLEXITY DISCRIMINATION: DEEP rate diff = {complexity_discrimination*100:+.0f}pp "
              f"(complex vs simple)")
    elif complexity_discrimination > 0.3:
        print(f"  ✓ GOOD COMPLEXITY DISCRIMINATION: DEEP rate diff = {complexity_discrimination*100:+.0f}pp")

    if overall_r >= 70:
        print(f"  ✓ Overall routing accuracy {overall_r:.0f}% is ACCEPTABLE")
    elif overall_r >= 50:
        print(f"  ⚠ Overall routing accuracy {overall_r:.0f}% is MARGINAL")
    else:
        print(f"  ✗ Overall routing accuracy {overall_r:.0f}% is POOR — model needs retraining")

    print(f"\n{'=' * W}")


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────

def main() -> None:
    print("Loading DPO model for orthogonal benchmark...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()

    print(f"Model loaded. Running 20 prompts across 4 quadrants...\n")

    all_results: List[Dict[str, Any]] = []
    for q_name, config in QUADRANTS.items():
        print(f"  Evaluating {q_name}...")
        results = evaluate_quadrant(model, tokenizer, q_name, config)
        all_results.extend(results)

    print_report(all_results)


if __name__ == "__main__":
    main()
