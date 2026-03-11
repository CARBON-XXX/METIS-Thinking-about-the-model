#!/usr/bin/env python3
"""
METIS Phase 8 — Inference Router Gateway
=========================================
Deterministic state-machine wrapper that sits on top of the DPO-balanced
model and guarantees well-formed structured output regardless of the
model's autoregressive syntax degradation.

The gateway intercepts raw generation, classifies the cognitive route,
repairs broken `<thinking>` closures, and returns a clean JSON envelope:

    {"route": str, "thinking": str | None, "answer": str}

Three cases:
  Case A (Explicit DEEP)  — `[COGNITIVE_STATE: DEEP]` detected
  Case B (Explicit FAST)  — `[COGNITIVE_STATE: FAST]` detected
  Case C (Implicit FAST)  — No cognitive tag → short-circuit as FAST
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────

MODEL_PATH = "experiment_output_dpo_balanced/metis_dpo_cognitive"

SYSTEM_PROMPT = (
    "You are METIS, an AI with a dynamic cognitive routing layer. "
    "Analyze the complexity of the user's request and allocate compute accordingly."
)

# Regex patterns
_TAG_RE = re.compile(r"\[COGNITIVE_STATE:\s*(FAST|DEEP)\]", re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<thinking>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</thinking>", re.IGNORECASE)

# Common answer-boundary prefixes used to forcefully close an unclosed
# <thinking> block when the model forgets </thinking>.
_ANSWER_BOUNDARY_RE = re.compile(
    r"(?:^|\n)\s*(?:"
    r"FINAL\s*ANSWER\s*[:\-]"
    r"|(?:So|Therefore|Thus|Hence|In\s+(?:summary|conclusion)),?\s"
    r"|(?:The\s+(?:answer|result|solution)\s+is)"
    r"|(?:\*\*(?:Answer|Solution|Result)\*\*)"
    r"|(?:#{1,3}\s*(?:Answer|Solution|Result))"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

# ─────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("metis_gateway")

# ─────────────────────────────────────────────────────
# Model loader (singleton)
# ─────────────────────────────────────────────────────

_model = None
_tokenizer = None


def _load_model() -> None:
    """Lazy-load model and tokenizer once."""
    global _model, _tokenizer
    if _model is not None:
        return

    logger.info(f"Loading model from {MODEL_PATH} ...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    _model.eval()

    mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    logger.info(f"  Model loaded. VRAM: {mem_gb:.1f} GB")


# ─────────────────────────────────────────────────────
# Raw generation
# ─────────────────────────────────────────────────────


def _generate_raw(prompt: str, max_new_tokens: int = 1024) -> str:
    """Run a single chat-templated generation and return decoded text."""
    _load_model()
    assert _model is not None and _tokenizer is not None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    input_ids = _tokenizer.apply_chat_template(
        messages, tokenize=True, return_tensors="pt",
        add_generation_prompt=True,
    ).to(_model.device)

    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        output_ids = _model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=_tokenizer.pad_token_id,
        )

    # Decode only the newly generated tokens
    # CRITICAL: skip_special_tokens=False to preserve cognitive tags
    # which are registered as special tokens in the tokenizer.
    new_tokens = output_ids[0, input_ids.shape[1]:]
    raw = _tokenizer.decode(new_tokens, skip_special_tokens=False)
    if _tokenizer.eos_token:
        raw = raw.split(_tokenizer.eos_token)[0]
    return raw.strip()


# ─────────────────────────────────────────────────────
# Deterministic State Machine (Parser)
# ─────────────────────────────────────────────────────


def _strip_tag(text: str, tag_match: re.Match) -> str:  # type: ignore[type-arg]
    """Remove the cognitive tag from the text, return remainder."""
    return (text[:tag_match.start()] + text[tag_match.end():]).strip()


def _repair_deep_output(text_after_tag: str) -> Dict[str, Optional[str]]:
    """Parse a DEEP output, repairing broken <thinking> closures.

    Returns {"thinking": ..., "answer": ...}.
    """
    # Try to find <thinking> open
    open_m = _THINK_OPEN_RE.search(text_after_tag)
    if not open_m:
        # No <thinking> at all — treat entire text as answer
        return {"thinking": None, "answer": text_after_tag.strip()}

    after_open = text_after_tag[open_m.end():]

    # Try to find </thinking> close
    close_m = _THINK_CLOSE_RE.search(after_open)
    if close_m:
        # Clean case: properly closed
        thinking = after_open[:close_m.start()].strip()
        answer = after_open[close_m.end():].strip()
        return {"thinking": thinking, "answer": answer}

    # ── Degradation repair: </thinking> is MISSING ──
    # Strategy: look for an answer-boundary pattern to split
    boundary_m = _ANSWER_BOUNDARY_RE.search(after_open)
    if boundary_m:
        thinking = after_open[:boundary_m.start()].strip()
        answer = after_open[boundary_m.start():].strip()
        return {"thinking": thinking, "answer": f"[auto-closed] {answer}"}

    # Last resort: use last paragraph break as boundary
    last_para = after_open.rfind("\n\n")
    if last_para > 0 and last_para < len(after_open) - 10:
        thinking = after_open[:last_para].strip()
        answer = after_open[last_para:].strip()
        return {"thinking": thinking, "answer": f"[auto-closed] {answer}"}

    # Absolute fallback: everything is thinking, answer is empty
    return {"thinking": after_open.strip(), "answer": "[auto-closed] (answer embedded in thinking)"}


def generate_with_metis(prompt: str, max_new_tokens: int = 1024) -> Dict[str, Any]:
    """METIS Inference Gateway — structured cognitive output.

    Args:
        prompt: User query string.
        max_new_tokens: Generation budget.

    Returns:
        {
            "route": "DEEP" | "FAST" | "FAST (Implicit)",
            "thinking": str | None,
            "answer": str,
            "_raw": str,       # original model output for debugging
            "_latency_s": float,
        }
    """
    t0 = time.time()
    raw = _generate_raw(prompt, max_new_tokens=max_new_tokens)
    latency = time.time() - t0

    tag_m = _TAG_RE.search(raw)

    # ── Case A: Explicit DEEP ──
    if tag_m and tag_m.group(1).upper() == "DEEP":
        body = _strip_tag(raw, tag_m)
        parsed = _repair_deep_output(body)
        return {
            "route": "DEEP",
            "thinking": parsed["thinking"],
            "answer": parsed["answer"],
            "_raw": raw,
            "_latency_s": round(latency, 2),
        }

    # ── Case B: Explicit FAST ──
    if tag_m and tag_m.group(1).upper() == "FAST":
        body = _strip_tag(raw, tag_m)
        # FAST should have no thinking — strip any accidental thinking tags
        body = _THINK_OPEN_RE.sub("", body)
        body = _THINK_CLOSE_RE.sub("", body)
        return {
            "route": "FAST",
            "thinking": None,
            "answer": body.strip(),
            "_raw": raw,
            "_latency_s": round(latency, 2),
        }

    # ── Case C: No tag detected → Implicit FAST short-circuit ──
    return {
        "route": "FAST (Implicit)",
        "thinking": None,
        "answer": raw.strip(),
        "_raw": raw,
        "_latency_s": round(latency, 2),
    }


# ─────────────────────────────────────────────────────
# Gateway test
# ─────────────────────────────────────────────────────

TEST_QUERIES = [
    {
        "prompt": "What is 5 + 7?",
        "expected_case": "Case C → FAST (Implicit)",
    },
    {
        "prompt": "Solve: 3x + 2y = 16, x - y = 2",
        "expected_case": "Case A → DEEP (auto-repaired thinking closure)",
    },
    {
        "prompt": "What is the capital of France?",
        "expected_case": "Case B or C → FAST",
    },
]


def main() -> None:
    logger.info("=" * 70)
    logger.info("METIS Phase 8 — Inference Router Gateway Test")
    logger.info("=" * 70)
    logger.info(f"Model: {MODEL_PATH}")
    logger.info("")

    results = []
    for i, tq in enumerate(TEST_QUERIES, 1):
        logger.info(f"─── Query {i}/{len(TEST_QUERIES)} ───")
        logger.info(f"  Prompt:   {tq['prompt']}")
        logger.info(f"  Expected: {tq['expected_case']}")

        result = generate_with_metis(tq["prompt"])
        results.append(result)

        logger.info(f"  Route:    {result['route']}")
        logger.info(f"  Latency:  {result['_latency_s']}s")
        if result["thinking"]:
            preview = result["thinking"][:200] + ("..." if len(result["thinking"]) > 200 else "")
            logger.info(f"  Thinking: {preview}")
        logger.info(f"  Answer:   {result['answer'][:300]}")
        logger.info("")

    # ── Pretty-print JSON ──
    logger.info("=" * 70)
    logger.info("STRUCTURED JSON OUTPUT")
    logger.info("=" * 70)
    for i, (tq, result) in enumerate(zip(TEST_QUERIES, results), 1):
        # Create clean output (without _raw for readability)
        clean = {k: v for k, v in result.items() if k != "_raw"}
        print(f"\n{'─'*60}")
        print(f"Query {i}: \"{tq['prompt']}\"")
        print(f"{'─'*60}")
        print(json.dumps(clean, indent=2, ensure_ascii=False))

    # ── Summary ──
    print(f"\n{'='*70}")
    print("GATEWAY SUMMARY")
    print(f"{'='*70}")
    routes = [r["route"] for r in results]
    deep_count = sum(1 for r in routes if r == "DEEP")
    fast_count = sum(1 for r in routes if r == "FAST")
    implicit_count = sum(1 for r in routes if "Implicit" in r)
    repaired = sum(1 for r in results if r.get("answer", "").startswith("[auto-closed]"))
    print(f"  DEEP routes:           {deep_count}")
    print(f"  FAST routes (explicit):{fast_count}")
    print(f"  FAST routes (implicit):{implicit_count}")
    print(f"  Thinking auto-repairs: {repaired}")
    print(f"  Total queries:         {len(results)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
