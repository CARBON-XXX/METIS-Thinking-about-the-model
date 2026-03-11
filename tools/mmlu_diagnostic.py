#!/usr/bin/env python3
"""
MMLU Parser Diagnostic — 取证脚本

对比裸编码 vs chat template 生成效果，暴露 regex 假阳性问题。
用法:
    python tools/mmlu_diagnostic.py --model Qwen/Qwen2.5-7B-Instruct --n 10
"""
from __future__ import annotations

import argparse
import re
import sys
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── 当前 regex（复制自 benchmarks.py，用于对比） ──

_ANSWER_LETTER_RE = re.compile(
    r'(?:'
    r'FINAL\s*ANSWER\s*[:：]\s*\(?([A-Da-d])\)?'
    r'|(?:answer|Answer|ANSWER)\s*(?:is|:|：)\s*\(?([A-Da-d])\)?'
    r'|\b([A-D])\s*[\.。)）]'
    r'|^\s*\(?([A-Da-d])\)?\s*$'
    r')',
    re.MULTILINE,
)


def _strip_thinking(text: str) -> str:
    return re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL).strip()


def _extract_answer_letter_OLD(text: str) -> Optional[str]:
    """Current (buggy) extractor — for comparison."""
    cleaned = _strip_thinking(text)
    if not cleaned:
        return None
    matches = _ANSWER_LETTER_RE.findall(cleaned)
    if matches:
        last_match = matches[-1]
        for g in last_match:
            if g:
                return g.upper()
    tail = cleaned[-20:].strip()
    m = re.search(r'([A-D])', tail)
    if m:
        return m.group(1)
    return None


# ── NEW fixed extractor (imported from benchmarks.py) ──
from metis.training.benchmarks import _extract_answer_letter as _extract_answer_letter_NEW

_MMLU_CHOICES = ["A", "B", "C", "D"]


def _format_mmlu_prompt(
    question: str,
    choices: List[str],
    subject: str,
    few_shot: List[Dict[str, Any]],
) -> str:
    subject_name = subject.replace("_", " ").title()
    parts = [f"The following are multiple choice questions about {subject_name}.\n"]
    for ex in few_shot:
        parts.append(f"{ex['question']}")
        for i, c in enumerate(ex["choices"]):
            parts.append(f"{_MMLU_CHOICES[i]}. {c}")
        parts.append(f"Answer: {_MMLU_CHOICES[ex['answer']]}\n")
    parts.append(f"{question}")
    for i, c in enumerate(choices):
        parts.append(f"{_MMLU_CHOICES[i]}. {c}")
    parts.append("Answer:")
    return "\n".join(parts)


def _format_mmlu_prompt_FIXED(
    question: str,
    choices: List[str],
    subject: str,
    few_shot: List[Dict[str, Any]],
) -> str:
    subject_name = subject.replace("_", " ").title()
    parts = [f"The following are multiple choice questions about {subject_name}.\n"]
    for ex in few_shot:
        parts.append(f"{ex['question']}")
        for i, c in enumerate(ex["choices"]):
            parts.append(f"{_MMLU_CHOICES[i]}. {c}")
        parts.append(f"Answer: {_MMLU_CHOICES[ex['answer']]}\n")
    parts.append(f"{question}")
    for i, c in enumerate(choices):
        parts.append(f"{_MMLU_CHOICES[i]}. {c}")
    parts.append("Answer with just the letter (A, B, C, or D):")
    return "\n".join(parts)


@torch.inference_mode()
def generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    device: str,
    use_chat_template: bool = False,
    max_new_tokens: int = 64,
) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    else:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    generated_ids = outputs[0, input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n", type=int, default=10, help="questions to test")
    parser.add_argument("--subject", default="high_school_statistics")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = str(next(model.parameters()).device)

    print(f"Loading MMLU subject: {args.subject}")
    test_ds = load_dataset("cais/mmlu", args.subject, split="test")
    dev_ds = load_dataset("cais/mmlu", args.subject, split="dev")

    few_shot = []
    for i in range(min(5, len(dev_ds))):
        few_shot.append({
            "question": dev_ds[i]["question"],
            "choices": dev_ds[i]["choices"],
            "answer": dev_ds[i]["answer"],
        })

    n = min(args.n, len(test_ds))
    raw_correct = 0
    chat_correct = 0
    fixed_correct = 0

    for idx in range(n):
        ex = test_ds[idx]
        question = ex["question"]
        choices = ex["choices"]
        correct_idx = ex["answer"]
        correct_letter = _MMLU_CHOICES[correct_idx]

        prompt = _format_mmlu_prompt(question, choices, args.subject, few_shot)

        # ── Mode A: Raw encoding (current buggy path) ──
        raw_gen = generate(model, tokenizer, prompt, device, use_chat_template=False)
        raw_pred = _extract_answer_letter_OLD(raw_gen)

        # ── Mode B: Chat template + OLD prompt ──
        chat_gen = generate(model, tokenizer, prompt, device, use_chat_template=True, max_new_tokens=256)
        chat_pred_old = _extract_answer_letter_OLD(chat_gen)
        chat_pred_new = _extract_answer_letter_NEW(chat_gen)

        # ── Mode C: Chat template + FIXED prompt ──
        fixed_prompt = _format_mmlu_prompt_FIXED(question, choices, args.subject, few_shot)
        fixed_gen = generate(model, tokenizer, fixed_prompt, device, use_chat_template=True, max_new_tokens=256)
        fixed_pred = _extract_answer_letter_NEW(fixed_gen)

        raw_ok = raw_pred == correct_letter
        chat_ok = chat_pred_new == correct_letter
        fixed_ok = fixed_pred == correct_letter
        raw_correct += int(raw_ok)
        chat_correct += int(chat_ok)
        fixed_correct += int(fixed_ok)

        print(f"\n{'='*80}")
        print(f"Q{idx+1}/{n} | Correct: {correct_letter}")
        print(f"  Raw(old regex):        {raw_pred} {'✓' if raw_ok else '✗'}")
        print(f"  Chat+OldRegex:         {chat_pred_old}")
        print(f"  Chat+NewRegex:         {chat_pred_new} {'✓' if chat_ok else '✗'}")
        print(f"  Chat+FixedPrompt+New:  {fixed_pred} {'✓' if fixed_ok else '✗'}")
        print(f"{'─'*80}")
        print(f"[RAW GEN] ({len(raw_gen)} chars): {repr(raw_gen[:200])}")
        print(f"[CHAT GEN] ({len(chat_gen)} chars): {repr(chat_gen[:200])}")
        print(f"[FIXED GEN] ({len(fixed_gen)} chars): {repr(fixed_gen[:200])}")

    print(f"\n{'='*80}")
    print(f"SUMMARY ({n} questions):")
    print(f"  Raw(old regex):        {raw_correct}/{n} ({raw_correct/n:.1%})")
    print(f"  Chat+NewRegex:         {chat_correct}/{n} ({chat_correct/n:.1%})")
    print(f"  Chat+FixedPrompt+New:  {fixed_correct}/{n} ({fixed_correct/n:.1%})  <-- FULL FIX")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
