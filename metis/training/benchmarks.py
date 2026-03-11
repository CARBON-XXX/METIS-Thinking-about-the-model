"""
METIS External Benchmark Evaluator

Independent evaluation on standard NLP benchmarks to validate that
METIS cognitive reward training produces genuine improvements,
not just metric gaming.

Supported benchmarks:
═══════════════════════════════════════════════════════════════════
1. TruthfulQA (MC1 + MC2)
   - Tests hallucination resistance
   - METIS hypothesis: R_epistemic penalizes overconfidence
     → should reduce hallucination rate

2. MMLU (5-shot, 57 subjects)
   - Tests knowledge retention
   - METIS hypothesis: DPO should NOT cause catastrophic forgetting
     → MMLU should remain stable or improve

3. IFEval (instruction following)
   - Tests structural compliance
   - METIS hypothesis: R_phase rewards reasoning arcs
     → should improve instruction adherence
═══════════════════════════════════════════════════════════════════

MMLU and TruthfulQA MC1 use **generative parsing**: the model generates
a response, then regex extraction pulls the final answer letter.  This
correctly evaluates models that emit <thinking> blocks before answering.
TruthfulQA MC2 still uses log-likelihood scoring (multi-correct target).

Usage:
    from metis.training.benchmarks import BenchmarkSuite
    suite = BenchmarkSuite(model, tokenizer)
    results = suite.run_all()
    print(results)
"""
from __future__ import annotations

import gc
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# Generative Answer Extraction Helpers
# ─────────────────────────────────────────────────────

# ── Tier 1: Explicit answer markers (highest confidence) ──
_TIER1_RE = re.compile(
    r'(?:'
    r'FINAL\s*ANSWER\s*[:：]\s*\(?([A-Da-d])\)?'
    r'|(?:最终答案|答案)\s*[:：]?\s*\(?([A-Da-d])\)?'
    r'|(?:the\s+)?(?:correct\s+)?answer\s+is\s*[:：]?\s*\(?([A-Da-d])\)?'
    r'|(?:I\s+(?:choose|pick|select|would\s+(?:choose|pick|select)))\s+\(?([A-Da-d])\)?'
    r')',
    re.IGNORECASE,
)

# ── Tier 2: Structural patterns (medium confidence) ──
_TIER2_RE = re.compile(
    r'(?:'
    r'(?:选|选择)\s*\(?([A-Da-d])\)?'
    r'|\b([A-Da-d])\s+is\s+(?:the\s+)?(?:correct|right|best)'
    r'|(?:option|choice)\s+\(?([A-Da-d])\)?\s+is\s+(?:correct|right|best)'
    r')',
    re.IGNORECASE,
)

# ── Tier 3: Standalone letter on its own line ──
_TIER3_RE = re.compile(r'^\s*\(?([A-Da-d])\)?[.。]?\s*$', re.MULTILINE)


def _strip_thinking(text: str) -> str:
    """Remove <thinking>...</thinking> blocks from generated text."""
    cleaned = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    return cleaned.strip()


def _extract_answer_letter(text: str) -> Optional[str]:
    """Extract A/B/C/D answer letter from generated text.

    4-tier priority extraction strategy:
      Tier 0: First non-whitespace character (completion-style "Answer: D")
      Tier 1: Explicit markers ("FINAL ANSWER: A", "the answer is B")
      Tier 2: Structural patterns ("option C is correct", "选 D")
      Tier 3: Standalone letter on its own line
    No dangerous fallback — returns None if all tiers fail.
    """
    cleaned = _strip_thinking(text)
    if not cleaned:
        return None

    # ── Tier 0: First-token extraction (completion-style models) ──
    # After "Answer:" prompt, model often outputs " D\n\n..." as continuation
    first_chars = cleaned.lstrip()[:3].strip()
    if len(first_chars) >= 1 and first_chars[0].upper() in 'ABCD':
        # Verify it's a standalone letter, not start of a word like "An" or "But"
        if len(first_chars) == 1 or not first_chars[1].isalpha():
            return first_chars[0].upper()

    # ── Tier 1: Explicit answer markers (last match wins for CoT) ──
    matches = _TIER1_RE.findall(cleaned)
    if matches:
        last_match = matches[-1]
        for g in (last_match if isinstance(last_match, tuple) else [last_match]):
            if g:
                return g.upper()

    # ── Tier 2: Structural patterns ──
    matches = _TIER2_RE.findall(cleaned)
    if matches:
        last_match = matches[-1]
        for g in (last_match if isinstance(last_match, tuple) else [last_match]):
            if g:
                return g.upper()

    # ── Tier 3: Standalone letter on its own line ──
    matches = _TIER3_RE.findall(cleaned)
    if matches:
        return matches[-1].upper()

    # ── No match: return None (parse failure, do NOT guess) ──
    logger.debug(
        f"[Parser] Failed to extract answer from: {cleaned[:200]!r}"
    )
    return None


@torch.inference_mode()
def _generate_answer(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    device: str,
    max_new_tokens: int = 256,
) -> str:
    """Generate a text answer from the model.

    Uses greedy decoding (temperature=0) for deterministic evaluation.
    Applies chat template if available (critical for Instruct models).
    """
    # Apply chat template for Instruct/Chat models
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": prompt}]
        text_input = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        input_ids = tokenizer.encode(text_input, return_tensors="pt").to(device)
    else:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Greedy decoding for reproducible evaluation
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,  # ignored when do_sample=False
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    # Decode only the new tokens
    generated_ids = outputs[0, input_ids.shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text


# ─────────────────────────────────────────────────────
# Result Types
# ─────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    """Result from a single benchmark."""
    name: str
    accuracy: float = 0.0
    n_correct: int = 0
    n_total: int = 0
    sub_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "accuracy": round(self.accuracy, 4),
            "n_correct": self.n_correct,
            "n_total": self.n_total,
        }
        if self.sub_scores:
            d["sub_scores"] = {k: round(v, 4) for k, v in self.sub_scores.items()}
        return d


@dataclass
class BenchmarkSuiteResult:
    """Combined results from all benchmarks."""
    results: Dict[str, BenchmarkResult] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v.to_dict() for k, v in self.results.items()}

    def __str__(self) -> str:
        lines = ["External Benchmark Results:"]
        for name, r in self.results.items():
            lines.append(f"  {name}: {r.accuracy:.1%} ({r.n_correct}/{r.n_total})")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────
# Log-Likelihood Scoring Engine
# ─────────────────────────────────────────────────────

@torch.inference_mode()
def _score_completions(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    completions: List[str],
    device: Optional[str] = None,
) -> List[float]:
    """Score multiple completions for a prompt using mean log-likelihood.

    For each completion, computes:
        score = (1/T) * sum(log P(token_t | prompt, token_1..t-1))

    where T = number of completion tokens. Mean normalization prevents
    length bias (longer completions aren't automatically lower-scored).

    Args:
        model: HuggingFace causal LM
        tokenizer: Matching tokenizer
        prompt: Context/question text
        completions: List of possible answer strings
        device: Override device

    Returns:
        List of mean log-likelihood scores (higher = more likely)
    """
    if device is None:
        device = str(next(model.parameters()).device)

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_len = len(prompt_ids)

    scores: List[float] = []

    for completion in completions:
        comp_ids = tokenizer.encode(completion, add_special_tokens=False)
        if not comp_ids:
            scores.append(float("-inf"))
            continue

        full_ids = prompt_ids + comp_ids
        input_tensor = torch.tensor([full_ids], device=device)

        outputs = model(input_ids=input_tensor, use_cache=False)
        logits = outputs.logits  # [1, seq_len, vocab_size]

        # Score only completion tokens (not prompt)
        log_probs = F.log_softmax(logits[0].float(), dim=-1)

        total_log_prob = 0.0
        n_tokens = len(comp_ids)
        for i, token_id in enumerate(comp_ids):
            # Position in logits: prompt_len - 1 + i predicts token at prompt_len + i
            pos = prompt_len - 1 + i
            if pos < log_probs.shape[0]:
                total_log_prob += log_probs[pos, token_id].item()

        mean_log_prob = total_log_prob / n_tokens
        scores.append(mean_log_prob)

        del outputs, logits, log_probs

    return scores


# ─────────────────────────────────────────────────────
# TruthfulQA Benchmark
# ─────────────────────────────────────────────────────

def evaluate_truthfulqa(
    model: torch.nn.Module,
    tokenizer: Any,
    max_questions: int = 200,
    device: Optional[str] = None,
) -> BenchmarkResult:
    """Evaluate on TruthfulQA using **generative parsing**.

    MC1 (generative): Generate an answer, then fuzzy-match against the
    correct choice text.  This avoids the log-likelihood blind spot
    where models that emit <thinking> before the answer token get
    mis-scored.

    MC2: Still uses log-likelihood scoring (multi-correct, no single
    generation target).

    Dataset: truthfulqa/truthful_qa (HuggingFace)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not installed. Run: pip install datasets")
        return BenchmarkResult(name="TruthfulQA-MC1")

    logger.info("[Benchmark] Loading TruthfulQA...")
    try:
        ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    except Exception as e:
        logger.error(f"[Benchmark] Failed to load TruthfulQA: {e}")
        return BenchmarkResult(name="TruthfulQA-MC1")

    if device is None:
        device = str(next(model.parameters()).device)

    # Subsample if needed
    indices = list(range(len(ds)))
    if len(indices) > max_questions:
        rng = random.Random(42)
        indices = rng.sample(indices, max_questions)

    n_correct_mc1 = 0
    n_correct_mc2 = 0
    n_total = 0
    by_category: Dict[str, List[bool]] = {}

    for idx in indices:
        example = ds[idx]
        question = example["question"]
        mc1_choices = example["mc1_targets"]["choices"]
        mc1_labels = example["mc1_targets"]["labels"]
        mc2_choices = example["mc2_targets"]["choices"]
        mc2_labels = example["mc2_targets"]["labels"]
        category = example.get("category", "unknown")

        # ── MC1: Generative evaluation ──
        # Build a numbered-choice prompt so the model can reference by index
        choice_lines = [f"  ({chr(65+i)}) {c}" for i, c in enumerate(mc1_choices)]
        prompt = (
            f"Q: {question}\n"
            f"Choices:\n" + "\n".join(choice_lines) + "\n"
            f"Answer with just the letter of the correct choice."
        )

        generated = _generate_answer(model, tokenizer, prompt, device, max_new_tokens=256)
        pred_letter = _extract_answer_letter(generated)

        correct = False
        if pred_letter is not None:
            pred_idx = ord(pred_letter) - ord('A')
            if 0 <= pred_idx < len(mc1_labels):
                correct = mc1_labels[pred_idx] == 1
        else:
            # Fallback: fuzzy-match generated text against choice strings
            cleaned = _strip_thinking(generated).lower().strip()
            correct_texts = [mc1_choices[i].lower() for i, l in enumerate(mc1_labels) if l == 1]
            correct = any(ct in cleaned or cleaned in ct for ct in correct_texts if len(ct) > 3)

        n_correct_mc1 += int(correct)

        if category not in by_category:
            by_category[category] = []
        by_category[category].append(correct)

        # ── MC2: Log-likelihood scoring (multi-correct, no single answer) ──
        mc2_prompt = f"Q: {question}\nA:"
        mc2_scores = _score_completions(model, tokenizer, mc2_prompt, mc2_choices, device)
        if mc2_scores:
            correct_scores = [
                mc2_scores[i] for i, l in enumerate(mc2_labels) if l == 1
            ]
            incorrect_scores = [
                mc2_scores[i] for i, l in enumerate(mc2_labels) if l == 0
            ]
            if correct_scores and incorrect_scores:
                avg_correct = sum(correct_scores) / len(correct_scores)
                avg_incorrect = sum(incorrect_scores) / len(incorrect_scores)
                n_correct_mc2 += int(avg_correct > avg_incorrect)

        n_total += 1

        if (n_total % 50) == 0:
            logger.info(
                f"  [TruthfulQA] {n_total}/{len(indices)}: "
                f"MC1={n_correct_mc1}/{n_total} ({n_correct_mc1/n_total:.1%})"
            )

        # VRAM hygiene
        if n_total % 20 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    accuracy_mc1 = n_correct_mc1 / max(n_total, 1)
    accuracy_mc2 = n_correct_mc2 / max(n_total, 1)

    sub_scores: Dict[str, float] = {"mc2_accuracy": accuracy_mc2}
    for cat, results in by_category.items():
        sub_scores[f"cat_{cat}"] = sum(results) / len(results) if results else 0.0

    logger.info(
        f"[Benchmark] TruthfulQA (generative): MC1={accuracy_mc1:.1%} MC2={accuracy_mc2:.1%} "
        f"({n_total} questions)"
    )

    return BenchmarkResult(
        name="TruthfulQA-MC1",
        accuracy=accuracy_mc1,
        n_correct=n_correct_mc1,
        n_total=n_total,
        sub_scores=sub_scores,
    )


# ─────────────────────────────────────────────────────
# MMLU Benchmark
# ─────────────────────────────────────────────────────

# Representative subset of MMLU subjects (covers STEM + humanities + social)
_MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "college_biology",
    "college_chemistry", "college_computer_science", "college_mathematics",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_mathematics", "high_school_physics",
    "high_school_statistics", "logical_fallacies",
    "machine_learning", "moral_scenarios", "philosophy",
    "professional_medicine", "virology", "world_religions",
]

_MMLU_CHOICES = ["A", "B", "C", "D"]


def _format_mmlu_prompt(
    question: str,
    choices: List[str],
    subject: str,
    few_shot_examples: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Format MMLU question with optional few-shot examples."""
    subject_name = subject.replace("_", " ").title()
    parts = [f"The following are multiple choice questions about {subject_name}.\n"]

    if few_shot_examples:
        for ex in few_shot_examples:
            parts.append(f"{ex['question']}")
            for i, c in enumerate(ex["choices"]):
                parts.append(f"{_MMLU_CHOICES[i]}. {c}")
            parts.append(f"Answer: {_MMLU_CHOICES[ex['answer']]}\n")

    parts.append(f"{question}")
    for i, c in enumerate(choices):
        parts.append(f"{_MMLU_CHOICES[i]}. {c}")
    parts.append("Answer with just the letter (A, B, C, or D):")

    return "\n".join(parts)


def evaluate_mmlu(
    model: torch.nn.Module,
    tokenizer: Any,
    n_subjects: int = 10,
    max_per_subject: int = 30,
    n_shot: int = 5,
    device: Optional[str] = None,
) -> BenchmarkResult:
    """Evaluate on MMLU using **generative parsing**.

    Generates a response for each question, then extracts the A/B/C/D
    letter from the generated text using regex.  This handles models
    that emit <thinking> blocks before the answer token.

    Supports few-shot prompting (default 5-shot).

    Dataset: cais/mmlu (HuggingFace)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not installed. Run: pip install datasets")
        return BenchmarkResult(name="MMLU")

    if device is None:
        device = str(next(model.parameters()).device)

    # Select subjects
    rng = random.Random(42)
    subjects = _MMLU_SUBJECTS[:n_subjects] if n_subjects <= len(_MMLU_SUBJECTS) else _MMLU_SUBJECTS
    if n_subjects < len(_MMLU_SUBJECTS):
        subjects = rng.sample(_MMLU_SUBJECTS, n_subjects)

    n_correct = 0
    n_total = 0
    n_parse_fail = 0
    by_subject: Dict[str, Tuple[int, int]] = {}

    for subject in subjects:
        logger.info(f"  [MMLU] Loading subject: {subject}")
        try:
            test_ds = load_dataset("cais/mmlu", subject, split="test")
            dev_ds = load_dataset("cais/mmlu", subject, split="dev")
        except Exception as e:
            logger.warning(f"  [MMLU] Failed to load {subject}: {e}")
            continue

        # Few-shot examples from dev set
        few_shot: List[Dict[str, Any]] = []
        if n_shot > 0 and len(dev_ds) > 0:
            for i in range(min(n_shot, len(dev_ds))):
                few_shot.append({
                    "question": dev_ds[i]["question"],
                    "choices": dev_ds[i]["choices"],
                    "answer": dev_ds[i]["answer"],
                })

        # Evaluate test set
        test_indices = list(range(len(test_ds)))
        if len(test_indices) > max_per_subject:
            test_indices = rng.sample(test_indices, max_per_subject)

        subj_correct = 0
        subj_total = 0

        for idx in test_indices:
            example = test_ds[idx]
            question = example["question"]
            choices = example["choices"]
            answer = example["answer"]  # int: 0-3

            prompt = _format_mmlu_prompt(question, choices, subject, few_shot)

            # Generative evaluation: generate answer, extract letter
            generated = _generate_answer(model, tokenizer, prompt, device, max_new_tokens=256)
            pred_letter = _extract_answer_letter(generated)

            if pred_letter is not None:
                predicted = ord(pred_letter) - ord('A')
            else:
                # Parse failure: count as wrong, log for diagnostics
                predicted = -1
                n_parse_fail += 1

            if predicted == answer:
                subj_correct += 1
                n_correct += 1

            subj_total += 1
            n_total += 1

        by_subject[subject] = (subj_correct, subj_total)
        subj_acc = subj_correct / max(subj_total, 1)
        logger.info(
            f"  [MMLU] {subject}: {subj_acc:.1%} ({subj_correct}/{subj_total})"
        )

        # VRAM hygiene
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    accuracy = n_correct / max(n_total, 1)
    parse_fail_rate = n_parse_fail / max(n_total, 1)
    sub_scores = {
        subj: correct / max(total, 1)
        for subj, (correct, total) in by_subject.items()
    }
    sub_scores["parse_fail_rate"] = parse_fail_rate

    logger.info(
        f"[Benchmark] MMLU (generative): {accuracy:.1%} ({n_correct}/{n_total}) "
        f"across {len(by_subject)} subjects | parse_fail={n_parse_fail}/{n_total}"
    )

    return BenchmarkResult(
        name="MMLU",
        accuracy=accuracy,
        n_correct=n_correct,
        n_total=n_total,
        sub_scores=sub_scores,
    )


# ─────────────────────────────────────────────────────
# Benchmark Suite
# ─────────────────────────────────────────────────────

class BenchmarkSuite:
    """Run all external benchmarks on a model.

    Usage:
        suite = BenchmarkSuite(model, tokenizer)
        results = suite.run_all()
        print(results)
        # Or run individually:
        tqa = suite.run_truthfulqa()
        mmlu = suite.run_mmlu()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: Optional[str] = None,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._device = device or str(next(model.parameters()).device)

    def run_truthfulqa(
        self, max_questions: int = 200
    ) -> BenchmarkResult:
        return evaluate_truthfulqa(
            self._model, self._tokenizer,
            max_questions=max_questions, device=self._device,
        )

    def run_mmlu(
        self,
        n_subjects: int = 10,
        max_per_subject: int = 30,
        n_shot: int = 5,
    ) -> BenchmarkResult:
        return evaluate_mmlu(
            self._model, self._tokenizer,
            n_subjects=n_subjects, max_per_subject=max_per_subject,
            n_shot=n_shot, device=self._device,
        )

    def run_all(
        self,
        truthfulqa_questions: int = 200,
        mmlu_subjects: int = 10,
        mmlu_per_subject: int = 30,
    ) -> BenchmarkSuiteResult:
        """Run all benchmarks and return combined results."""
        logger.info("[Benchmark] Starting external benchmark suite...")

        suite_result = BenchmarkSuiteResult()

        # TruthfulQA
        tqa = self.run_truthfulqa(max_questions=truthfulqa_questions)
        suite_result.results["TruthfulQA-MC1"] = tqa
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # MMLU
        mmlu = self.run_mmlu(
            n_subjects=mmlu_subjects, max_per_subject=mmlu_per_subject,
        )
        suite_result.results["MMLU"] = mmlu
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"[Benchmark] Suite complete: {suite_result}")
        return suite_result
