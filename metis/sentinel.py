"""
METIS Degradation Sentinel — Post-Training Regression Gate

Prevents evolutionary mode collapse by running a lightweight canary
benchmark after every DreamingDaemon training cycle.  If accuracy drops
below configurable thresholds the new weights are automatically rolled
back and the training cycle is marked as failed.

Architecture:
  train() completes  →  Sentinel loads merged model
                     →  runs 20 canary questions (10 math + 10 QA)
                     →  compares against stored baseline
                     →  PASS  → promote model, mark gaps resolved
                     →  FAIL  → archive output, keep old base, alert

Thresholds (all configurable):
  max_accuracy_drop_pct   : max relative drop vs baseline  (default 5%)
  min_absolute_accuracy   : hard floor                     (default 70%)
  max_consecutive_drops   : consecutive regressions before emergency halt (default 3)
  max_kl_divergence       : latent-space warp ceiling in nats   (default 0.15)
                            Phase 23.5 — rejects model even at 100% accuracy
                            if output distribution has drifted too far
"""
from __future__ import annotations

import gc
import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger("metis.sentinel")

# ═══════════════════════════════════════════════════════════
# Canary Benchmark — deterministic, self-contained, offline
# ═══════════════════════════════════════════════════════════

# 10 math canary questions (GSM8K-style, with known numeric answers)
_MATH_CANARIES: List[Dict[str, str]] = [
    {
        "q": "A store sells apples for $2 each. If Sarah buys 5 apples and pays with a $20 bill, how much change does she receive?",
        "a": "10",
    },
    {
        "q": "A train travels at 60 km/h for 2.5 hours. How many kilometers does it travel?",
        "a": "150",
    },
    {
        "q": "If a rectangle has a length of 8 cm and a width of 5 cm, what is its area in square centimeters?",
        "a": "40",
    },
    {
        "q": "A baker makes 12 cookies per batch. If he needs 84 cookies for a party, how many batches must he bake?",
        "a": "7",
    },
    {
        "q": "Maria has 3 times as many books as Tom. If Tom has 15 books, how many books do they have together?",
        "a": "60",
    },
    {
        "q": "A shirt originally costs $40. It is on sale for 25% off. What is the sale price in dollars?",
        "a": "30",
    },
    {
        "q": "If 5 workers can build a wall in 10 days, how many days would it take 10 workers to build the same wall?",
        "a": "5",
    },
    {
        "q": "A car uses 8 liters of fuel per 100 km. How many liters are needed for a 350 km trip?",
        "a": "28",
    },
    {
        "q": "The sum of three consecutive integers is 72. What is the largest of the three?",
        "a": "25",
    },
    {
        "q": "A class has 30 students. If 60% are girls, how many boys are in the class?",
        "a": "12",
    },
]

# 10 factual QA canary questions (substring-match gold answers)
_QA_CANARIES: List[Dict[str, str]] = [
    {"q": "What is the capital of France?", "a": "Paris"},
    {"q": "What is the chemical symbol for gold?", "a": "Au"},
    {"q": "Who wrote Romeo and Juliet?", "a": "Shakespeare"},
    {"q": "What is the largest planet in our solar system?", "a": "Jupiter"},
    {"q": "What year did World War II end?", "a": "1945"},
    {"q": "What is the boiling point of water in Celsius?", "a": "100"},
    {"q": "What element has the atomic number 1?", "a": "Hydrogen"},
    {"q": "Who developed the theory of general relativity?", "a": "Einstein"},
    {"q": "How many continents are there?", "a": "7"},
    {"q": "What is the hardest natural substance?", "a": "Diamond"},
]


# ═══════════════════════════════════════════════════════════
# Answer Accuracy Checkers (self-contained, no import needed)
# ═══════════════════════════════════════════════════════════

_NUM_RE = re.compile(r"[-+]?\d*\.\d+|\d+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def _check_math(answer: str, gold: str) -> bool:
    """Robust numeric accuracy: last-number extraction."""
    try:
        gold_val = float(gold.strip().replace(",", ""))
    except ValueError:
        return gold.strip().lower() in answer.lower()
    clean = re.sub(r"(\d),(\d)", r"\1\2", answer)
    nums = _NUM_RE.findall(clean)
    if not nums:
        return False
    for num_str in reversed(nums):
        try:
            if abs(float(num_str) - gold_val) < 0.01:
                return True
        except ValueError:
            continue
    return False


def _check_qa(answer: str, gold: str) -> bool:
    """Robust substring check for factual QA."""
    ans_lower = answer.lower()
    gold_lower = gold.lower()
    if gold_lower in ans_lower:
        return True
    ans_clean = _PUNCT_RE.sub("", ans_lower)
    gold_clean = _PUNCT_RE.sub("", gold_lower)
    return gold_clean in ans_clean


# ═══════════════════════════════════════════════════════════
# Verdict
# ═══════════════════════════════════════════════════════════

@dataclass
class SentinelVerdict:
    """Result of a single sentinel evaluation."""
    passed: bool
    overall_accuracy: float       # 0.0 – 1.0
    complex_accuracy: float       # math subset
    simple_accuracy: float        # QA subset
    baseline_accuracy: float      # stored baseline overall
    accuracy_delta: float         # overall - baseline (negative = regression)
    rollback_triggered: bool
    reason: str                   # human-readable explanation
    timestamp: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


# ═══════════════════════════════════════════════════════════
# DegradationSentinel
# ═══════════════════════════════════════════════════════════

class DegradationSentinel:
    """Post-training regression gate preventing mode collapse.

    Usage:
        sentinel = DegradationSentinel(
            baseline_path="data/sentinel_baseline.json",
            history_path="data/sentinel_history.json",
        )

        # First run: establish baseline from current production model
        sentinel.establish_baseline("experiment_output_dpo_balanced/metis_dpo_cognitive")

        # After each training cycle:
        verdict = sentinel.evaluate("experiment_output_dreams/merged")
        if not verdict.passed:
            sentinel.rollback("experiment_output_dreams")
    """

    def __init__(
        self,
        baseline_path: str | Path = "data/sentinel_baseline.json",
        history_path: str | Path = "data/sentinel_history.json",
        max_accuracy_drop_pct: float = 5.0,
        min_absolute_accuracy: float = 70.0,
        max_consecutive_drops: int = 3,
        max_kl_divergence: float = 0.15,
        anchor_model_path: Optional[str] = None,
    ):
        """
        Args:
            baseline_path: JSON file storing the golden baseline scores.
            history_path: JSON file tracking all evaluation history.
            max_accuracy_drop_pct: Maximum allowed accuracy drop in percentage
                points relative to baseline before triggering rollback.
            min_absolute_accuracy: Absolute accuracy floor (%) — rollback if
                new model falls below this regardless of baseline delta.
            max_consecutive_drops: Number of consecutive regressions before
                the daemon should halt entirely (emergency brake).
            max_kl_divergence: Maximum allowed KL divergence (nats) between
                the new model and the anchor model.  Phase 23.5 latent-space
                warp ceiling — triggers rollback even if accuracy is 100%.
            anchor_model_path: Path or HF ID of the pristine anchor model
                for KL-divergence gating.  If None, KL check is skipped.
        """
        self._baseline_path = Path(baseline_path)
        self._history_path = Path(history_path)
        self._max_drop = max_accuracy_drop_pct
        self._min_abs = min_absolute_accuracy
        self._max_consec = max_consecutive_drops
        self._max_kl = max_kl_divergence
        self._anchor_model_path = anchor_model_path

        # Ensure data directory exists
        self._baseline_path.parent.mkdir(parents=True, exist_ok=True)
        self._history_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Baseline Management ──

    def establish_baseline(self, model_path: str) -> Dict[str, Any]:
        """Run canary benchmark on current model and save as golden baseline.

        This should be called ONCE before the daemon starts, using the
        production model. The baseline is the quality floor that all
        future training cycles must not degrade below.

        Returns:
            Dict with accuracy scores that were saved.
        """
        logger.info("Establishing sentinel baseline...")
        logger.info(f"  Model: {model_path}")

        scores = self._run_canary_benchmark(model_path)

        baseline = {
            "model_path": model_path,
            "established_at": datetime.now().isoformat(),
            "overall_accuracy": scores["overall"],
            "complex_accuracy": scores["complex"],
            "simple_accuracy": scores["simple"],
            "per_question": scores["per_question"],
        }

        with open(self._baseline_path, "w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=2, ensure_ascii=False)

        logger.info(
            f"  Baseline established: overall={scores['overall']*100:.1f}%, "
            f"complex={scores['complex']*100:.1f}%, "
            f"simple={scores['simple']*100:.1f}%"
        )
        return baseline

    def load_baseline(self) -> Optional[Dict[str, Any]]:
        """Load stored baseline. Returns None if not established."""
        if not self._baseline_path.exists():
            return None
        try:
            with open(self._baseline_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load baseline: {e}")
            return None

    # ── Evaluation ──

    def evaluate(self, model_path: str) -> SentinelVerdict:
        """Run canary benchmark on new model and compare to baseline.

        Args:
            model_path: Path to the newly trained (merged) model.

        Returns:
            SentinelVerdict indicating PASS/FAIL and rollback status.
        """
        logger.info("=" * 60)
        logger.info("  DEGRADATION SENTINEL — Post-Training Regression Gate")
        logger.info(f"  Evaluating: {model_path}")
        logger.info("=" * 60)

        # Load baseline
        baseline = self.load_baseline()
        if baseline is None:
            logger.warning("No baseline found — auto-establishing from new model.")
            self.establish_baseline(model_path)
            return SentinelVerdict(
                passed=True,
                overall_accuracy=1.0,
                complex_accuracy=1.0,
                simple_accuracy=1.0,
                baseline_accuracy=1.0,
                accuracy_delta=0.0,
                rollback_triggered=False,
                reason="First evaluation — baseline auto-established.",
                details={"auto_baseline": True},
            )

        baseline_overall = baseline["overall_accuracy"]

        # Run canary benchmark
        scores = self._run_canary_benchmark(model_path)
        new_overall = scores["overall"]
        new_complex = scores["complex"]
        new_simple = scores["simple"]

        accuracy_delta = (new_overall - baseline_overall) * 100  # in pct points
        new_pct = new_overall * 100
        baseline_pct = baseline_overall * 100

        logger.info(f"  New model:  overall={new_pct:.1f}%, "
                     f"complex={new_complex*100:.1f}%, simple={new_simple*100:.1f}%")
        logger.info(f"  Baseline:   overall={baseline_pct:.1f}%")
        logger.info(f"  Delta:      {accuracy_delta:+.1f} pct points")

        # ── Decision Logic ──
        failed = False
        reasons: List[str] = []

        # Check 1: Relative drop
        if accuracy_delta < -self._max_drop:
            failed = True
            reasons.append(
                f"Accuracy dropped {accuracy_delta:.1f}pp "
                f"(threshold: -{self._max_drop:.1f}pp)"
            )

        # Check 2: Absolute floor
        if new_pct < self._min_abs:
            failed = True
            reasons.append(
                f"Accuracy {new_pct:.1f}% below absolute floor "
                f"{self._min_abs:.1f}%"
            )

        # Check 3: Category-specific collapse
        # If math OR QA drops below 50%, something is very wrong
        if new_complex < 0.5:
            failed = True
            reasons.append(
                f"Math accuracy collapsed to {new_complex*100:.1f}% (<50%)"
            )
        if new_simple < 0.5:
            failed = True
            reasons.append(
                f"QA accuracy collapsed to {new_simple*100:.1f}% (<50%)"
            )

        # Check 4: Consecutive drops (trend detection)
        history = self._load_history()
        consecutive_drops = self._count_consecutive_drops(history)
        if not failed and consecutive_drops >= self._max_consec - 1 and accuracy_delta < 0:
            # This would be the Nth consecutive drop
            failed = True
            reasons.append(
                f"Consecutive regression #{consecutive_drops + 1} "
                f"(limit: {self._max_consec})"
            )

        # Check 5: KL-divergence latent-space warp (Phase 23.5)
        # Runs INDEPENDENTLY of accuracy — even 100% accuracy is rejected
        # if the output distribution has warped beyond the threshold.
        kl_value: Optional[float] = None
        if self._anchor_model_path is not None:
            kl_value = self._compute_kl_divergence(model_path)
            if kl_value is not None:
                logger.info(f"  KL divergence: {kl_value:.4f} nats (threshold: {self._max_kl:.4f})")
                if kl_value > self._max_kl:
                    failed = True
                    reasons.append(
                        f"Latent Space Warping detected "
                        f"(KL={kl_value:.4f} > threshold={self._max_kl:.4f})"
                    )

        reason = "; ".join(reasons) if reasons else "All checks passed"
        passed = not failed

        verdict = SentinelVerdict(
            passed=passed,
            overall_accuracy=new_overall,
            complex_accuracy=new_complex,
            simple_accuracy=new_simple,
            baseline_accuracy=baseline_overall,
            accuracy_delta=accuracy_delta / 100,  # store as fraction
            rollback_triggered=failed,
            reason=reason,
            details={
                "model_path": model_path,
                "per_question": scores["per_question"],
                "baseline_path": str(self._baseline_path),
                "consecutive_drops": consecutive_drops + (1 if accuracy_delta < 0 else 0),
                "kl_divergence": kl_value,
                "thresholds": {
                    "max_drop_pct": self._max_drop,
                    "min_absolute": self._min_abs,
                    "max_consecutive": self._max_consec,
                    "max_kl_divergence": self._max_kl,
                },
            },
        )

        # Log verdict
        if passed:
            logger.info(f"  ✓ SENTINEL PASS: {reason}")
        else:
            logger.warning(f"  ✗ SENTINEL FAIL: {reason}")
            logger.warning("  → ROLLBACK TRIGGERED — new weights will be discarded.")

        # Record history
        self._append_history(verdict)

        return verdict

    # ── Phase 23.5: KL-Divergence Latent-Space Gate ──

    # Subset of canary prompts used for KL measurement (kept small for OOM safety)
    _KL_PROBE_PROMPTS: List[str] = [
        "A store sells apples for $2 each. If Sarah buys 5 apples and pays with a $20 bill, how much change does she receive?",
        "What is the chemical symbol for gold?",
        "A train travels at 60 km/h for 2.5 hours. How many kilometers does it travel?",
        "Who wrote the play Romeo and Juliet?",
    ]

    def _compute_kl_divergence(self, new_model_path: str) -> Optional[float]:
        """Compute batch-mean KL divergence between anchor and new model.

        Phase 23.5 — Latent-Space Warping Detection:
            D_KL(anchor || new) = F.kl_div(anchor_log_probs, new_log_probs,
                                           log_target=True, reduction='batchmean')

        Uses a small subset of canary prompts (4) with short max_new_tokens
        to avoid OOM.  All computation is under torch.inference_mode().

        Returns:
            KL divergence in nats, or None if computation failed.
        """
        if self._anchor_model_path is None:
            return None

        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("  Computing KL divergence (Phase 23.5)...")
        logger.info(f"    Anchor: {self._anchor_model_path}")
        logger.info(f"    New:    {new_model_path}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self._anchor_model_path, trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load anchor model (frozen, eval-only)
            anchor_model = AutoModelForCausalLM.from_pretrained(
                self._anchor_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            anchor_model.eval()
            for p in anchor_model.parameters():
                p.requires_grad = False

            # Load new model (frozen, eval-only)
            new_model = AutoModelForCausalLM.from_pretrained(
                new_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            new_model.eval()
            for p in new_model.parameters():
                p.requires_grad = False

            kl_values: List[float] = []

            with torch.inference_mode():
                for prompt in self._KL_PROBE_PROMPTS:
                    inputs = tokenizer(
                        prompt, return_tensors="pt", truncation=True, max_length=256
                    )
                    inputs = {k: v.to(anchor_model.device) for k, v in inputs.items()}

                    anchor_logits = anchor_model(**inputs).logits[:, -1, :]
                    new_logits = new_model(**inputs).logits[:, -1, :]

                    # Log-softmax for numerically stable KL computation
                    anchor_log_probs = F.log_softmax(anchor_logits.float(), dim=-1)
                    new_log_probs = F.log_softmax(new_logits.float(), dim=-1)

                    kl = F.kl_div(
                        anchor_log_probs, new_log_probs,
                        log_target=True, reduction="batchmean",
                    )
                    kl_values.append(kl.item())

            # Cleanup
            del anchor_model, new_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            mean_kl = sum(kl_values) / len(kl_values) if kl_values else 0.0
            logger.info(
                f"    KL per prompt: {[f'{v:.4f}' for v in kl_values]}  "
                f"mean={mean_kl:.4f}"
            )
            return mean_kl

        except Exception as e:
            logger.warning(f"    KL computation failed: {e}")
            # Cleanup on failure
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None

    # ── Rollback ──

    def rollback(self, output_dir: str) -> Path:
        """Archive failed training output to prevent accidental use.

        Moves ``output_dir/merged`` → ``output_dir/rolled_back_YYYYMMDD_HHMMSS/``
        so the merged model cannot be accidentally loaded.

        Returns:
            Path to the archive directory.
        """
        merged_path = Path(output_dir) / "merged"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"rolled_back_{timestamp}"
        archive_path = Path(output_dir) / archive_name

        if merged_path.exists():
            shutil.move(str(merged_path), str(archive_path))
            logger.info(f"  Rolled back: {merged_path} → {archive_path}")
        else:
            # No merged model to roll back — just create a marker
            archive_path.mkdir(parents=True, exist_ok=True)
            marker = archive_path / "ROLLBACK_MARKER.txt"
            marker.write_text(
                f"Rollback at {timestamp}\n"
                f"No merged model found at {merged_path}\n"
            )
            logger.info(f"  No merged model found; created rollback marker at {archive_path}")

        return archive_path

    # ── Promotion (Evolutionary Mode) ──

    def promote(self, output_dir: str, production_path: str) -> bool:
        """Promote a sentinel-approved merged model to production.

        Copies ``output_dir/merged`` → ``production_path`` so the daemon
        uses the improved model as the base for the next training cycle.

        This enables **compounding evolution** but only after sentinel approval.

        Returns:
            True if promotion succeeded.
        """
        merged_path = Path(output_dir) / "merged"
        if not merged_path.exists():
            logger.warning(f"Cannot promote: {merged_path} does not exist.")
            return False

        prod = Path(production_path)
        # Keep a backup of current production
        if prod.exists():
            backup = prod.parent / f"{prod.name}_pre_promotion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(str(prod), str(backup), dirs_exist_ok=True)
            logger.info(f"  Backed up production model: {backup}")

        shutil.copytree(str(merged_path), str(prod), dirs_exist_ok=True)
        logger.info(f"  Promoted: {merged_path} → {prod}")

        # Update baseline to the new production model
        logger.info("  Updating baseline to promoted model...")
        self.establish_baseline(str(prod))

        return True

    # ── Emergency Brake ──

    def should_halt(self) -> bool:
        """Check if the daemon should halt entirely due to repeated failures.

        Returns True if the last ``max_consecutive_drops`` evaluations were
        all regressions.
        """
        history = self._load_history()
        drops = self._count_consecutive_drops(history)
        if drops >= self._max_consec:
            logger.critical(
                f"EMERGENCY BRAKE: {drops} consecutive regressions. "
                f"Daemon should halt to prevent further degradation."
            )
            return True
        return False

    # ── Internal: Canary Benchmark ──

    def _run_canary_benchmark(self, model_path: str) -> Dict[str, Any]:
        """Load model, run 20 canary questions, unload, return scores."""
        from metis import Metis
        from metis.inference import MetisInference

        logger.info(f"  Loading model for sentinel evaluation...")
        t0 = time.time()
        metis = Metis.from_pretrained(model_path)
        engine = MetisInference(metis)
        load_time = time.time() - t0
        logger.info(f"  Model loaded in {load_time:.1f}s")

        results: List[Dict[str, Any]] = []
        correct_math = 0
        correct_qa = 0
        total_math = len(_MATH_CANARIES)
        total_qa = len(_QA_CANARIES)

        # ── Math canaries ──
        for i, item in enumerate(_MATH_CANARIES):
            try:
                result = engine.generate_cognitive(
                    item["q"], max_new_tokens=256,
                )
                answer_text = result.text
                thinking = result.thinking_text or ""
                full_text = answer_text + " " + thinking
                is_correct = _check_math(full_text, item["a"])
            except Exception as e:
                logger.warning(f"  Math canary {i} error: {e}")
                answer_text = ""
                is_correct = False

            if is_correct:
                correct_math += 1
            results.append({
                "type": "math",
                "question": item["q"],
                "gold": item["a"],
                "answer": answer_text[:200],
                "correct": is_correct,
            })
            logger.debug(
                f"  [{'✓' if is_correct else '✗'}] math/{i}: "
                f"{item['q'][:50]}... → {answer_text[:50]}"
            )

        # ── QA canaries ──
        for i, item in enumerate(_QA_CANARIES):
            try:
                result = engine.generate_cognitive(
                    item["q"], max_new_tokens=128,
                )
                answer_text = result.text
                thinking = result.thinking_text or ""
                full_text = answer_text + " " + thinking
                is_correct = _check_qa(full_text, item["a"])
            except Exception as e:
                logger.warning(f"  QA canary {i} error: {e}")
                answer_text = ""
                is_correct = False

            if is_correct:
                correct_qa += 1
            results.append({
                "type": "qa",
                "question": item["q"],
                "gold": item["a"],
                "answer": answer_text[:200],
                "correct": is_correct,
            })
            logger.debug(
                f"  [{'✓' if is_correct else '✗'}] qa/{i}: "
                f"{item['q'][:50]}... → {answer_text[:50]}"
            )

        # Cleanup
        del engine, metis
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        total = total_math + total_qa
        correct_total = correct_math + correct_qa

        scores = {
            "overall": correct_total / total if total > 0 else 0.0,
            "complex": correct_math / total_math if total_math > 0 else 0.0,
            "simple": correct_qa / total_qa if total_qa > 0 else 0.0,
            "per_question": results,
            "n_total": total,
            "n_correct": correct_total,
        }

        logger.info(
            f"  Canary results: {correct_total}/{total} "
            f"({scores['overall']*100:.1f}%) — "
            f"math={correct_math}/{total_math}, qa={correct_qa}/{total_qa}"
        )
        return scores

    # ── Internal: History ──

    def _load_history(self) -> List[Dict[str, Any]]:
        """Load evaluation history."""
        if not self._history_path.exists():
            return []
        try:
            with open(self._history_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _append_history(self, verdict: SentinelVerdict) -> None:
        """Append a verdict to evaluation history."""
        history = self._load_history()
        entry = {
            "timestamp": verdict.timestamp,
            "passed": verdict.passed,
            "overall_accuracy": verdict.overall_accuracy,
            "complex_accuracy": verdict.complex_accuracy,
            "simple_accuracy": verdict.simple_accuracy,
            "baseline_accuracy": verdict.baseline_accuracy,
            "accuracy_delta": verdict.accuracy_delta,
            "rollback_triggered": verdict.rollback_triggered,
            "reason": verdict.reason,
        }
        history.append(entry)

        # Keep at most 90 days of history
        if len(history) > 90:
            history = history[-90:]

        with open(self._history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def _count_consecutive_drops(self, history: List[Dict[str, Any]]) -> int:
        """Count how many consecutive evaluations showed accuracy regression."""
        drops = 0
        for entry in reversed(history):
            if entry.get("accuracy_delta", 0) < 0:
                drops += 1
            else:
                break
        return drops
