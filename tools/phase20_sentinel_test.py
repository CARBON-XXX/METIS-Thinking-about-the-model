#!/usr/bin/env python3
"""
Phase 20: Degradation Sentinel — Mode Collapse Defense Validation

Tests:
  T1. Baseline establishment — run canary benchmark on production model,
      verify scores stored correctly.
  T2. Live regression gate — re-evaluate same model, verify PASS
      (no degradation against itself).
  T3. Simulated degradation — mock a model that scores lower than
      baseline, verify FAIL + rollback triggered.
  T4. Consecutive drop detection — simulate 3 consecutive regressions,
      verify emergency brake fires.
  T5. Daemon integration — wire sentinel into DreamingDaemon, run a
      mock cycle, verify sentinel gate executes.
  T6. Evolutionary promotion guard — verify promote() backs up old
      model and copies new weights.

Output: Sentinel Defense Report
"""
from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase20")

# ── Report accumulator ──
REPORT: Dict[str, Dict[str, Any]] = {}
PASS_COUNT = 0
FAIL_COUNT = 0


def record(test_name: str, status: str, details: Dict[str, Any]) -> None:
    global PASS_COUNT, FAIL_COUNT
    REPORT[test_name] = {"status": status, **details}
    if status == "PASS":
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    symbol = "✓" if status == "PASS" else "✗"
    logger.info(f"  [{symbol}] {test_name}: {status}")


# ═══════════════════════════════════════════════════════════
# TEST 1: BASELINE ESTABLISHMENT
# ═══════════════════════════════════════════════════════════

def test_baseline_establishment() -> Dict[str, Any]:
    """Run canary benchmark on production model and establish baseline."""
    logger.info("=" * 60)
    logger.info("TEST 1: Baseline Establishment")
    logger.info("=" * 60)

    from metis.sentinel import DegradationSentinel

    record("T1_import", "PASS", {"class": "DegradationSentinel"})

    # Use temp files to avoid polluting data/
    tmpdir = tempfile.mkdtemp(prefix="sentinel_test_")
    baseline_path = os.path.join(tmpdir, "sentinel_baseline.json")
    history_path = os.path.join(tmpdir, "sentinel_history.json")

    sentinel = DegradationSentinel(
        baseline_path=baseline_path,
        history_path=history_path,
        max_accuracy_drop_pct=5.0,
        min_absolute_accuracy=70.0,
        max_consecutive_drops=3,
    )

    record("T1_init", "PASS", {
        "baseline_path": baseline_path,
        "history_path": history_path,
        "thresholds": {
            "max_drop": 5.0,
            "min_abs": 70.0,
            "max_consec": 3,
        },
    })

    MODEL_PATH = str(PROJECT_ROOT / "experiment_output_dpo_balanced" / "metis_dpo_cognitive")

    if not Path(MODEL_PATH).exists():
        record("T1_baseline", "FAIL", {"reason": f"Model not found: {MODEL_PATH}"})
        return {"tmpdir": tmpdir, "sentinel": sentinel, "model_path": MODEL_PATH}

    logger.info(f"  Establishing baseline on: {MODEL_PATH}")
    t0 = time.time()
    baseline = sentinel.establish_baseline(MODEL_PATH)
    elapsed = time.time() - t0

    # Verify baseline file was created
    baseline_exists = os.path.exists(baseline_path)
    record("T1_file_created", "PASS" if baseline_exists else "FAIL", {
        "path": baseline_path,
        "exists": baseline_exists,
    })

    # Verify baseline content
    overall = baseline.get("overall_accuracy", 0)
    complex_acc = baseline.get("complex_accuracy", 0)
    simple_acc = baseline.get("simple_accuracy", 0)

    # Baseline should be reasonable (>50% at minimum)
    baseline_sane = overall > 0.5
    record("T1_baseline_quality", "PASS" if baseline_sane else "FAIL", {
        "overall": f"{overall*100:.1f}%",
        "complex": f"{complex_acc*100:.1f}%",
        "simple": f"{simple_acc*100:.1f}%",
        "elapsed_s": round(elapsed, 1),
        "n_questions": 20,
    })

    # Verify per-question details
    per_q = baseline.get("per_question", [])
    record("T1_per_question", "PASS" if len(per_q) == 20 else "FAIL", {
        "count": len(per_q),
        "expected": 20,
        "math_correct": sum(1 for q in per_q if q["type"] == "math" and q["correct"]),
        "qa_correct": sum(1 for q in per_q if q["type"] == "qa" and q["correct"]),
    })

    return {
        "tmpdir": tmpdir,
        "sentinel": sentinel,
        "model_path": MODEL_PATH,
        "baseline": baseline,
    }


# ═══════════════════════════════════════════════════════════
# TEST 2: LIVE REGRESSION GATE (same model → should PASS)
# ═══════════════════════════════════════════════════════════

def test_live_regression(ctx: Dict[str, Any]) -> None:
    """Re-evaluate same model — should pass since it's identical."""
    logger.info("=" * 60)
    logger.info("TEST 2: Live Regression Gate (same model)")
    logger.info("=" * 60)

    sentinel = ctx["sentinel"]
    model_path = ctx["model_path"]

    verdict = sentinel.evaluate(model_path)

    record("T2_verdict_pass", "PASS" if verdict.passed else "FAIL", {
        "passed": verdict.passed,
        "overall": f"{verdict.overall_accuracy*100:.1f}%",
        "baseline": f"{verdict.baseline_accuracy*100:.1f}%",
        "delta": f"{verdict.accuracy_delta*100:+.1f}pp",
        "reason": verdict.reason,
    })

    record("T2_no_rollback", "PASS" if not verdict.rollback_triggered else "FAIL", {
        "rollback": verdict.rollback_triggered,
    })

    # Verify history was updated
    history_path = ctx["sentinel"]._history_path
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        record("T2_history_recorded", "PASS" if len(history) >= 1 else "FAIL", {
            "entries": len(history),
        })
    else:
        record("T2_history_recorded", "FAIL", {"reason": "History file not created"})


# ═══════════════════════════════════════════════════════════
# TEST 3: SIMULATED DEGRADATION (mock low scores → FAIL)
# ═══════════════════════════════════════════════════════════

def test_simulated_degradation(ctx: Dict[str, Any]) -> None:
    """Mock a model that scores far below baseline → verify rollback."""
    logger.info("=" * 60)
    logger.info("TEST 3: Simulated Degradation (mock low accuracy)")
    logger.info("=" * 60)

    sentinel = ctx["sentinel"]

    # Mock _run_canary_benchmark to return low scores
    fake_scores = {
        "overall": 0.40,   # 40% — way below 70% floor
        "complex": 0.30,   # 30% math — catastrophic
        "simple": 0.50,    # 50% QA
        "per_question": [
            {"type": "math", "question": f"q{i}", "gold": "0",
             "answer": "wrong", "correct": False}
            for i in range(10)
        ] + [
            {"type": "qa", "question": f"q{i}", "gold": "x",
             "answer": "wrong", "correct": i < 5}
            for i in range(10)
        ],
        "n_total": 20,
        "n_correct": 8,
    }

    with patch.object(sentinel, "_run_canary_benchmark", return_value=fake_scores):
        verdict = sentinel.evaluate("/tmp/fake_degraded_model")

    record("T3_verdict_fail", "PASS" if not verdict.passed else "FAIL", {
        "passed": verdict.passed,
        "overall": f"{verdict.overall_accuracy*100:.1f}%",
        "reason": verdict.reason,
    })

    record("T3_rollback_triggered", "PASS" if verdict.rollback_triggered else "FAIL", {
        "rollback": verdict.rollback_triggered,
    })

    # Verify specific failure reasons
    has_floor = "floor" in verdict.reason.lower() or "below" in verdict.reason.lower()
    has_collapse = "collapse" in verdict.reason.lower() or "<50%" in verdict.reason
    record("T3_failure_reasons", "PASS" if (has_floor or has_collapse) else "FAIL", {
        "reason": verdict.reason,
        "has_floor_check": has_floor,
        "has_collapse_check": has_collapse,
    })


# ═══════════════════════════════════════════════════════════
# TEST 4: CONSECUTIVE DROP DETECTION (emergency brake)
# ═══════════════════════════════════════════════════════════

def test_consecutive_drops(ctx: Dict[str, Any]) -> None:
    """Simulate 3 consecutive regressions → emergency brake."""
    logger.info("=" * 60)
    logger.info("TEST 4: Consecutive Drop Detection (emergency brake)")
    logger.info("=" * 60)

    tmpdir = tempfile.mkdtemp(prefix="sentinel_consec_")
    baseline_path = os.path.join(tmpdir, "baseline.json")
    history_path = os.path.join(tmpdir, "history.json")

    from metis.sentinel import DegradationSentinel

    sentinel = DegradationSentinel(
        baseline_path=baseline_path,
        history_path=history_path,
        max_accuracy_drop_pct=5.0,
        min_absolute_accuracy=70.0,
        max_consecutive_drops=3,
    )

    # Create a fake baseline
    with open(baseline_path, "w") as f:
        json.dump({
            "overall_accuracy": 0.90,
            "complex_accuracy": 0.90,
            "simple_accuracy": 0.90,
        }, f)

    # Simulate 3 consecutive drops via history
    fake_history = [
        {"timestamp": "2026-03-07", "passed": False, "accuracy_delta": -0.02,
         "overall_accuracy": 0.88, "reason": "drop 1"},
        {"timestamp": "2026-03-08", "passed": False, "accuracy_delta": -0.03,
         "overall_accuracy": 0.85, "reason": "drop 2"},
        {"timestamp": "2026-03-09", "passed": False, "accuracy_delta": -0.04,
         "overall_accuracy": 0.81, "reason": "drop 3"},
    ]
    with open(history_path, "w") as f:
        json.dump(fake_history, f)

    # should_halt() should trigger after 3 consecutive drops
    should_halt = sentinel.should_halt()
    record("T4_emergency_brake", "PASS" if should_halt else "FAIL", {
        "should_halt": should_halt,
        "consecutive_drops": 3,
        "threshold": 3,
    })

    # With only 2 drops, should NOT halt
    with open(history_path, "w") as f:
        json.dump(fake_history[:2], f)

    should_not_halt = not sentinel.should_halt()
    record("T4_no_false_alarm", "PASS" if should_not_halt else "FAIL", {
        "should_halt": not should_not_halt,
        "consecutive_drops": 2,
        "threshold": 3,
    })

    # With a pass in between, should NOT halt
    mixed_history = [
        {"timestamp": "2026-03-07", "accuracy_delta": -0.02},
        {"timestamp": "2026-03-08", "accuracy_delta": 0.01},  # recovery!
        {"timestamp": "2026-03-09", "accuracy_delta": -0.03},
    ]
    with open(history_path, "w") as f:
        json.dump(mixed_history, f)

    should_not_halt2 = not sentinel.should_halt()
    record("T4_recovery_resets", "PASS" if should_not_halt2 else "FAIL", {
        "should_halt": not should_not_halt2,
        "pattern": "drop-recovery-drop",
        "consecutive": 1,
    })

    shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════
# TEST 5: DAEMON INTEGRATION (sentinel gate in _cycle)
# ═══════════════════════════════════════════════════════════

def test_daemon_integration(ctx: Dict[str, Any]) -> None:
    """Verify DreamingDaemon respects sentinel verdict."""
    logger.info("=" * 60)
    logger.info("TEST 5: Daemon Integration")
    logger.info("=" * 60)

    from metis.daemon import DreamingDaemon
    from metis.sentinel import DegradationSentinel

    tmpdir = tempfile.mkdtemp(prefix="daemon_sentinel_")
    gap_path = os.path.join(tmpdir, "gaps.json")
    baseline_path = os.path.join(tmpdir, "baseline.json")
    history_path = os.path.join(tmpdir, "history.json")

    # Create mock gaps
    with open(gap_path, "w") as f:
        json.dump([
            {"query": "test q", "category": "complete_unknown",
             "entropy_peak": 3.0, "resolved": False},
        ], f)

    sentinel = DegradationSentinel(
        baseline_path=baseline_path,
        history_path=history_path,
    )

    daemon = DreamingDaemon(
        gap_storage_path=Path(gap_path),
        sentinel=sentinel,
        evolutionary=False,
        gpu_idle_threshold=100.0,
    )

    # Verify sentinel is wired
    record("T5_sentinel_wired", "PASS" if daemon._sentinel is not None else "FAIL", {
        "has_sentinel": daemon._sentinel is not None,
        "evolutionary": daemon._evolutionary,
    })

    # Test _run_sentinel_gate with mock — PASS case
    with open(baseline_path, "w") as f:
        json.dump({"overall_accuracy": 0.80, "complex_accuracy": 0.80,
                    "simple_accuracy": 0.80}, f)

    pass_scores = {
        "overall": 0.85, "complex": 0.80, "simple": 0.90,
        "per_question": [], "n_total": 20, "n_correct": 17,
    }
    with patch.object(sentinel, "_run_canary_benchmark", return_value=pass_scores):
        # Create a fake merged dir
        merged_dir = os.path.join(daemon._output_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)
        gate_result = daemon._run_sentinel_gate()

    record("T5_gate_pass", "PASS" if gate_result else "FAIL", {
        "gate_result": gate_result,
    })

    # Test _run_sentinel_gate with mock — FAIL case
    fail_scores = {
        "overall": 0.40, "complex": 0.30, "simple": 0.50,
        "per_question": [], "n_total": 20, "n_correct": 8,
    }
    with patch.object(sentinel, "_run_canary_benchmark", return_value=fail_scores):
        gate_result_fail = daemon._run_sentinel_gate()

    record("T5_gate_fail", "PASS" if not gate_result_fail else "FAIL", {
        "gate_result": gate_result_fail,
        "rollback_count": daemon._rollback_count,
    })

    record("T5_rollback_count", "PASS" if daemon._rollback_count >= 1 else "FAIL", {
        "rollback_count": daemon._rollback_count,
    })

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)
    shutil.rmtree(daemon._output_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════
# TEST 6: ROLLBACK AND PROMOTION MECHANICS
# ═══════════════════════════════════════════════════════════

def test_rollback_and_promotion() -> None:
    """Verify rollback archives and promotion copies correctly."""
    logger.info("=" * 60)
    logger.info("TEST 6: Rollback & Promotion Mechanics")
    logger.info("=" * 60)

    from metis.sentinel import DegradationSentinel

    tmpdir = tempfile.mkdtemp(prefix="sentinel_mechanics_")

    sentinel = DegradationSentinel(
        baseline_path=os.path.join(tmpdir, "baseline.json"),
        history_path=os.path.join(tmpdir, "history.json"),
    )

    # ── 6a. Rollback ──
    output_dir = os.path.join(tmpdir, "training_output")
    merged_dir = os.path.join(output_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)

    # Create a fake model file
    fake_model = os.path.join(merged_dir, "model.safetensors")
    with open(fake_model, "w") as f:
        f.write("FAKE_WEIGHTS")

    archive = sentinel.rollback(output_dir)

    # Merged should be gone, archive should exist
    merged_gone = not os.path.exists(merged_dir)
    archive_exists = archive.exists()
    weights_archived = (archive / "model.safetensors").exists() if archive_exists else False

    record("T6_rollback_archive", "PASS" if (merged_gone and archive_exists and weights_archived) else "FAIL", {
        "merged_removed": merged_gone,
        "archive_created": archive_exists,
        "weights_preserved": weights_archived,
        "archive_path": str(archive),
    })

    # ── 6b. Promotion ──
    # Create a new "merged" model
    os.makedirs(merged_dir, exist_ok=True)
    with open(os.path.join(merged_dir, "config.json"), "w") as f:
        json.dump({"model_type": "test", "version": "v2"}, f)
    with open(os.path.join(merged_dir, "model.safetensors"), "w") as f:
        f.write("NEW_WEIGHTS_V2")

    prod_path = os.path.join(tmpdir, "production_model")
    os.makedirs(prod_path, exist_ok=True)
    with open(os.path.join(prod_path, "config.json"), "w") as f:
        json.dump({"model_type": "test", "version": "v1"}, f)

    # Mock establish_baseline to avoid loading real model
    with patch.object(sentinel, "establish_baseline", return_value={}):
        promoted = sentinel.promote(output_dir, prod_path)

    record("T6_promotion_success", "PASS" if promoted else "FAIL", {
        "promoted": promoted,
    })

    # Verify production was updated
    if promoted:
        with open(os.path.join(prod_path, "config.json")) as f:
            config = json.load(f)
        version_updated = config.get("version") == "v2"
        record("T6_production_updated", "PASS" if version_updated else "FAIL", {
            "new_version": config.get("version"),
        })

        # Verify backup was created
        backups = [d for d in os.listdir(tmpdir)
                   if d.startswith("production_model_pre_promotion")]
        record("T6_backup_created", "PASS" if len(backups) >= 1 else "FAIL", {
            "backup_count": len(backups),
            "backup_name": backups[0] if backups else "(none)",
        })

    shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════

def print_report() -> None:
    """Output the Sentinel Defense Report."""
    print("\n")
    print("╔" + "═" * 62 + "╗")
    print("║   SENTINEL DEFENSE REPORT — Phase 20                        ║")
    print("║   Evolutionary Mode Collapse Prevention                     ║")
    print("╚" + "═" * 62 + "╝")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total: {PASS_COUNT + FAIL_COUNT} checks  |  "
          f"PASS: {PASS_COUNT}  |  FAIL: {FAIL_COUNT}")
    print()

    groups: Dict[str, List] = {}
    for name, info in REPORT.items():
        prefix = name.split("_")[0]
        groups.setdefault(prefix, []).append((name, info))

    titles = {
        "T1": "TEST 1: Baseline Establishment",
        "T2": "TEST 2: Live Regression Gate",
        "T3": "TEST 3: Simulated Degradation",
        "T4": "TEST 4: Consecutive Drop Detection",
        "T5": "TEST 5: Daemon Integration",
        "T6": "TEST 6: Rollback & Promotion",
    }

    for prefix in ["T1", "T2", "T3", "T4", "T5", "T6"]:
        items = groups.get(prefix, [])
        if not items:
            continue
        title = titles.get(prefix, prefix)
        all_pass = all(i[1]["status"] == "PASS" for i in items)
        status_str = "✓ ALL PASS" if all_pass else "✗ HAS FAILURES"
        print(f"  ┌─ {title}")
        print(f"  │  Status: {status_str}")
        for name, info in items:
            status = info["status"]
            symbol = "✓" if status == "PASS" else "✗"
            short_name = name[len(prefix) + 1:]
            detail_items = {k: v for k, v in info.items() if k != "status"}
            shown = list(detail_items.items())[:3]
            detail_str = "  " + ", ".join(f"{k}={v}" for k, v in shown) if shown else ""
            print(f"  │  [{symbol}] {short_name}{detail_str}")
        print(f"  └{'─' * 50}")
        print()

    if FAIL_COUNT == 0:
        print("  ══════════════════════════════════════════════")
        print("  ║  VERDICT: SENTINEL DEFENSE OPERATIONAL  ✓  ║")
        print("  ══════════════════════════════════════════════")
    else:
        print("  ══════════════════════════════════════════════")
        print(f"  ║  VERDICT: {FAIL_COUNT} FAILURE(S) DETECTED  ✗       ║")
        print("  ══════════════════════════════════════════════")
    print()


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main() -> None:
    logger.info("Phase 20: Degradation Sentinel — Mode Collapse Defense")
    print()

    # T1: Baseline (heavy — loads model)
    try:
        ctx = test_baseline_establishment()
    except Exception as e:
        record("T1_fatal", "FAIL", {"error": str(e)})
        logger.error(f"T1 fatal: {e}", exc_info=True)
        ctx = {}

    print()

    # T2: Live regression (reuses loaded model scores)
    if ctx.get("model_path"):
        try:
            test_live_regression(ctx)
        except Exception as e:
            record("T2_fatal", "FAIL", {"error": str(e)})
            logger.error(f"T2 fatal: {e}", exc_info=True)
    else:
        record("T2_skipped", "FAIL", {"reason": "No model context from T1"})

    # Free GPU before mock tests
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    print()

    # T3: Simulated degradation (mock, no GPU)
    if ctx.get("sentinel"):
        try:
            test_simulated_degradation(ctx)
        except Exception as e:
            record("T3_fatal", "FAIL", {"error": str(e)})
            logger.error(f"T3 fatal: {e}", exc_info=True)

    print()

    # T4: Consecutive drops (mock, no GPU)
    try:
        test_consecutive_drops(ctx)
    except Exception as e:
        record("T4_fatal", "FAIL", {"error": str(e)})
        logger.error(f"T4 fatal: {e}", exc_info=True)

    print()

    # T5: Daemon integration (mock, no GPU)
    try:
        test_daemon_integration(ctx)
    except Exception as e:
        record("T5_fatal", "FAIL", {"error": str(e)})
        logger.error(f"T5 fatal: {e}", exc_info=True)

    print()

    # T6: Rollback and promotion mechanics (pure filesystem, no GPU)
    try:
        test_rollback_and_promotion()
    except Exception as e:
        record("T6_fatal", "FAIL", {"error": str(e)})
        logger.error(f"T6 fatal: {e}", exc_info=True)

    # Cleanup T1 temp dir
    if ctx.get("tmpdir"):
        shutil.rmtree(ctx["tmpdir"], ignore_errors=True)

    # Report
    print_report()

    # Save JSON
    report_path = PROJECT_ROOT / "phase20_sentinel_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "pass_count": PASS_COUNT,
            "fail_count": FAIL_COUNT,
            "checks": REPORT,
        }, f, indent=2, default=str)
    logger.info(f"Report saved: {report_path}")

    sys.exit(0 if FAIL_COUNT == 0 else 1)


if __name__ == "__main__":
    main()
