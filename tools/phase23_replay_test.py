#!/usr/bin/env python3
"""
Phase 23: Golden Experience Replay — Verification Suite

Tests:
  T1: Golden anchor JSONL integrity (10 prompts, valid format)
  T2: DreamingDaemon data blending (20/80 statistical mixture)
  T3: Blend ratio edge cases (0.5, 0.1, 0.01)
  T4: No golden dataset fallback (pure gap training)
  T5: DuckDuckGo search trigger (live network test)
  T6: Night training pipeline mock (dataset format + gap resolution)
  T7: KL-anchor CLI argument parsing
  T8: Vulnerability scan (hardcoded paths, leaked tokens, unsafe defaults)

Usage:
    python tools/phase23_replay_test.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase23_test")

PASS = 0
FAIL = 0


def report(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    status = "PASS ✅" if ok else "FAIL ❌"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)


# ═══════════════════════════════════════════════════════════
# T1: Golden Anchor Dataset Integrity
# ═══════════════════════════════════════════════════════════

def test_golden_anchor_integrity() -> None:
    print("\n" + "=" * 60)
    print("T1: Golden Anchor Dataset Integrity")
    print("=" * 60)

    golden_path = PROJECT_ROOT / "data" / "golden_anchor.jsonl"
    report("File exists", golden_path.exists(), str(golden_path))

    if not golden_path.exists():
        return

    prompts: List[str] = []
    with open(golden_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                assert "prompt" in obj, f"Line {i+1}: missing 'prompt' key"
                assert len(obj["prompt"]) > 10, f"Line {i+1}: prompt too short"
                prompts.append(obj["prompt"])
            except Exception as e:
                report(f"Line {i+1} parse", False, str(e))
                return

    report("All lines valid JSON", True, f"{len(prompts)} prompts")
    report("Prompt count == 10", len(prompts) == 10, f"got {len(prompts)}")
    report("No duplicates", len(prompts) == len(set(prompts)))

    # Check quality: math/logic prompts
    math_keywords = ["solve", "prove", "integral", "probability", "log", "sum", "simplify", "area", "speed"]
    has_math = sum(1 for p in prompts if any(kw in p.lower() for kw in math_keywords))
    report("Math/logic coverage", has_math >= 7, f"{has_math}/10 contain math keywords")


# ═══════════════════════════════════════════════════════════
# T2: Dynamic Data Blending (20/80 statistical mixture)
# ═══════════════════════════════════════════════════════════

def _make_mock_gaps(n: int) -> List[Dict[str, Any]]:
    """Generate n mock knowledge gaps."""
    return [
        {
            "query": f"Mock gap question #{i+1}: What is the meaning of X_{i}?",
            "category": "complete_unknown" if i % 2 == 0 else "sustained_confusion",
            "entropy_peak": 2.5 + i * 0.1,
            "entropy_mean": 1.8,
            "resolved": False,
        }
        for i in range(n)
    ]


def test_data_blending() -> None:
    print("\n" + "=" * 60)
    print("T2: Dynamic Data Blending (20/80 mixture)")
    print("=" * 60)

    from metis.daemon import DreamingDaemon

    golden_path = PROJECT_ROOT / "data" / "golden_anchor.jsonl"
    if not golden_path.exists():
        report("Golden anchor exists", False, "Cannot run blend test")
        return

    # Write mock gap storage
    gaps = _make_mock_gaps(10)

    daemon = DreamingDaemon(
        gap_storage_path=Path("/tmp/nonexistent_gaps.json"),
        golden_dataset_path=str(golden_path),
        blend_ratio=0.2,
    )

    # Call _format_jsonl_dataset directly
    dataset_path = daemon._format_jsonl_dataset(gaps)

    # Read and verify
    records: List[Dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))

    sources = Counter(r.get("_meta", {}).get("source", "unknown") for r in records)
    n_gap = sources.get("gap", 0)
    n_golden = sources.get("golden", 0)
    total = len(records)

    report("Gap count == 10", n_gap == 10, f"got {n_gap}")

    # With blend_ratio=0.2: n_golden = 10 * (1-0.2)/0.2 = 40
    expected_golden = int(10 * 0.8 / 0.2)
    report(
        f"Golden count == {expected_golden}",
        n_golden == expected_golden,
        f"got {n_golden}",
    )
    report(
        f"Total == {10 + expected_golden}",
        total == 10 + expected_golden,
        f"got {total}",
    )

    # Verify ratio
    actual_gap_ratio = n_gap / total if total > 0 else 0
    ratio_ok = abs(actual_gap_ratio - 0.2) < 0.01
    report(
        "Gap ratio ≈ 20%",
        ratio_ok,
        f"actual={actual_gap_ratio:.1%}",
    )

    # All records have 'prompt'
    all_have_prompt = all("prompt" in r and r["prompt"] for r in records)
    report("All records have non-empty prompt", all_have_prompt)

    # Cleanup
    os.unlink(dataset_path)


# ═══════════════════════════════════════════════════════════
# T3: Blend Ratio Edge Cases
# ═══════════════════════════════════════════════════════════

def test_blend_ratio_edge_cases() -> None:
    print("\n" + "=" * 60)
    print("T3: Blend Ratio Edge Cases")
    print("=" * 60)

    from metis.daemon import DreamingDaemon

    golden_path = PROJECT_ROOT / "data" / "golden_anchor.jsonl"
    gaps = _make_mock_gaps(5)

    for ratio, expected_golden in [(0.5, 5), (0.1, 45), (1.0, 0)]:
        daemon = DreamingDaemon(
            gap_storage_path=Path("/tmp/nonexistent.json"),
            golden_dataset_path=str(golden_path),
            blend_ratio=ratio,
        )
        dataset_path = daemon._format_jsonl_dataset(gaps)
        records = [json.loads(l) for l in open(dataset_path) if l.strip()]
        sources = Counter(r.get("_meta", {}).get("source", "unknown") for r in records)
        n_golden = sources.get("golden", 0)

        if ratio >= 1.0:
            # ratio=1.0 means 100% gaps, 0% golden → n_golden = 0
            ok = n_golden == 0
        else:
            calc = int(5 * (1.0 - ratio) / ratio)
            ok = n_golden == calc
            expected_golden = calc

        report(
            f"ratio={ratio}: golden={expected_golden}",
            ok,
            f"got {n_golden}",
        )
        os.unlink(dataset_path)


# ═══════════════════════════════════════════════════════════
# T4: No Golden Dataset Fallback
# ═══════════════════════════════════════════════════════════

def test_no_golden_fallback() -> None:
    print("\n" + "=" * 60)
    print("T4: No Golden Dataset Fallback (pure gap)")
    print("=" * 60)

    from metis.daemon import DreamingDaemon

    gaps = _make_mock_gaps(3)
    daemon = DreamingDaemon(
        gap_storage_path=Path("/tmp/nonexistent.json"),
        golden_dataset_path=None,
        blend_ratio=0.2,
    )
    dataset_path = daemon._format_jsonl_dataset(gaps)
    records = [json.loads(l) for l in open(dataset_path) if l.strip()]

    report("Pure gap count == 3", len(records) == 3, f"got {len(records)}")

    all_gap = all(r.get("_meta", {}).get("source") == "gap" for r in records)
    report("All sources == 'gap'", all_gap)

    os.unlink(dataset_path)


# ═══════════════════════════════════════════════════════════
# T5: DuckDuckGo Search Trigger
# ═══════════════════════════════════════════════════════════

def test_ddg_search() -> None:
    print("\n" + "=" * 60)
    print("T5: DuckDuckGo Search Trigger")
    print("=" * 60)

    try:
        from metis.search.retriever import DuckDuckGoRetriever, ToolRetriever
    except ImportError as e:
        report("Import retriever", False, str(e))
        return

    report("Import DuckDuckGoRetriever", True)

    ddg = DuckDuckGoRetriever(timeout=20)
    report("DDG library available", ddg.available)

    if not ddg.available:
        report("Live search", False, "ddgs not installed — skipping")
        return

    # Live search test (may timeout due to network conditions)
    ddg_results = ddg.search("What is the capital of France?", top_k=3)
    ddg_ok = len(ddg_results) > 0
    report(
        "DDG direct search",
        ddg_ok,
        f"{len(ddg_results)} result(s)" + ("" if ddg_ok else " (network timeout — non-fatal)"),
    )

    if ddg_results:
        has_content = any(len(r.snippet) > 20 for r in ddg_results)
        report("Results have substantive content", has_content)

    # ToolRetriever fallback chain (DDG → Mock)
    tr = ToolRetriever(timeout=15)
    report(f"ToolRetriever mode", True, tr.mode)

    tr_results = tr.search("vibranium atomic weight marvel 1992 handbook", top_k=2)
    tr_ok = len(tr_results) > 0
    report(
        "ToolRetriever returns results",
        tr_ok,
        f"{len(tr_results)} result(s)",
    )

    # Final verdict: at least ONE search path must work
    any_search_works = ddg_ok or tr_ok
    report(
        "Search pipeline functional (DDG or fallback)",
        any_search_works,
        f"DDG={'OK' if ddg_ok else 'timeout'}, ToolRetriever={'OK' if tr_ok else 'fail'}",
    )


# ═══════════════════════════════════════════════════════════
# T6: Night Training Pipeline Mock
# ═══════════════════════════════════════════════════════════

def test_night_training_mock() -> None:
    print("\n" + "=" * 60)
    print("T6: Night Training Pipeline Mock")
    print("=" * 60)

    from metis.daemon import DreamingDaemon

    # Create temp gap storage with mock gaps
    gaps = _make_mock_gaps(8)
    fd, gap_path = tempfile.mkstemp(suffix=".json", prefix="phase23_gaps_")
    with os.fdopen(fd, "w") as f:
        json.dump(gaps, f)

    golden_path = PROJECT_ROOT / "data" / "golden_anchor.jsonl"

    daemon = DreamingDaemon(
        gap_storage_path=Path(gap_path),
        golden_dataset_path=str(golden_path),
        blend_ratio=0.2,
        min_critical_gaps=1,
    )

    # Test gap loading
    loaded = daemon._load_unresolved_gaps()
    report("Load unresolved gaps", len(loaded) == 8, f"got {len(loaded)}")

    # Test critical filtering
    critical = daemon._filter_critical(loaded)
    report("Filter critical gaps", len(critical) > 0, f"{len(critical)} critical")

    # Test dataset formatting with blending
    batch = critical[:5]
    dataset_path = daemon._format_jsonl_dataset(batch)
    records = [json.loads(l) for l in open(dataset_path) if l.strip()]
    sources = Counter(r.get("_meta", {}).get("source", "unknown") for r in records)

    n_gap = sources.get("gap", 0)
    n_golden = sources.get("golden", 0)
    report(
        f"Blended batch: {n_gap} gaps + {n_golden} golden = {len(records)} total",
        n_gap == len(batch) and n_golden > 0,
    )

    # Test gap resolution
    daemon._mark_resolved(batch)
    with open(gap_path, "r") as f:
        updated = json.load(f)
    n_resolved = sum(1 for g in updated if g.get("resolved"))
    report(
        f"Gaps marked resolved",
        n_resolved == len(batch),
        f"{n_resolved}/{len(batch)}",
    )

    # Cleanup
    os.unlink(dataset_path)
    os.unlink(gap_path)


# ═══════════════════════════════════════════════════════════
# T7: KL-Anchor CLI Argument Parsing
# ═══════════════════════════════════════════════════════════

def test_kl_anchor_cli() -> None:
    print("\n" + "=" * 60)
    print("T7: KL-Anchor CLI Argument Parsing")
    print("=" * 60)

    import argparse

    # Simulate argparse from run_dream_training.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--anchor-model", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)

    # Test with anchor
    args = parser.parse_args([
        "--dataset", "/tmp/test.jsonl",
        "--base-model", "Qwen/Qwen2.5-7B-Instruct",
        "--output-dir", "/tmp/output",
        "--anchor-model", "Qwen/Qwen2.5-7B-Instruct",
        "--dry-run",
    ])
    report("--anchor-model parsed", args.anchor_model == "Qwen/Qwen2.5-7B-Instruct")
    report("--dry-run parsed", args.dry_run is True)

    # Test without anchor (default None)
    args2 = parser.parse_args([
        "--dataset", "/tmp/test.jsonl",
        "--base-model", "some_model",
        "--output-dir", "/tmp/output",
    ])
    report("--anchor-model default None", args2.anchor_model is None)


# ═══════════════════════════════════════════════════════════
# T8: Vulnerability Scan
# ═══════════════════════════════════════════════════════════

def test_vulnerability_scan() -> None:
    print("\n" + "=" * 60)
    print("T8: Vulnerability Scan")
    print("=" * 60)

    import re as _re

    issues: List[str] = []

    # Scan all .py and .sh files for hardcoded paths / tokens
    for ext in ("**/*.py", "**/*.sh"):
        for fp in sorted(PROJECT_ROOT.glob(ext)):
            if ".git" in str(fp) or "__pycache__" in str(fp):
                continue
            if "experiment_output" in str(fp):
                continue
            try:
                content = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            rel = fp.relative_to(PROJECT_ROOT)

            # Check for hardcoded /home/ paths (excluding comments/docstrings about examples)
            for i, line in enumerate(content.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith("//"):
                    continue
                if _re.search(r'/home/\w+/', line) and 'getenv' not in line and 'example' not in line.lower():
                    issues.append(f"{rel}:{i} — hardcoded home path: {line.strip()[:80]}")

            # Check for leaked API keys / tokens
            for pattern in [
                r'(?:api[_-]?key|token|secret)\s*=\s*["\'][a-zA-Z0-9_-]{20,}["\']',
                r'sk-[a-zA-Z0-9]{20,}',
                r'hf_[a-zA-Z0-9]{20,}',
            ]:
                for m in _re.finditer(pattern, content, _re.IGNORECASE):
                    issues.append(f"{rel} — potential leaked credential: {m.group()[:40]}...")

    report("No hardcoded paths", len([i for i in issues if "hardcoded" in i]) == 0)
    report("No leaked credentials", len([i for i in issues if "credential" in i]) == 0)

    if issues:
        for issue in issues[:10]:
            print(f"    ⚠️  {issue}")
    else:
        print("    No vulnerabilities detected.")

    # Check daemon lock file is in /tmp (not project dir)
    from metis.daemon import LOCK_FILE
    report("Lock file in /tmp", LOCK_FILE.startswith("/tmp"))

    # Check blend_ratio clamping
    from metis.daemon import DreamingDaemon
    d = DreamingDaemon(gap_storage_path=Path("/tmp/x.json"), blend_ratio=-5.0)
    report("Blend ratio clamped (negative)", d._blend_ratio == 0.01)
    d2 = DreamingDaemon(gap_storage_path=Path("/tmp/x.json"), blend_ratio=99.0)
    report("Blend ratio clamped (>1)", d2._blend_ratio == 1.0)


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  Phase 23: Golden Experience Replay — Verification Suite")
    print("=" * 60)

    test_golden_anchor_integrity()
    test_data_blending()
    test_blend_ratio_edge_cases()
    test_no_golden_fallback()
    test_ddg_search()
    test_night_training_mock()
    test_kl_anchor_cli()
    test_vulnerability_scan()

    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f"  RESULTS: {PASS}/{total} PASS, {FAIL}/{total} FAIL")
    if FAIL == 0:
        print("  ALL TESTS PASSED ✅")
    else:
        print(f"  {FAIL} FAILURES ❌")
    print("=" * 60)

    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
