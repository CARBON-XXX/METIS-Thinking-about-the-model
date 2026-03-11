#!/usr/bin/env python3
"""
Phase 22 Integration Verification — Validate all断链修复.

Tests:
  T1: Import verification — all modified modules importable
  T2: LangChain stub — no crash without langchain
  T3: LlamaIndex stub — no crash without llamaindex  
  T4: CLI dry-run — python -m metis info
  T5: Online Loop reward fn — correct Metis API wiring
  T6: TRL adapter — generate_cognitive + trace property
  T7: Daemon EGTS mode — training_mode parameter validation
  T8: Bridge broadcast — generate_cognitive triggers listeners
  T9: Serve module — create_app importable
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PYTHON = sys.executable

# ─────────────────────────────────────────
# Test infrastructure
# ─────────────────────────────────────────

results: List[Dict[str, Any]] = []


def run_test(test_id: str, description: str):
    """Decorator for test functions."""
    def decorator(fn):
        def wrapper():
            print(f"\n{'='*60}")
            print(f"  {test_id}: {description}")
            print(f"{'='*60}")
            t0 = time.perf_counter()
            try:
                passed, details = fn()
                elapsed = (time.perf_counter() - t0) * 1000
                status = "PASS" if passed else "FAIL"
                print(f"  [{status}] {test_id} ({elapsed:.0f}ms)")
                if details:
                    for d in details:
                        print(f"    - {d}")
                results.append({
                    "test_id": test_id,
                    "description": description,
                    "status": status,
                    "elapsed_ms": round(elapsed, 1),
                    "details": details,
                })
                return passed
            except Exception as e:
                elapsed = (time.perf_counter() - t0) * 1000
                tb = traceback.format_exc()
                print(f"  [FAIL] {test_id} — Exception: {e}")
                print(f"    {tb}")
                results.append({
                    "test_id": test_id,
                    "description": description,
                    "status": "FAIL",
                    "elapsed_ms": round(elapsed, 1),
                    "details": [f"Exception: {e}"],
                    "traceback": tb,
                })
                return False
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator


# ─────────────────────────────────────────
# T1: Import verification
# ─────────────────────────────────────────

@run_test("T1", "Import verification — all modified modules")
def test_imports() -> Tuple[bool, List[str]]:
    modules = [
        "metis.integrations.langchain",
        "metis.integrations.llamaindex",
        "metis.pipeline.online_loop",
        "metis.__main__",
        "metis.training.trl_adapter",
        "metis.daemon",
        "metis.inference",
        "metis.serve",
    ]
    details: List[str] = []
    all_ok = True
    for mod_name in modules:
        try:
            importlib.import_module(mod_name)
            details.append(f"✓ {mod_name}")
        except Exception as e:
            details.append(f"✗ {mod_name}: {e}")
            all_ok = False
    return all_ok, details


# ─────────────────────────────────────────
# T2: LangChain stub
# ─────────────────────────────────────────

@run_test("T2", "LangChain stub — no crash without langchain")
def test_langchain_stub() -> Tuple[bool, List[str]]:
    details: List[str] = []

    # Check MetisInference import is correct (not MetisInferenceEngine)
    import metis.integrations.langchain as lc_mod
    source = Path(lc_mod.__file__).read_text()

    if "MetisInferenceEngine" in source:
        details.append("✗ Still references MetisInferenceEngine (dead import)")
        return False, details
    details.append("✓ No MetisInferenceEngine reference")

    if "from ..inference import MetisInference" in source or "MetisInference" in source:
        details.append("✓ Correct MetisInference import")
    else:
        details.append("✗ Missing MetisInference import")
        return False, details

    if "generate_cognitive" in source:
        details.append("✓ Uses generate_cognitive() API")
    else:
        details.append("✗ Missing generate_cognitive() call")
        return False, details

    # Verify stub classes exist
    assert hasattr(lc_mod, "MetisCallbackHandler"), "Missing MetisCallbackHandler"
    assert hasattr(lc_mod, "MetisLLM"), "Missing MetisLLM"
    details.append("✓ Stub classes present (MetisCallbackHandler, MetisLLM)")

    return True, details


# ─────────────────────────────────────────
# T3: LlamaIndex stub
# ─────────────────────────────────────────

@run_test("T3", "LlamaIndex stub — no crash without llamaindex")
def test_llamaindex_stub() -> Tuple[bool, List[str]]:
    details: List[str] = []

    import metis.integrations.llamaindex as li_mod

    # Check stub classes exist
    for cls_name in ["MetisCallbackHandler", "MetisResponseEvaluator", "MetisRetrieverGuard"]:
        if hasattr(li_mod, cls_name):
            details.append(f"✓ {cls_name} accessible")
        else:
            details.append(f"✗ {cls_name} missing")
            return False, details

    # Verify stubs raise ImportError when instantiated (if llamaindex not installed)
    try:
        from llama_index.core.base.response.schema import Response  # noqa
        details.append("⚠ llama-index-core installed — stubs not tested")
    except ImportError:
        try:
            li_mod.MetisResponseEvaluator()
            details.append("✗ MetisResponseEvaluator stub did not raise")
            return False, details
        except ImportError:
            details.append("✓ MetisResponseEvaluator stub raises ImportError")

        try:
            li_mod.MetisRetrieverGuard()
            details.append("✗ MetisRetrieverGuard stub did not raise")
            return False, details
        except ImportError:
            details.append("✓ MetisRetrieverGuard stub raises ImportError")

    return True, details


# ─────────────────────────────────────────
# T4: CLI dry-run
# ─────────────────────────────────────────

@run_test("T4", "CLI dry-run — python -m metis info")
def test_cli() -> Tuple[bool, List[str]]:
    details: List[str] = []
    result = subprocess.run(
        [PYTHON, "-m", "metis", "info"],
        capture_output=True, text=True, timeout=30,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        details.append("✓ Exit code 0")
        # Check banner appears
        if "METIS" in result.stdout:
            details.append("✓ Banner printed")
        else:
            details.append("⚠ Banner not found in output")
    else:
        details.append(f"✗ Exit code {result.returncode}")
        if result.stderr:
            details.append(f"  stderr: {result.stderr[:200]}")
        return False, details

    # Verify --thinking flag removed
    help_result = subprocess.run(
        [PYTHON, "-m", "metis", "attach", "--help"],
        capture_output=True, text=True, timeout=10,
        cwd=str(PROJECT_ROOT),
    )
    if "--thinking" in help_result.stdout:
        details.append("✗ Stale --thinking flag still present")
        return False, details
    details.append("✓ --thinking flag removed from CLI")

    return True, details


# ─────────────────────────────────────────
# T5: Online Loop reward fn API wiring
# ─────────────────────────────────────────

@run_test("T5", "Online Loop reward fn — correct Metis API wiring")
def test_online_loop() -> Tuple[bool, List[str]]:
    details: List[str] = []

    source_path = PROJECT_ROOT / "metis" / "pipeline" / "online_loop.py"
    source = source_path.read_text()

    # Should NOT have old broken patterns
    # Use patterns specific enough to avoid matching correct self._metis.step() etc.
    broken_patterns = [
        "from metis import MetisInference",
        "metis.reset()",
        "metis.get_trace()",
        "MetisInference(\n            model=",
    ]
    for pat in broken_patterns:
        if pat in source:
            details.append(f"✗ Still contains broken pattern: {pat!r}")
            return False, details
    details.append("✓ No broken API patterns")

    # Should have correct patterns
    correct_patterns = [
        "Metis.attach(model, tokenizer)",
        "self._metis.start_session(",
        "self._metis.step(",
        "self._metis.end_session()",
        "self._metis.trace",
    ]
    for pat in correct_patterns:
        if pat not in source:
            details.append(f"✗ Missing correct pattern: {pat!r}")
            return False, details
    details.append("✓ All correct Metis API patterns present")

    return True, details


# ─────────────────────────────────────────
# T6: TRL adapter API
# ─────────────────────────────────────────

@run_test("T6", "TRL adapter — generate_cognitive + trace property")
def test_trl_adapter() -> Tuple[bool, List[str]]:
    details: List[str] = []

    source_path = PROJECT_ROOT / "metis" / "training" / "trl_adapter.py"
    source = source_path.read_text()

    if "generate_cognitive(prompt)" in source:
        details.append("✓ Uses generate_cognitive()")
    else:
        details.append("✗ Missing generate_cognitive() call")
        return False, details

    if "self._inference._metis.trace" in source:
        details.append("✓ Accesses trace as property")
    else:
        details.append("✗ Missing trace property access")
        return False, details

    # Should NOT have old broken patterns
    if "self._inference.generate(prompt)" in source:
        details.append("✗ Still uses old generate() call")
        return False, details
    if "get_trace()" in source:
        details.append("✗ Still uses get_trace() method call")
        return False, details
    details.append("✓ No broken API patterns")

    return True, details


# ─────────────────────────────────────────
# T7: Daemon EGTS mode
# ─────────────────────────────────────────

@run_test("T7", "Daemon EGTS mode — training_mode parameter validation")
def test_daemon_egts() -> Tuple[bool, List[str]]:
    details: List[str] = []
    import tempfile

    from metis.daemon import DreamingDaemon

    # Test valid modes
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w") as f:
        json.dump([], f)
        f.flush()

        # grpo mode (default)
        d1 = DreamingDaemon(gap_storage_path=Path(f.name), training_mode="grpo")
        assert d1._training_mode == "grpo"
        details.append("✓ grpo mode accepted")

        # egts mode
        d2 = DreamingDaemon(gap_storage_path=Path(f.name), training_mode="egts")
        assert d2._training_mode == "egts"
        details.append("✓ egts mode accepted")

        # invalid mode
        try:
            DreamingDaemon(gap_storage_path=Path(f.name), training_mode="invalid")
            details.append("✗ Invalid mode accepted without error")
            return False, details
        except ValueError as e:
            details.append(f"✓ Invalid mode rejected: {e}")

    # Check _run_egts_training method exists
    assert hasattr(DreamingDaemon, "_run_egts_training"), "Missing _run_egts_training"
    details.append("✓ _run_egts_training method exists")

    return True, details


# ─────────────────────────────────────────
# T8: Bridge broadcast from generate_cognitive
# ─────────────────────────────────────────

@run_test("T8", "Bridge broadcast — generate_cognitive triggers listeners")
def test_bridge_broadcast() -> Tuple[bool, List[str]]:
    details: List[str] = []

    source_path = PROJECT_ROOT / "metis" / "inference.py"
    source = source_path.read_text()

    # Check that generate_cognitive broadcasts to listeners
    if "self._metis._listeners" in source:
        # Find it specifically in the generate_cognitive context
        gc_start = source.find("def generate_cognitive(")
        if gc_start > 0:
            gc_body = source[gc_start:]
            if "self._metis._listeners" in gc_body and "listener(summary_signal" in gc_body:
                details.append("✓ generate_cognitive broadcasts to listeners")
            else:
                details.append("✗ Listener broadcast not in generate_cognitive")
                return False, details
        else:
            details.append("✗ generate_cognitive not found")
            return False, details
    else:
        details.append("✗ No listener broadcast code found")
        return False, details

    return True, details


# ─────────────────────────────────────────
# T9: Serve module
# ─────────────────────────────────────────

@run_test("T9", "Serve module — create_app importable")
def test_serve() -> Tuple[bool, List[str]]:
    details: List[str] = []

    from metis.serve import create_app, main as serve_main
    details.append("✓ create_app and main importable")

    # Check create_app is callable
    assert callable(create_app), "create_app not callable"
    details.append("✓ create_app is callable")

    return True, details


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main() -> None:
    print("\n" + "="*60)
    print("  Phase 22: Integration Verification")
    print("="*60)

    tests = [
        test_imports,
        test_langchain_stub,
        test_llamaindex_stub,
        test_cli,
        test_online_loop,
        test_trl_adapter,
        test_daemon_egts,
        test_bridge_broadcast,
        test_serve,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        if test_fn():
            passed += 1
        else:
            failed += 1

    # Summary
    print("\n" + "="*60)
    print(f"  RESULTS: {passed}/{passed+failed} PASS, {failed} FAIL")
    print("="*60)

    # Save report
    report = {
        "phase": "22",
        "title": "Integration Verification",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total": passed + failed,
        "passed": passed,
        "failed": failed,
        "tests": results,
    }
    report_path = PROJECT_ROOT / "phase22_integration_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: {report_path}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
