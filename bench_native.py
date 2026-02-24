"""
METIS Native Acceleration Benchmark
Rust (PyO3) vs Pure Python — per-component latency comparison.
"""
import sys
import time
import statistics

sys.path.insert(0, "G:/SEDACV9.0 PRO")

# ── Direct Rust imports ──
from metis_native import (
    AdaptiveControllerNative,
    BoundaryGuardNative,
    CotCusumNative,
    SlidingWindowStats as NativeStats,
    cornish_fisher_quantile,
)

# ── Pure Python imports ──
from metis.core.statistics import _PySlidingWindowStats
from metis.core.controller import AdaptiveController
from metis.core.types import ControllerConfig

N_ITERS = 50_000
WARMUP = 1_000


def bench(label: str, fn, n: int = N_ITERS) -> float:
    """Run fn() n times, return median μs/call."""
    # Warmup
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        elapsed = time.perf_counter() - t0
        times.append(elapsed / n * 1e6)  # μs/call
    med = statistics.median(times)
    print(f"  {label:40s} {med:8.2f} μs/call")
    return med


def main() -> None:
    print("=" * 60)
    print("METIS Rust Native Acceleration Benchmark")
    print(f"  iterations = {N_ITERS:,}, warmup = {WARMUP:,}")
    print("=" * 60)

    # ── 1. SlidingWindowStats ──
    print("\n── SlidingWindowStats (update + get_stats) ──")
    py_sw = _PySlidingWindowStats(500)
    rs_sw = NativeStats(500)
    for i in range(200):
        py_sw.update(float(i) * 0.01)
        rs_sw.update(float(i) * 0.01)

    step = [0]

    def py_sw_step():
        step[0] += 1
        py_sw.update(float(step[0]) * 0.01)
        py_sw.get_stats()

    step2 = [0]

    def rs_sw_step():
        step2[0] += 1
        rs_sw.update(float(step2[0]) * 0.01)
        rs_sw.get_stats()

    t_py = bench("Python", py_sw_step)
    t_rs = bench("Rust  ", rs_sw_step)
    print(f"  → Speedup: {t_py / t_rs:.1f}x")

    # ── 2. BoundaryGuard (CUSUM evaluate) ──
    print("\n── BoundaryGuard (CUSUM evaluate) ──")
    rs_bg = BoundaryGuardNative()
    # Warm it past cold start
    for _ in range(10):
        rs_bg.evaluate(1.0, 0.5, 0.5)

    def rs_bg_step():
        rs_bg.evaluate(1.2, 0.4, 0.6)

    # Python boundary guard (no native)
    from metis.cognitive.boundary import (
        EpistemicBoundaryGuard,
        CUSUM_K,
        CUSUM_HEDGE_H,
        CUSUM_REFUSE_H,
        CUSUM_DECAY,
        SURPRISE_BASELINE,
        SURPRISE_WEIGHT,
        CONFIDENCE_REFUSE,
        CONFIDENCE_SEEK,
        CONFIDENCE_KNOWN,
    )
    from metis.core.types import CognitiveSignal, Decision

    py_bg = EpistemicBoundaryGuard()
    py_bg._native = None  # Force Python path
    sig = CognitiveSignal(
        semantic_entropy=1.5,
        token_entropy=1.3,
        semantic_diversity=0.6,
        confidence=0.4,
        z_score=1.2,
        decision=Decision.NORMAL,
        entropy_trend="stable",
        cognitive_phase="reasoning",
        entropy_momentum=0.05,
    )
    for _ in range(10):
        py_bg.evaluate(sig)

    def py_bg_step():
        py_bg.evaluate(sig)

    t_py = bench("Python", py_bg_step)
    t_rs = bench("Rust  ", rs_bg_step)
    print(f"  → Speedup: {t_py / t_rs:.1f}x")

    # ── 3. AdaptiveController (update + decide) ──
    print("\n── AdaptiveController (update + decide) ──")

    py_ctrl = AdaptiveController()
    py_ctrl._native = None  # Force Python path
    for i in range(20):
        py_ctrl.update(1.0 + i * 0.05, 0.7)

    rs_ctrl = AdaptiveControllerNative()
    for i in range(20):
        rs_ctrl.update(1.0 + i * 0.05, 0.7)

    cnt = [0]

    def py_ctrl_step():
        cnt[0] += 1
        e = 1.0 + (cnt[0] % 50) * 0.03
        py_ctrl.update(e, 0.6)
        py_ctrl.decide(e, 0.6)

    cnt2 = [0]

    def rs_ctrl_step():
        cnt2[0] += 1
        e = 1.0 + (cnt2[0] % 50) * 0.03
        rs_ctrl.update(e, 0.6)
        rs_ctrl.decide(e, 0.6)

    t_py = bench("Python", py_ctrl_step)
    t_rs = bench("Rust  ", rs_ctrl_step)
    print(f"  → Speedup: {t_py / t_rs:.1f}x")

    # ── 4. CoT CUSUM (observe + should_inject) ──
    print("\n── CoT CUSUM (observe + should_inject) ──")
    rs_cot = CotCusumNative()

    def rs_cot_step():
        rs_cot.observe(0.8, 0.5, 1, 0.02)
        rs_cot.should_inject()

    from metis.cognitive.cot import CoTManager

    py_cot = CoTManager()
    py_cot._native = None  # Force Python path

    def py_cot_step():
        py_cot.observe(sig)
        py_cot.should_inject()

    t_py = bench("Python", py_cot_step)
    t_rs = bench("Rust  ", rs_cot_step)
    print(f"  → Speedup: {t_py / t_rs:.1f}x")

    # ── 5. Cornish-Fisher ──
    print("\n── Cornish-Fisher quantile ──")

    def py_cf():
        z = 1.96
        s, k = 0.5, 0.3
        return z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24

    def rs_cf():
        return cornish_fisher_quantile(1.96, 0.5, 0.3)

    t_py = bench("Python", py_cf)
    t_rs = bench("Rust  ", rs_cf)
    print(f"  → Speedup: {t_py / t_rs:.1f}x")

    print("\n" + "=" * 60)
    print("Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
