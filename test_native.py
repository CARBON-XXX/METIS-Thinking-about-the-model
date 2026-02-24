"""Quick smoke test for Rust native acceleration integration."""
import sys
sys.path.insert(0, "G:/SEDACV9.0 PRO")

from metis.core.controller import AdaptiveController, _HAS_NATIVE
from metis.cognitive.boundary import EpistemicBoundaryGuard, _HAS_NATIVE as BN
from metis.cognitive.cot import CoTManager, _HAS_NATIVE as CN

print(f"Controller native={_HAS_NATIVE}")
print(f"Boundary  native={BN}")
print(f"CoT       native={CN}")

# --- AdaptiveController ---
ctrl = AdaptiveController()
for i in range(15):
    ctrl.update(1.0 + i * 0.1, 0.8 - i * 0.02)
d = ctrl.decide(2.5, 0.3)
print(f"\nController decide={d}")
s = ctrl.stats
print(f"  mean={s['entropy_mean']:.3f}, std={s['entropy_std']:.3f}, calibrated={s['is_calibrated']}")
z = ctrl.get_z_score(2.5)
print(f"  z_score={z:.3f}")
g, m = ctrl.get_predictive_signals()
print(f"  gradient={g:.3f}, momentum={m:.3f}")
zt = ctrl.get_dynamic_z_thresholds()
print(f"  dynamic_z={zt}")

# --- EpistemicBoundaryGuard ---
from metis.core.types import CognitiveSignal, Decision
bg = EpistemicBoundaryGuard()
sig = CognitiveSignal(
    semantic_entropy=2.0, token_entropy=1.8, semantic_diversity=0.7,
    confidence=0.3, z_score=1.5, decision=Decision.DEEP,
    entropy_trend="rising", cognitive_phase="exploration",
    entropy_momentum=0.1,
)
for _ in range(6):
    state, action, expl = bg.evaluate(sig)
print(f"\nBoundary: state={state}, action={action}, expl='{expl}'")
print(f"  uncertainty_score={bg.get_uncertainty_score():.3f}")

# --- CoTManager ---
cot = CoTManager()
for _ in range(10):
    cot.observe(sig)
print(f"\nCoT should_inject={cot.should_inject()}")
print(f"  stats={cot.stats}")

print("\n=== ALL INTEGRATION TESTS PASSED ===")
