import json

d = json.load(open("experiment_output/experiment_report.json"))

print("=" * 60)
print("EXPERIMENT RESULTS (v11)")
print("=" * 60)

for name in ["base", "metis_dpo", "random_dpo"]:
    m = d[name]
    print(f"\n--- {m['name']} ---")
    print(f"  reward_total:    {m['reward_total']:.4f}")
    print(f"  reward_coherence:{m['reward_coherence']:.4f}")
    print(f"  reward_calibration:{m['reward_calibration']:.4f}")
    print(f"  reward_phase:    {m['reward_phase_quality']:.4f}")
    print(f"  reward_epistemic:{m['reward_epistemic']:.4f}")
    print(f"  reward_efficiency:{m['reward_efficiency']:.4f}")
    print(f"  mean_entropy:    {m['mean_entropy']:.3f}")
    print(f"  mean_surprise:   {m['mean_surprise']:.3f}")
    print(f"  confusion_ratio: {m['confusion_ratio']:.4f}")
    print(f"  fast_ratio:      {m['fast_ratio']:.3f}")
    print(f"  avg_tokens:      {m['avg_tokens']:.1f}")

print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
for k, v in d["summary"].items():
    print(f"  {k}: {v:+.4f}")

# Statistical quick check
base_r = d["base"]["per_prompt_rewards"]
metis_r = d["metis_dpo"]["per_prompt_rewards"]
random_r = d["random_dpo"]["per_prompt_rewards"]
n = len(base_r)

import math
def mean_std(arr):
    m = sum(arr) / len(arr)
    v = sum((x - m) ** 2 for x in arr) / (len(arr) - 1)
    return m, math.sqrt(v)

def cohens_d(a, b):
    ma, sa = mean_std(a)
    mb, sb = mean_std(b)
    pooled = math.sqrt((sa**2 + sb**2) / 2)
    return (ma - mb) / pooled if pooled > 0 else 0

def ci95(a, b):
    diffs = [x - y for x, y in zip(a, b)]
    md, sd = mean_std(diffs)
    se = sd / math.sqrt(len(diffs))
    return md - 1.96 * se, md + 1.96 * se

d_val = cohens_d(metis_r, random_r)
lo, hi = ci95(metis_r, random_r)
print(f"\n  METIS vs Random: Cohen's d = {d_val:.3f}")
print(f"  95% CI: [{lo:+.4f}, {hi:+.4f}]")
if lo > 0:
    print("  --> METIS significantly better")
elif hi < 0:
    print("  --> Random significantly better")
else:
    print("  --> NOT statistically significant (CI includes 0)")
