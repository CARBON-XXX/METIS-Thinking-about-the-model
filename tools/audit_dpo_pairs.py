"""Audit DPO preference pairs from Phase 1 scored data."""
import json
import sys
from collections import defaultdict
from pathlib import Path

data_path = Path("experiment_output/phase1_scored_data.json")
data = json.loads(data_path.read_text(encoding="utf-8"))

# Group by prompt
groups: dict[str, list] = defaultdict(list)
for s in data:
    groups[s["prompt"]].append(s)

print(f"=== DPO Pair Audit: {len(groups)} prompts, {len(data)} samples ===\n")

# Stats
margins = []
label_inversions = 0

for i, (prompt, samples) in enumerate(groups.items()):
    rewards = sorted(samples, key=lambda x: x["reward_total"], reverse=True)
    best = rewards[0]
    worst = rewards[-1]
    gap = best["reward_total"] - worst["reward_total"]
    margins.append(gap)

    bb = best["reward_breakdown"]
    wb = worst["reward_breakdown"]

    print(f"[{i+1}/{len(groups)}] {prompt[:70]}...")
    print(f"  CHOSEN  (idx={best['sample_idx']}): R={best['reward_total']:.4f}  tok={best['trace_stats']['total_tokens']}")
    print(f"    coherence={bb['coherence']:.3f} calib={bb['calibration']:.3f} phase={bb['phase_quality']:.3f} epist={bb['epistemic_honesty']:.3f} eff={bb['efficiency']:.3f} comp_bonus={bb.get('completeness_bonus',0):.3f}")
    print(f"    resp: {best['response'][:120]}...")
    print(f"  REJECTED(idx={worst['sample_idx']}): R={worst['reward_total']:.4f}  tok={worst['trace_stats']['total_tokens']}")
    print(f"    coherence={wb['coherence']:.3f} calib={wb['calibration']:.3f} phase={wb['phase_quality']:.3f} epist={wb['epistemic_honesty']:.3f} eff={wb['efficiency']:.3f} comp_bonus={wb.get('completeness_bonus',0):.3f}")
    print(f"    resp: {worst['response'][:120]}...")
    print(f"  Margin: {gap:.4f}")

    # Check if shorter response is chosen (potential gaming)
    if best["trace_stats"]["total_tokens"] < worst["trace_stats"]["total_tokens"] - 5:
        print(f"  ⚠️  CHOSEN is SHORTER ({best['trace_stats']['total_tokens']} < {worst['trace_stats']['total_tokens']})")

    # Check reward component dominance
    eff_diff = bb["efficiency"] - wb["efficiency"]
    calib_diff = bb["calibration"] - wb["calibration"]
    if abs(eff_diff) > gap * 0.5:
        print(f"  ⚠️  Efficiency dominates margin: eff_diff={eff_diff:.3f} vs total_gap={gap:.3f}")

    print()

print("=== Summary ===")
print(f"Mean margin:   {sum(margins)/len(margins):.4f}")
print(f"Min margin:    {min(margins):.4f}")
print(f"Max margin:    {max(margins):.4f}")
print(f"Margins < 0.1: {sum(1 for m in margins if m < 0.1)}/{len(margins)}")
print(f"Margins < 0.05:{sum(1 for m in margins if m < 0.05)}/{len(margins)}")
