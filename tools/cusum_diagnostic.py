"""Diagnostic: generate a few samples and dump per-token cognitive events
for CUSUM parameter grid search."""
import sys
import json
import torch
from pathlib import Path
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from metis.training.generator import MetisGenerator

PROMPTS = [
    "Explain quantum entanglement in simple terms.",
    "What is the standard model of particle physics?",
    "How does nuclear fusion produce energy?",
    "What is the significance of the Higgs boson?",
    "Explain the Heisenberg uncertainty principle.",
]

def main() -> None:
    print("Loading model...")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    gen = MetisGenerator(model, tokenizer)

    all_z_scores: list[float] = []
    all_events: list[dict] = []
    phase_counts: Counter = Counter()
    decision_counts: Counter = Counter()

    for i, prompt in enumerate(PROMPTS):
        print(f"\n[{i+1}/{len(PROMPTS)}] {prompt}")
        chat = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        response, trace = gen.generate(text, max_new_tokens=200, temperature=0.7)

        print(f"  Tokens: {len(trace.events)}")
        print(f"  Response: {response[:80]}...")
        print(f"  --- Per-token events ---")

        for j, ev in enumerate(trace.events):
            z = ev.z_score
            all_z_scores.append(z)
            phase_counts[ev.cognitive_phase] += 1
            decision_counts[ev.decision.name] += 1

            all_events.append({
                "prompt_idx": i,
                "step": ev.step,
                "z_score": round(z, 4),
                "entropy": round(ev.token_entropy, 4),
                "sem_entropy": round(ev.semantic_entropy, 4),
                "confidence": round(ev.confidence, 4),
                "surprise": round(ev.token_surprise, 4),
                "decision": ev.decision.name,
                "phase": ev.cognitive_phase,
                "boundary": ev.boundary_action.name,
                "cusum_alarm": ev.cusum_alarm,
                "momentum": round(ev.entropy_momentum, 4),
            })

            flag = ""
            if ev.cognitive_phase == "confusion":
                flag = " <<<CONFUSION"
            elif ev.cusum_alarm:
                flag = " <<<CUSUM_ALARM"
            print(f"    step={ev.step:2d} z={z:+.3f} H={ev.token_entropy:.3f} "
                  f"dec={ev.decision.name:6s} phase={ev.cognitive_phase:12s} "
                  f"bound={ev.boundary_action.name:8s} mom={ev.entropy_momentum:+.3f}"
                  f"{flag}")

    # Summary
    print("\n" + "=" * 60)
    print("=== Z-SCORE DISTRIBUTION ===")
    if all_z_scores:
        import statistics
        zs = sorted(all_z_scores)
        print(f"  Count:  {len(zs)}")
        print(f"  Mean:   {statistics.mean(zs):.4f}")
        print(f"  Median: {statistics.median(zs):.4f}")
        print(f"  Std:    {statistics.stdev(zs):.4f}")
        print(f"  Min:    {zs[0]:.4f}")
        print(f"  Max:    {zs[-1]:.4f}")
        # Percentiles
        for p in [10, 25, 50, 75, 90, 95, 99]:
            idx = int(len(zs) * p / 100)
            print(f"  P{p:02d}:    {zs[min(idx, len(zs)-1)]:.4f}")

        # How many exceed CONFUSION threshold (w_z > 0.5)?
        above_05 = sum(1 for z in zs if z > 0.5)
        above_03 = sum(1 for z in zs if z > 0.3)
        print(f"\n  z > 0.5: {above_05}/{len(zs)} ({100*above_05/len(zs):.1f}%)")
        print(f"  z > 0.3: {above_03}/{len(zs)} ({100*above_03/len(zs):.1f}%)")

    print(f"\n=== PHASE DISTRIBUTION ===")
    for phase, cnt in phase_counts.most_common():
        print(f"  {phase:12s}: {cnt:3d} ({100*cnt/len(all_events):.1f}%)")

    print(f"\n=== DECISION DISTRIBUTION ===")
    for dec, cnt in decision_counts.most_common():
        print(f"  {dec:6s}: {cnt:3d} ({100*cnt/len(all_events):.1f}%)")

    # Save raw events for offline replay
    out_path = Path("experiment_output/diagnostic_events.json")
    out_path.write_text(json.dumps(all_events, indent=2), encoding="utf-8")
    print(f"\nSaved {len(all_events)} events to {out_path}")


if __name__ == "__main__":
    main()
