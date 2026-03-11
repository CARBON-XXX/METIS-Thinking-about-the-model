#!/usr/bin/env python3
"""
vLLM Offline Batch Generation — runs inside WSL2.

Generates all training samples using vLLM's offline LLM class (no HTTP server).
Saves results as JSON for Windows-side teacher-forcing + METIS analysis.

Usage (from Windows):
    wsl -e bash -c "source ~/miniconda3/etc/profile.d/conda.sh && \
        conda activate sedac_dev && \
        cd /mnt/g/SEDACV9.0\ PRO && \
        python vllm_batch_generate.py --output experiment_vllm"
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm

# Disable torch dynamo to prevent ptxas sm_121a compilation errors on this environment
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from vllm import LLM, SamplingParams


# ═══════════════════════════════════════════════════════
# Training Prompts
# ═══════════════════════════════════════════════════════

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from metis.pipeline.config import TRAIN_PROMPTS

def main():
    parser = argparse.ArgumentParser(description="vLLM Offline Batch Generation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--n-prompts", type=int, default=300)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output", type=str, default="experiment_vllm")
    parser.add_argument("--gpu-mem", type=float, default=0.70, help="GPU memory utilization (0 to 1)")
    args = parser.parse_args()

    prompts = TRAIN_PROMPTS[:args.n_prompts]
    print(f"\n{'='*60}")
    print(f"vLLM Offline Batch Generation")
    print(f"  Model:    {args.model}")
    print(f"  Prompts:  {len(prompts)}")
    print(f"  Samples:  {args.n_samples} per prompt")
    print(f"  Total:    {len(prompts) * args.n_samples} generations")
    print(f"{'='*60}\n")

    # Initialize vLLM (offline mode — no HTTP server)
    print("Loading vLLM model...")
    t0 = time.time()
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="float16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=1024,
        enforce_eager=True,  # Disable CUDA graphs — fixes WSL2 hang
    )
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # Build all requests: each prompt × n_samples with different temperatures
    all_requests: List[Dict[str, Any]] = []
    all_conversations = []
    base_t = 0.7
    spread = 0.15

    for i, prompt in enumerate(prompts):
        for j in range(args.n_samples):
            temp = max(0.1, base_t + (j - args.n_samples / 2) * spread / args.n_samples)
            all_requests.append({
                "prompt_idx": i,
                "sample_idx": j,
                "prompt": prompt,
                "temperature": temp,
            })
            # Chat format for vLLM
            all_conversations.append([{"role": "user", "content": prompt}])

    print(f"Built {len(all_requests)} requests")

    # Generate ALL samples in one batch call
    # vLLM handles continuous batching internally for maximum throughput
    print("Starting batch generation...")
    t1 = time.time()

    # Group by temperature for efficient batching
    # (vLLM can handle different sampling params per request via generate())
    temp_groups: Dict[float, List[int]] = {}
    for idx, req in enumerate(all_requests):
        t = round(req["temperature"], 3)
        if t not in temp_groups:
            temp_groups[t] = []
        temp_groups[t].append(idx)

    # Process each temperature group
    outputs_map: Dict[int, Any] = {}
    for temp, indices in temp_groups.items():
        conversations_batch = [all_conversations[i] for i in indices]
        sampling = SamplingParams(
            temperature=temp,
            top_p=0.9,
            max_tokens=args.max_tokens,
            repetition_penalty=1.1,
            logprobs=5,
        )
        results = llm.chat(conversations_batch, sampling_params=sampling)
        for local_idx, global_idx in enumerate(indices):
            outputs_map[global_idx] = results[local_idx]

        print(f"  temp={temp:.3f}: {len(indices)} samples done")

    gen_time = time.time() - t1
    print(f"\nGeneration complete: {gen_time:.0f}s ({gen_time/60:.1f}m)")
    print(f"Throughput: {len(all_requests)/gen_time:.1f} samples/sec")

    # Package results as JSON
    results_data: List[Dict[str, Any]] = []
    for idx in range(len(all_requests)):
        req = all_requests[idx]
        output = outputs_map[idx]
        text = output.outputs[0].text

        results_data.append({
            "prompt_idx": req["prompt_idx"],
            "sample_idx": req["sample_idx"],
            "prompt": req["prompt"],
            "temperature": req["temperature"],
            "text": text,
            "finish_reason": output.outputs[0].finish_reason,
            "num_tokens": len(output.outputs[0].token_ids),
        })

    # Save to shared filesystem (accessible from Windows)
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "vllm_raw_samples.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results_data)} samples to {out_path}")
    print(f"Total time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}m)")
    print(f"\nNext step (from Windows):")
    print(f"  python run_experiment.py --phase generate --vllm-data {args.output}/vllm_raw_samples.json")


if __name__ == "__main__":
    main()
