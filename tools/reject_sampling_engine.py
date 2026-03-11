#!/usr/bin/env python3
"""
METIS Phase 2 — High-Entropy Reject Sampling Engine
====================================================

Loads the SFT v4 merged model and generates N independent cognitive
trajectories per prompt using high-entropy sampling parameters.

Architecture:
  1. Fetch OOD prompts: 500 Alpaca (FAST targets) + 500 GSM8K test (DEEP targets)
  2. Apply METIS chat template (system + user) without cognitive tags
  3. Generate N=6 trajectories per prompt with exploration hyperparameters
  4. Export to data/metis_sampled_trajectories.jsonl

VRAM strategy:
  - GB10 has 122GB unified memory; merged 7B bfloat16 ≈ 14GB
  - Dynamic micro-batching: start with batch=8, halve on OOM
  - Each prompt generates N=6 responses sequentially per micro-batch
  - Checkpoint progress to allow resume on crash
"""

import argparse
import gc
import json
import logging
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────

METIS_SYSTEM_PROMPT = (
    "You are METIS, an AI with a dynamic cognitive routing layer. "
    "Analyze the complexity of the user's request and allocate compute accordingly."
)

DEFAULT_MODEL_PATH = "experiment_output_sft_cognitive_v4/metis_sft_cognitive"
DEFAULT_OUTPUT_PATH = "data/metis_sampled_trajectories.jsonl"
DEFAULT_CHECKPOINT_PATH = "data/.reject_sampling_checkpoint.json"

# High-entropy generation hyperparameters
SAMPLING_CONFIG = {
    "temperature": 0.85,
    "top_p": 0.95,
    "do_sample": True,
    "max_new_tokens": 1500,
}

N_TRAJECTORIES = 6
SEED = 42


# ─────────────────────────────────────────────────────
# Data Fetching
# ─────────────────────────────────────────────────────

def fetch_alpaca_prompts(n_samples: int = 500, seed: int = SEED) -> List[Dict[str, str]]:
    """Fetch random prompts from tatsu-lab/alpaca. Raw user messages only."""
    from datasets import load_dataset

    logger.info(f"Fetching {n_samples} Alpaca prompts (FAST targets)...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    # Shuffle deterministically and pick n_samples
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    prompts: List[Dict[str, str]] = []
    for idx in indices:
        ex = ds[idx]
        instruction = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        if not instruction:
            continue

        # Combine instruction + input into a single user message
        user_msg = f"{instruction}\n{inp}".strip() if inp else instruction
        prompts.append({
            "user": user_msg,
            "source": "alpaca",
            "prompt_id": f"alpaca_{len(prompts):04d}",
        })
        if len(prompts) >= n_samples:
            break

    logger.info(f"  Collected {len(prompts)} Alpaca prompts")
    return prompts


def fetch_gsm8k_prompts(n_samples: int = 500, seed: int = SEED) -> List[Dict[str, str]]:
    """Fetch random prompts from openai/gsm8k TEST split. Raw user messages only."""
    from datasets import load_dataset

    logger.info(f"Fetching {n_samples} GSM8K test prompts (DEEP targets)...")
    ds = load_dataset("openai/gsm8k", "main", split="test")

    # Shuffle deterministically and pick n_samples
    indices = list(range(len(ds)))
    rng = random.Random(seed + 1)  # Different seed to avoid correlation
    rng.shuffle(indices)

    prompts: List[Dict[str, str]] = []
    for idx in indices:
        ex = ds[idx]
        question = (ex.get("question") or "").strip()
        if not question:
            continue

        prompts.append({
            "user": question,
            "source": "gsm8k_test",
            "prompt_id": f"gsm8k_{len(prompts):04d}",
        })
        if len(prompts) >= n_samples:
            break

    logger.info(f"  Collected {len(prompts)} GSM8K test prompts")
    return prompts


def build_prompt_set(
    n_fast: int = 500,
    n_deep: int = 500,
    seed: int = SEED,
) -> List[Dict[str, str]]:
    """Build combined OOD prompt set: Alpaca + GSM8K test, shuffled."""
    alpaca = fetch_alpaca_prompts(n_fast, seed)
    gsm8k = fetch_gsm8k_prompts(n_deep, seed)

    combined = alpaca + gsm8k
    rng = random.Random(seed + 2)
    rng.shuffle(combined)

    logger.info(f"Combined prompt set: {len(combined)} prompts "
                f"({len(alpaca)} FAST + {len(gsm8k)} DEEP)")
    return combined


# ─────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_path: str,
) -> tuple:
    """Load merged SFT v4 model in bfloat16.

    VRAM strategy: 7B bfloat16 ≈ 14GB model weights.
    Cap GPU allocation at 50GB to leave ~70GB for KV cache during
    high-entropy generation (max_new_tokens=1500).
    """
    logger.info(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        fix_mistral_regex=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with SDPA attention for efficient KV cache computation
    # No memory cap needed — model is ~14GB bfloat16, plenty of headroom
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    # Log VRAM after loading
    try:
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        logger.info(f"  GPU VRAM after model load: {free_mem / (1024**3):.1f}GB free "
                    f"/ {total_mem / (1024**3):.1f}GB total")
    except Exception:
        pass

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model loaded: {n_params / 1e9:.2f}B params, dtype=bfloat16")
    return model, tokenizer


# ─────────────────────────────────────────────────────
# Chat Template Formatting
# ─────────────────────────────────────────────────────

def format_prompt(tokenizer: AutoTokenizer, user_message: str) -> str:
    """Apply METIS chat template: system + user, with generation prompt."""
    messages = [
        {"role": "system", "content": METIS_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    # Return tokenized IDs for generation
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    return input_ids


def annotate_prompt_lengths(
    tokenizer: AutoTokenizer,
    prompts: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for prompt in prompts:
        prompt_len = len(format_prompt(tokenizer, prompt["user"]))
        enriched.append({**prompt, "_prompt_token_len": prompt_len})
    return enriched


def schedule_prompts(prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    source_rank = {"alpaca": 0, "gsm8k_test": 1}
    return sorted(
        prompts,
        key=lambda p: (source_rank.get(p["source"], 99), p["_prompt_token_len"]),
    )


def select_next_batch(
    prompts: List[Dict[str, Any]],
    start_idx: int,
    batch_size: int,
) -> List[Dict[str, Any]]:
    first = prompts[start_idx]
    source = first["source"]
    min_len = first["_prompt_token_len"]
    length_window = 16 if source == "alpaca" else 24

    batch: List[Dict[str, Any]] = []
    idx = start_idx
    while idx < len(prompts) and len(batch) < batch_size:
        prompt = prompts[idx]
        if prompt["source"] != source:
            break
        if prompt["_prompt_token_len"] - min_len > length_window and batch:
            break
        batch.append(prompt)
        idx += 1

    return batch


# ─────────────────────────────────────────────────────
# Trajectory Generation
# ─────────────────────────────────────────────────────

def generate_trajectories_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batch_prompts: List[Dict[str, str]],
    n_trajectories: int = N_TRAJECTORIES,
) -> List[Dict[str, Any]]:
    """Generate N trajectories for a batch of prompts in one forward pass.

    Batches B prompts with num_return_sequences=N → B*N output sequences.
    Left-pads inputs for batched generation. With ~30GB used for 1 prompt,
    batching 4 prompts should use ~90GB of the 121GB available.
    """
    pad_id = tokenizer.pad_token_id
    B = len(batch_prompts)

    # Tokenize all prompts
    all_ids = [format_prompt(tokenizer, p["user"]) for p in batch_prompts]
    prompt_lens = [len(ids) for ids in all_ids]
    max_len = max(prompt_lens)

    # Left-pad to same length
    padded_ids = []
    attention_masks = []
    for ids in all_ids:
        pad_len = max_len - len(ids)
        padded_ids.append([pad_id] * pad_len + ids)
        attention_masks.append([0] * pad_len + [1] * len(ids))

    input_tensor = torch.tensor(padded_ids, dtype=torch.long, device=model.device)
    attn_tensor = torch.tensor(attention_masks, dtype=torch.long, device=model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_tensor,
            attention_mask=attn_tensor,
            temperature=SAMPLING_CONFIG["temperature"],
            top_p=SAMPLING_CONFIG["top_p"],
            do_sample=SAMPLING_CONFIG["do_sample"],
            max_new_tokens=SAMPLING_CONFIG["max_new_tokens"],
            pad_token_id=pad_id,
            num_return_sequences=n_trajectories,
        )

    # outputs shape: [B * n_trajectories, max_len + generated_len]
    # HF order: prompt_0_traj_0, prompt_0_traj_1, ..., prompt_1_traj_0, ...
    # Move to CPU immediately to free GPU KV cache memory
    outputs_cpu = outputs.cpu()
    del outputs, input_tensor, attn_tensor
    gc.collect()
    torch.cuda.empty_cache()

    results: List[Dict[str, Any]] = []
    for b_idx, p in enumerate(batch_prompts):
        result: Dict[str, Any] = {
            "prompt_id": p["prompt_id"],
            "system": METIS_SYSTEM_PROMPT,
            "user": p["user"],
            "source": p["source"],
            "trajectories": [],
        }
        for traj_idx in range(n_trajectories):
            out_idx = b_idx * n_trajectories + traj_idx
            generated_ids = outputs_cpu[out_idx][max_len:]  # Skip padded prompt
            text = tokenizer.decode(generated_ids, skip_special_tokens=False)

            # Clean trailing pad/eos tokens
            if tokenizer.eos_token:
                text = text.split(tokenizer.eos_token)[0]
            if tokenizer.pad_token and tokenizer.pad_token != tokenizer.eos_token:
                text = text.replace(tokenizer.pad_token, "")

            result["trajectories"].append({
                "text": text.strip(),
                "index": traj_idx,
            })
        results.append(result)

    del outputs_cpu
    return results


def run_generation_loop(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[Dict[str, str]],
    n_trajectories: int = N_TRAJECTORIES,
    initial_batch_size: int = 4,
    fast_batch_size: Optional[int] = None,
    deep_batch_size: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Generate trajectories with adaptive batching and checkpoint resume.

    VRAM budget: ~30GB for 1 prompt × 6 traj → batch 4 prompts to use ~90GB.
    Adaptive: halves batch on OOM, scales up after consecutive successes.
    """
    # ── Resume from checkpoint if available ──
    completed_ids: set = set()
    all_results: List[Dict[str, Any]] = []
    output_results: Dict[str, Dict[str, Any]] = {}

    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        with open(checkpoint_path, "r") as f:
            ckpt = json.load(f)
        completed_ids = set(ckpt.get("completed_ids", []))
        logger.info(f"  {len(completed_ids)} prompts already completed")

    # Also reload existing output file to avoid re-writing
    if output_path and Path(output_path).exists():
        raw_output_rows = 0
        with open(output_path, "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    raw_output_rows += 1
                    output_results[item["prompt_id"]] = item

        if raw_output_rows != len(output_results):
            logger.info(f"Deduplicating output file: {raw_output_rows} rows -> "
                        f"{len(output_results)} unique prompts")
            with open(output_path, "w") as f:
                for item in output_results.values():
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        output_ids = set(output_results.keys())
        if output_ids - completed_ids:
            logger.info(f"  Found {len(output_ids - completed_ids)} prompts in output file "
                        f"ahead of checkpoint; marking them completed")
        completed_ids |= output_ids
        all_results.extend(output_results.values())

    # Filter out already-completed prompts
    remaining = [p for p in prompts if p["prompt_id"] not in completed_ids]
    remaining = annotate_prompt_lengths(tokenizer, remaining)
    remaining = schedule_prompts(remaining)
    total = len(remaining)
    fast_batch_size = fast_batch_size or max(initial_batch_size * 2, initial_batch_size)
    deep_batch_size = deep_batch_size or initial_batch_size
    logger.info(f"Prompts to process: {total} "
                f"(skipped {len(completed_ids)} from checkpoint)")
    logger.info(f"Batch policy: alpaca={fast_batch_size}, gsm8k_test={deep_batch_size}")

    idx = 0
    start_time = time.time()

    while idx < total:
        current_source = remaining[idx]["source"]
        current_batch_size = fast_batch_size if current_source == "alpaca" else deep_batch_size
        batch = select_next_batch(remaining, idx, current_batch_size)
        n_seqs = len(batch) * n_trajectories

        try:
            batch_start = time.time()
            logger.info(f"Batch: prompts {idx+1}-{idx+len(batch)}/{total} | "
                        f"src={current_source} | bs={len(batch)} | "
                        f"tok={batch[0]['_prompt_token_len']}-{batch[-1]['_prompt_token_len']} | "
                        f"{n_seqs} parallel seqs")

            batch_results = generate_trajectories_batch(
                model, tokenizer, batch, n_trajectories,
            )
            batch_elapsed = time.time() - batch_start

            # Accumulate results
            for r in batch_results:
                all_results.append(r)
                completed_ids.add(r["prompt_id"])

            # ── Append to output file (incremental) ──
            if output_path:
                with open(output_path, "a") as f:
                    for r in batch_results:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

            # ── Save checkpoint ──
            if checkpoint_path:
                with open(checkpoint_path, "w") as f:
                    json.dump({"completed_ids": list(completed_ids)}, f)

            # ── Progress report ──
            elapsed = time.time() - start_time
            done = len(completed_ids)
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(prompts) - done) / rate if rate > 0 else 0

            # Log VRAM usage periodically
            vram_info = ""
            try:
                free_mem, total_mem = torch.cuda.mem_get_info(0)
                used = (total_mem - free_mem) / (1024**3)
                vram_info = f" | VRAM: {used:.0f}GB"
            except Exception:
                pass

            logger.info(
                f"  Done {batch_elapsed:.1f}s | "
                f"{done}/{len(prompts)} total | "
                f"{rate:.2f}/s | "
                f"ETA: {eta/60:.1f}min{vram_info}"
            )

            idx += len(batch)

            # Clear KV cache after every batch to prevent VRAM fragmentation
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            next_batch_size = max(1, current_batch_size // 2)
            logger.warning(f"OOM with src={current_source} bs={current_batch_size}! "
                           f"Halving to {next_batch_size}")
            torch.cuda.empty_cache()
            if current_source == "alpaca":
                fast_batch_size = next_batch_size
            else:
                deep_batch_size = next_batch_size
            # Don't advance idx — retry with smaller batch

    # Final checkpoint
    if checkpoint_path:
        with open(checkpoint_path, "w") as f:
            json.dump({"completed_ids": list(completed_ids)}, f)

    total_time = time.time() - start_time
    logger.info(f"Generation complete: {len(all_results)} prompts, "
                f"{len(all_results) * n_trajectories} trajectories, "
                f"{total_time/60:.1f}min total")

    return all_results


# ─────────────────────────────────────────────────────
# Validation & Stats
# ─────────────────────────────────────────────────────

def validate_and_report(output_path: str, n_trajectories: int = N_TRAJECTORIES) -> None:
    """Validate output file and print statistics."""
    logger.info(f"Validating output: {output_path}")

    n_prompts = 0
    n_fast_tagged = 0
    n_deep_tagged = 0
    n_none_tagged = 0
    sources: Dict[str, int] = {}
    traj_lengths: List[int] = []

    with open(output_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            n_prompts += 1

            src = item.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1

            if len(item["trajectories"]) != n_trajectories:
                logger.warning(f"  {item['prompt_id']}: expected {n_trajectories} "
                               f"trajectories, got {len(item['trajectories'])}")

            for traj in item["trajectories"]:
                text = traj["text"]
                traj_lengths.append(len(text))
                if "[COGNITIVE_STATE: FAST]" in text:
                    n_fast_tagged += 1
                elif "[COGNITIVE_STATE: DEEP]" in text:
                    n_deep_tagged += 1
                else:
                    n_none_tagged += 1

    total_traj = n_prompts * n_trajectories
    logger.info(f"  Prompts: {n_prompts}")
    logger.info(f"  Sources: {sources}")
    logger.info(f"  Total trajectories: {total_traj}")
    logger.info(f"  FAST tagged: {n_fast_tagged} ({n_fast_tagged/total_traj*100:.1f}%)")
    logger.info(f"  DEEP tagged: {n_deep_tagged} ({n_deep_tagged/total_traj*100:.1f}%)")
    logger.info(f"  NONE tagged: {n_none_tagged} ({n_none_tagged/total_traj*100:.1f}%)")
    if traj_lengths:
        avg_len = sum(traj_lengths) / len(traj_lengths)
        logger.info(f"  Avg trajectory length: {avg_len:.0f} chars "
                    f"(min={min(traj_lengths)}, max={max(traj_lengths)})")


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="METIS Phase 2: High-Entropy Reject Sampling Engine"
    )
    parser.add_argument(
        "--model-path", type=str, default=DEFAULT_MODEL_PATH,
        help="Path to merged SFT model",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT_PATH,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--n-fast", type=int, default=500,
        help="Number of Alpaca prompts (FAST targets)",
    )
    parser.add_argument(
        "--n-deep", type=int, default=500,
        help="Number of GSM8K test prompts (DEEP targets)",
    )
    parser.add_argument(
        "--n-trajectories", type=int, default=N_TRAJECTORIES,
        help="Number of trajectories per prompt",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Initial micro-batch size (auto-halves on OOM)",
    )
    parser.add_argument(
        "--fast-batch-size", type=int, default=None,
        help="Batch size for Alpaca/FAST prompts (defaults to 2x --batch-size)",
    )
    parser.add_argument(
        "--deep-batch-size", type=int, default=None,
        help="Batch size for GSM8K/DEEP prompts (defaults to --batch-size)",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint if available",
    )
    args = parser.parse_args()

    # ── Banner ──
    logger.info("=" * 60)
    logger.info("METIS Phase 2: High-Entropy Reject Sampling Engine")
    logger.info("=" * 60)
    logger.info(f"  Model:          {args.model_path}")
    logger.info(f"  Output:         {args.output}")
    logger.info(f"  Prompts:        {args.n_fast} FAST + {args.n_deep} DEEP = "
                f"{args.n_fast + args.n_deep}")
    logger.info(f"  Trajectories:   {args.n_trajectories} per prompt")
    logger.info(f"  Total gen:      {(args.n_fast + args.n_deep) * args.n_trajectories}")
    logger.info(f"  Batch size:     {args.batch_size} (adaptive)")
    logger.info(f"  FAST batch:     {args.fast_batch_size or args.batch_size * 2}")
    logger.info(f"  DEEP batch:     {args.deep_batch_size or args.batch_size}")
    logger.info(f"  Sampling:       temp={SAMPLING_CONFIG['temperature']}, "
                f"top_p={SAMPLING_CONFIG['top_p']}, "
                f"max_new_tokens={SAMPLING_CONFIG['max_new_tokens']}")
    logger.info(f"  Seed:           {args.seed}")

    # ── Step 1: Build prompt set ──
    random.seed(args.seed)
    prompts = build_prompt_set(args.n_fast, args.n_deep, args.seed)

    # ── Step 2: Load model ──
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    # ── Step 3: Prepare output ──
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    checkpoint_path = DEFAULT_CHECKPOINT_PATH

    # Clear output file if not resuming
    if not args.resume:
        if Path(args.output).exists():
            Path(args.output).unlink()
        if Path(checkpoint_path).exists():
            Path(checkpoint_path).unlink()

    # ── Step 4: Generate trajectories ──
    logger.info("Starting high-entropy trajectory generation...")
    results = run_generation_loop(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        n_trajectories=args.n_trajectories,
        initial_batch_size=args.batch_size,
        fast_batch_size=args.fast_batch_size,
        deep_batch_size=args.deep_batch_size,
        checkpoint_path=checkpoint_path,
        output_path=args.output,
    )

    # ── Step 5: Validate ──
    validate_and_report(args.output, args.n_trajectories)

    # ── Cleanup checkpoint on success ──
    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
        logger.info("Checkpoint cleaned up (run completed successfully)")

    logger.info("=" * 60)
    logger.info("REJECT SAMPLING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
