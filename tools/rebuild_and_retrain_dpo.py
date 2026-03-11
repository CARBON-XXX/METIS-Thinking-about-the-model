#!/usr/bin/env python3
"""
METIS Phase 6 — DPO Dataset Re-balancing & KL-Anchored Re-training
====================================================================

Fixes the Phase 4 "Policy Collapse" where FAST routing was suppressed
because too many FAST trajectories appeared in the rejected pool.

PART 1: Extract perfectly balanced 800 DPO pairs (400 FAST chosen + 400 DEEP chosen)
PART 2: Re-train DPO from clean SFT v4 base with β=0.2, lr=2e-6

Key constraints:
  - FAST chosen: starts with [COGNITIVE_STATE: FAST], NO <thinking> tags
  - DEEP chosen: starts with [COGNITIVE_STATE: DEEP], HAS properly closed </thinking>
  - FAST rejected: wrong route (DEEP/NONE on simple task)
  - DEEP rejected: wrong route (FAST/NONE) OR broken <thinking> format
"""

import argparse
import gc
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────

TRAJECTORIES_PATH = "data/metis_sampled_trajectories.jsonl"
BALANCED_OUTPUT_PATH = "data/metis_dpo_pairs_balanced.jsonl"
SFT_MODEL_PATH = "experiment_output_sft_cognitive_v4/metis_sft_cognitive"
DPO_OUTPUT_DIR = "experiment_output_dpo_balanced"
DPO_MERGED_PATH = "experiment_output_dpo_balanced/metis_dpo_cognitive"

METIS_SYSTEM_PROMPT = (
    "You are METIS, an AI with a dynamic cognitive routing layer. "
    "Analyze the complexity of the user's request and allocate compute accordingly."
)

TAG_FAST = "FAST"
TAG_DEEP = "DEEP"
TAG_NONE = "NONE"

_COGNITIVE_TAG_RE = re.compile(r"\[COGNITIVE_STATE:\s*(FAST|DEEP)\]", re.IGNORECASE)
_THINKING_OPEN_RE = re.compile(r"<thinking>", re.IGNORECASE)
_THINKING_CLOSE_RE = re.compile(r"</thinking>", re.IGNORECASE)

TARGET_FAST_PAIRS = 400
TARGET_DEEP_PAIRS = 400


# ═════════════════════════════════════════════════════
# PART 1: DATASET RE-BALANCING
# ═════════════════════════════════════════════════════

def extract_tag(text: str) -> str:
    m = _COGNITIVE_TAG_RE.search(text)
    return m.group(1).upper() if m else TAG_NONE


def starts_with_tag(text: str, tag: str) -> bool:
    return text.strip().startswith(f"[COGNITIVE_STATE: {tag}]")


def has_thinking(text: str) -> bool:
    return bool(_THINKING_OPEN_RE.search(text)) and bool(_THINKING_CLOSE_RE.search(text))


def has_any_thinking_tag(text: str) -> bool:
    return bool(_THINKING_OPEN_RE.search(text)) or bool(_THINKING_CLOSE_RE.search(text))


def classify_task(source: str) -> str:
    """gsm8k → COMPLEX, alpaca → SIMPLE."""
    return "COMPLEX" if "gsm8k" in source.lower() else "SIMPLE"


def score_trajectory(text: str, task_type: str) -> Tuple[float, Dict[str, Any]]:
    """Lightweight scorer for pair selection ranking."""
    tag = extract_tag(text)
    score = 0.0
    info: Dict[str, Any] = {"tag": tag}

    # Format
    if tag != TAG_NONE and starts_with_tag(text, tag):
        score += 2.0
    elif tag != TAG_NONE:
        score += 1.0
    else:
        score -= 10.0

    # Thinking integrity
    info["has_thinking"] = has_thinking(text)
    info["has_any_thinking"] = has_any_thinking_tag(text)
    if tag == TAG_DEEP and has_thinking(text):
        score += 3.0  # Bonus for proper format
    elif tag == TAG_DEEP and not has_thinking(text):
        score -= 2.0  # Penalty for broken DEEP

    # Alignment
    if task_type == "COMPLEX" and tag == TAG_DEEP:
        score += 5.0
    elif task_type == "COMPLEX" and tag == TAG_FAST:
        score -= 5.0
    elif task_type == "SIMPLE" and tag == TAG_FAST:
        score += 5.0
    elif task_type == "SIMPLE" and tag == TAG_DEEP:
        score -= 3.0

    # Content length sanity
    if len(text.strip()) > 10:
        score += 1.0

    info["score"] = score
    return score, info


def _make_pair(
    prompt: Dict[str, Any],
    chosen: Dict[str, Any],
    rejected: Dict[str, Any],
    pair_type: str,
) -> Dict[str, Any]:
    dpo_prompt = f"{prompt['system']}\n{prompt['user']}"
    return {
        "prompt": dpo_prompt,
        "chosen": chosen["text"],
        "rejected": rejected["text"],
        "_meta": {
            "prompt_id": prompt["prompt_id"],
            "source": prompt["source"],
            "pair_type": pair_type,
            "chosen_tag": chosen["tag"],
            "rejected_tag": rejected["tag"],
            "score_gap": chosen["score"] - rejected["score"],
        },
    }


def _score_trajs(prompt: Dict[str, Any], task_type: str) -> List[Dict[str, Any]]:
    scored = []
    for t in prompt["trajectories"]:
        text = t["text"]
        sc, info = score_trajectory(text, task_type)
        scored.append({
            "index": t["index"],
            "text": text,
            "score": sc,
            "tag": info["tag"],
            "has_thinking": info["has_thinking"],
            "has_any_thinking": info["has_any_thinking"],
        })
    return scored


def extract_balanced_pairs(trajectories_path: str) -> List[Dict[str, Any]]:
    """Extract balanced FAST-chosen + DEEP-chosen pairs.

    Three-layer extraction strategy:
      Layer 1: FAST chosen from SIMPLE tasks (relaxed: allow thinking tags,
               prefer without). Rejected = DEEP or NONE.
      Layer 2: DEEP chosen from COMPLEX tasks (proper <thinking> required).
               Rejected = FAST/NONE or DEEP with broken thinking.
      Layer 3: Fill remaining DEEP slots from SIMPLE tasks where model used
               DEEP with proper thinking vs NONE rejected (teaches format
               integrity even when over-thinking).
    """
    logger.info(f"Loading trajectories from {trajectories_path}...")
    prompts: List[Dict[str, Any]] = []
    with open(trajectories_path) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    logger.info(f"  Loaded {len(prompts)} prompts")

    simple_prompts = [p for p in prompts if classify_task(p["source"]) == "SIMPLE"]
    complex_prompts = [p for p in prompts if classify_task(p["source"]) == "COMPLEX"]
    logger.info(f"  Simple: {len(simple_prompts)}, Complex: {len(complex_prompts)}")

    fast_pairs: List[Dict[str, Any]] = []
    deep_pairs: List[Dict[str, Any]] = []
    deep_format_pairs: List[Dict[str, Any]] = []  # Layer 3

    # ── Layer 1: FAST chosen from SIMPLE tasks ──
    # Relaxed: allow thinking tags. Prefer clean (no thinking) via score.
    for prompt in simple_prompts:
        if len(fast_pairs) >= TARGET_FAST_PAIRS:
            break
        scored = _score_trajs(prompt, "SIMPLE")

        # FAST candidates: starts with tag, has text (allow thinking)
        fast_cands = [
            s for s in scored
            if starts_with_tag(s["text"], TAG_FAST) and len(s["text"].strip()) > 10
        ]
        # Prefer clean FAST (no thinking) by giving bonus sort key
        fast_cands.sort(
            key=lambda s: (not s["has_any_thinking"], s["score"]),
            reverse=True,
        )

        rejected_cands = [
            s for s in scored
            if s["tag"] in (TAG_DEEP, TAG_NONE) and len(s["text"].strip()) > 10
        ]

        if fast_cands and rejected_cands:
            chosen = fast_cands[0]
            rejected = min(rejected_cands, key=lambda x: x["score"])
            gap = chosen["score"] - rejected["score"]
            if gap >= 3.0:
                fast_pairs.append(_make_pair(prompt, chosen, rejected, "FAST_CHOSEN"))

    # ── Layer 2: DEEP chosen from COMPLEX tasks ──
    for prompt in complex_prompts:
        if len(deep_pairs) >= TARGET_DEEP_PAIRS:
            break
        scored = _score_trajs(prompt, "COMPLEX")

        deep_cands = [
            s for s in scored
            if starts_with_tag(s["text"], TAG_DEEP)
            and s["has_thinking"]
            and len(s["text"].strip()) > 10
        ]

        rejected_cands = [
            s for s in scored
            if (
                s["tag"] in (TAG_FAST, TAG_NONE)
                or (s["tag"] == TAG_DEEP and not s["has_thinking"])
            )
            and len(s["text"].strip()) > 10
        ]

        if deep_cands and rejected_cands:
            chosen = max(deep_cands, key=lambda x: x["score"])
            rejected = min(rejected_cands, key=lambda x: x["score"])
            gap = chosen["score"] - rejected["score"]
            if gap >= 3.0:
                deep_pairs.append(_make_pair(prompt, chosen, rejected, "DEEP_CHOSEN"))

    # ── Layer 3: Fill remaining DEEP from SIMPLE tasks ──
    # DEEP+proper-thinking vs NONE/broken → teaches format integrity
    deep_deficit = TARGET_DEEP_PAIRS - len(deep_pairs)
    if deep_deficit > 0:
        logger.info(f"  Layer 3: need {deep_deficit} more DEEP pairs from simple tasks")
        for prompt in simple_prompts:
            if len(deep_format_pairs) >= deep_deficit:
                break
            scored = _score_trajs(prompt, "SIMPLE")

            deep_cands = [
                s for s in scored
                if starts_with_tag(s["text"], TAG_DEEP)
                and s["has_thinking"]
                and len(s["text"].strip()) > 10
            ]
            # Rejected: NONE, or DEEP with broken thinking, or FAST
            format_rejected = [
                s for s in scored
                if (
                    s["tag"] == TAG_NONE
                    or (s["tag"] == TAG_DEEP and not s["has_thinking"])
                    or s["tag"] == TAG_FAST
                )
                and len(s["text"].strip()) > 10
            ]

            if deep_cands and format_rejected:
                chosen = max(deep_cands, key=lambda x: x["score"])
                rejected = min(format_rejected, key=lambda x: x["score"])
                gap = chosen["score"] - rejected["score"]
                if gap >= 2.0:
                    deep_format_pairs.append(
                        _make_pair(prompt, chosen, rejected, "DEEP_FORMAT")
                    )

    logger.info(f"  Layer 1 — FAST chosen:  {len(fast_pairs)} / {TARGET_FAST_PAIRS}")
    logger.info(f"  Layer 2 — DEEP chosen:  {len(deep_pairs)} / {TARGET_DEEP_PAIRS}")
    logger.info(f"  Layer 3 — DEEP format:  {len(deep_format_pairs)} / {deep_deficit}")

    # Combine DEEP layers
    all_deep = deep_pairs + deep_format_pairs
    all_deep = all_deep[:TARGET_DEEP_PAIRS]

    # Match FAST count to DEEP to ensure balance
    final_count = min(len(fast_pairs), len(all_deep), TARGET_FAST_PAIRS)
    fast_pairs = fast_pairs[:final_count]
    all_deep = all_deep[:final_count]

    all_pairs = fast_pairs + all_deep
    logger.info(f"  Final: {len(fast_pairs)} FAST + {len(all_deep)} DEEP = {len(all_pairs)} balanced")

    # Stats
    if all_pairs:
        fast_gaps = [p["_meta"]["score_gap"] for p in fast_pairs]
        deep_gaps = [p["_meta"]["score_gap"] for p in deep_pairs]
        if fast_gaps:
            logger.info(f"  FAST pair gaps: avg={sum(fast_gaps)/len(fast_gaps):.2f}, "
                        f"min={min(fast_gaps):.1f}, max={max(fast_gaps):.1f}")
        if deep_gaps:
            logger.info(f"  DEEP pair gaps: avg={sum(deep_gaps)/len(deep_gaps):.2f}, "
                        f"min={min(deep_gaps):.1f}, max={max(deep_gaps):.1f}")

        # Rejected tag distribution
        from collections import Counter
        fast_rej_tags = Counter(p["_meta"]["rejected_tag"] for p in fast_pairs)
        deep_rej_tags = Counter(p["_meta"]["rejected_tag"] for p in deep_pairs)
        logger.info(f"  FAST rejected tags: {dict(fast_rej_tags)}")
        logger.info(f"  DEEP rejected tags: {dict(deep_rej_tags)}")

    return all_pairs


def save_pairs(pairs: List[Dict[str, Any]], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    logger.info(f"  Saved {len(pairs)} pairs to {output_path}")


# ═════════════════════════════════════════════════════
# PART 2: KL-ANCHORED RE-TRAINING
# ═════════════════════════════════════════════════════

def load_dpo_dataset(data_path: str) -> Dataset:
    records: List[Dict[str, str]] = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                records.append({
                    "prompt": item["prompt"],
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                })
    return Dataset.from_list(records)


def run_dpo_training(
    data_path: str,
    model_path: str = SFT_MODEL_PATH,
    output_dir: str = DPO_OUTPUT_DIR,
    merged_path: str = DPO_MERGED_PATH,
    beta: float = 0.2,
    learning_rate: float = 2e-6,
    batch_size: int = 2,
    grad_accum: int = 8,
    num_epochs: int = 1,
    max_length: int = 1536,
) -> None:
    """DPO training with stronger KL anchoring."""

    logger.info("=" * 60)
    logger.info("PART 2: KL-Anchored DPO Re-training")
    logger.info("=" * 60)

    # Load dataset
    dataset = load_dpo_dataset(data_path)
    logger.info(f"  Dataset: {len(dataset)} pairs")

    # Load model
    logger.info(f"  Loading SFT base from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # LoRA config — keep modules_to_save for special tokens
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "gate_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
        bias="none",
    )

    effective_bs = batch_size * grad_accum
    logger.info(f"  LoRA: r=16, α=32, modules_to_save=[embed_tokens, lm_head]")
    logger.info(f"  DPO: β={beta}, lr={learning_rate}, bs={batch_size}×{grad_accum}={effective_bs}")
    logger.info(f"  Epochs: {num_epochs}, max_length: {max_length}")

    # DPO config with stronger KL anchoring
    dpo_config = DPOConfig(
        output_dir=output_dir,
        beta=beta,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=max_length,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        report_to="none",
        seed=42,
    )

    # Trainer
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    logger.info(f"  Trainable: {trainable/1e6:.1f}M / {total/1e9:.2f}B ({trainable/total*100:.2f}%)")

    try:
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        logger.info(f"  VRAM: {free_mem/(1024**3):.1f}GB free / {total_mem/(1024**3):.1f}GB total")
    except Exception:
        pass

    # Train
    logger.info("  Starting DPO training...")
    train_result = trainer.train()

    metrics = train_result.metrics
    logger.info("  Training complete. Metrics:")
    for k, v in sorted(metrics.items()):
        logger.info(f"    {k}: {v}")

    # Save adapter
    trainer.save_model(output_dir)
    logger.info(f"  Adapter saved to {output_dir}")

    # Merge
    logger.info(f"  Merging LoRA → {merged_path}...")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)
    logger.info(f"  Merged model saved to {merged_path}")

    # Quick sanity check with clean load
    logger.info("  Running quick inference sanity check...")
    del model, trainer, merged_model
    gc.collect()
    torch.cuda.empty_cache()

    try:
        test_model = AutoModelForCausalLM.from_pretrained(
            merged_path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, attn_implementation="sdpa",
        )
        test_model.eval()
        test_tok = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)

        tests = [
            ("What is 5 + 7?", "FAST expected"),
            ("Solve: 3x + 2y = 16, x - y = 2", "DEEP expected"),
            ("What is the capital of France?", "FAST expected"),
        ]
        for prompt_text, label in tests:
            msgs = [
                {"role": "system", "content": METIS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ]
            ids = test_tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
                return_tensors="pt",
            ).to(test_model.device)
            with torch.no_grad():
                out = test_model.generate(
                    ids, max_new_tokens=300, do_sample=False,
                    attention_mask=torch.ones_like(ids),
                    pad_token_id=test_tok.pad_token_id,
                )
            resp = test_tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
            if test_tok.eos_token:
                resp = resp.split(test_tok.eos_token)[0]
            tag = extract_tag(resp)
            thinking = has_thinking(resp)
            logger.info(f"    [{label}] '{prompt_text[:40]}' → [{tag}] thinking={thinking}")
            logger.info(f"      {resp[:150]}")

        del test_model
    except Exception as e:
        logger.warning(f"  Inference check failed: {e}")

    # Final report
    logger.info("=" * 60)
    logger.info("PHASE 6 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Balanced pairs: {len(dataset)}")
    logger.info(f"  Train loss: {metrics.get('train_loss', 'N/A')}")
    logger.info(f"  β: {beta}, lr: {learning_rate}")
    logger.info(f"  Model: {merged_path}")
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="METIS Phase 6: Balanced DPO")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip pair extraction, use existing balanced file")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, only extract pairs")
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("METIS Phase 6: DPO Re-balancing & Re-training")
    logger.info("=" * 60)

    # PART 1: Extract balanced pairs
    if not args.skip_extract:
        logger.info("\nPART 1: Extracting balanced DPO pairs...")
        pairs = extract_balanced_pairs(TRAJECTORIES_PATH)
        save_pairs(pairs, BALANCED_OUTPUT_PATH)
    else:
        logger.info("Skipping extraction, using existing balanced file.")

    # PART 2: Re-train
    if not args.skip_train:
        run_dpo_training(
            data_path=BALANCED_OUTPUT_PATH,
            beta=args.beta,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
        )
    else:
        logger.info("Skipping training.")


if __name__ == "__main__":
    main()
