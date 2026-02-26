"""
METIS Pipeline — Night-time Dreaming Training (Offline Learning)

AGI infrastructure for continuous autonomous self-evolution.
When the system is idle (e.g., at night), it processes the "Knowledge Gaps"
collected during the day by the CuriosityDriver.

Pipeline:
    1. Collect unresolved KnowledgeGaps (recorded when H(s) spiked in prod)
    2. Deep search: Use EGTS (Entropy-Guided Tree Search) with massive beam
       width and relaxed token limits to find a low-entropy convergence path.
    3. Pair generation: The low-entropy path becomes the "chosen" sample.
       The original failing path (or a new high-entropy generation) becomes
       the "rejected" sample.
    4. Fine-tuning: Run DPO on these pairs to internalize the correct
       reasoning path back into System 1 weights.
    5. Mark gaps as resolved.

Usage:
    python -m metis.pipeline.night_training --config configs/dgx_full.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import torch

from metis.metis import Metis
from metis.pipeline.config import ExperimentConfig
from metis.pipeline.yaml_config import load_config
from metis.search.entropy_search import EntropyGuidedSearch
from metis.search.tree_node import SearchConfig

logger = logging.getLogger("metis.night")


def run_night_training(
    config: ExperimentConfig,
    knowledge_gap_path: str = "metis_knowledge_gaps.json",
    output_dir: str = "./night_dreams",
) -> None:
    """Run the offline dreaming training loop."""
    logger.info("=" * 60)
    logger.info("METIS Night-time Dreaming Phase Activated")
    logger.info("=" * 60)

    # ── 1. Load Knowledge Gaps ──
    if not os.path.exists(knowledge_gap_path):
        logger.info(f"No knowledge gaps found at {knowledge_gap_path}. Sleep well.")
        return

    with open(knowledge_gap_path, "r", encoding="utf-8") as f:
        all_gaps = json.load(f)

    unresolved = [g for g in all_gaps if not g.get("resolved", False)]
    if not unresolved:
        logger.info("No unresolved knowledge gaps. Sleep well.")
        return

    logger.info(f"Found {len(unresolved)} unresolved gaps. Waking up model...")

    # ── 2. Load Model ──
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, **model_kwargs
    ).to(device)
    model.eval()

    # ── 3. Deep Search Configuration ──
    # Night time has no latency constraints. Search deep and wide.
    search_cfg = SearchConfig(
        beam_width=32,            # Massive beam width
        max_depth=1024,           # Allow very long reasoning chains
        max_nodes=8192,           # Heavy compute budget
        entropy_branch_threshold=1.5, # Branch earlier
    )
    searcher = EntropyGuidedSearch(model, tokenizer, config=search_cfg)

    # Prepare DPO pairs
    dpo_pairs: List[Dict[str, str]] = []
    resolved_queries: List[str] = []

    logger.info("Starting deep search exploration for knowledge gaps...")

    for i, gap in enumerate(unresolved):
        query = gap["query"]
        logger.info(f"[{i+1}/{len(unresolved)}] Exploring gap: {query[:50]}...")

        # Run EGTS to find a correct, low-entropy path
        t0 = time.time()
        result = searcher.search(query, max_tokens=2048, chat_template=True)
        search_time = time.time() - t0

        if result.convergence_achieved:
            logger.info(
                f"  -> Solved! (H={result.best_path_entropy:.3f}, "
                f"time={search_time:.1f}s, nodes={result.total_nodes_created})"
            )
            # Create a rejected sample using greedy decoding (simulate the failure)
            # In a real setup, we would save the original failing trace.
            chat = [{"role": "user", "content": query}]
            prompt_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            with torch.no_grad():
                bad_outputs = model.generate(
                    **inputs, max_new_tokens=256, temperature=0.1, do_sample=False
                )
            bad_text = tokenizer.decode(
                bad_outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            )

            dpo_pairs.append({
                "prompt": query,
                "chosen": result.text,
                "rejected": bad_text,
            })
            resolved_queries.append(query)
        else:
            logger.warning(
                f"  -> Failed to converge. Minimum H={result.best_path_entropy:.3f}"
            )

    # ── 4. Offline Fine-tuning (DPO) ──
    if not dpo_pairs:
        logger.info("No gaps successfully resolved. Exiting.")
        return

    logger.info(f"Generated {len(dpo_pairs)} DPO pairs from successful searches.")
    os.makedirs(output_dir, exist_ok=True)

    # Save pairs for auditing
    pairs_file = os.path.join(output_dir, "dream_pairs.json")
    with open(pairs_file, "w", encoding="utf-8") as f:
        json.dump(dpo_pairs, f, indent=2, ensure_ascii=False)

    logger.info("Starting offline DPO fine-tuning to internalize knowledge...")
    # Dynamic import to avoid circular dependency
    from metis.pipeline.trainer_phase import _train_dpo
    model_output = os.path.join(output_dir, "model_checkpoint")
    _train_dpo(config, model, tokenizer, dpo_pairs, model_output)

    # ── 5. Mark as Resolved ──
    for gap in all_gaps:
        if gap["query"] in resolved_queries:
            gap["resolved"] = True

    with open(knowledge_gap_path, "w", encoding="utf-8") as f:
        json.dump(all_gaps, f, indent=2, ensure_ascii=False)

    logger.info(f"Night-time training complete. {len(resolved_queries)} gaps resolved.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="METIS Night-time Dreaming Training")
    parser.add_argument("--config", type=str, default="", help="Path to YAML config")
    parser.add_argument("--gaps", type=str, default="metis_knowledge_gaps.json")
    parser.add_argument("--output", type=str, default="./night_dreams")
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        config = ExperimentConfig()

    run_night_training(config, knowledge_gap_path=args.gaps, output_dir=args.output)


if __name__ == "__main__":
    main()
