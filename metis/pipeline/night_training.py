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


def _simulate_web_search(query: str) -> str:
    """Mock external tool call (e.g. Wikipedia/Google Search)."""
    # In a real system, this would call Tavily/Google Search API
    logger.info(f"    [Tool] Executing search for: {query}")
    # Returning a mock factual context to simulate breaking the knowledge gap
    return f"The widely accepted factual consensus regarding '{query}' is that it functions through established theoretical principles."


def _train_cpt(config: ExperimentConfig, model: Any, tokenizer: Any, texts: List[str], output_path: str) -> None:
    """Run Continual Pre-Training on factual text to inject knowledge into weights."""
    from datasets import Dataset
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    
    # Simple CPT dataset
    dataset = Dataset.from_dict({"text": texts})
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=config.max_new_tokens)
        
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=5,
        per_device_train_batch_size=config.dpo_batch_size,
        gradient_accumulation_steps=config.dpo_gradient_accumulation,
        learning_rate=2e-5,
        bf16=torch.cuda.is_available(),
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    trainer.save_model(output_path)
    logger.info(f"CPT model saved to {output_path}")


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
    
    # Tool for Epistemic Debate (Counterfactual Verification)
    from metis.search.counterfactual import CounterfactualSimulator
    cf_simulator = CounterfactualSimulator(model, tokenizer, config=search_cfg)

    # Prepare DPO pairs
    dpo_pairs: List[Dict[str, str]] = []
    cpt_texts: List[str] = []
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
                f"  -> Converged (H={result.best_path_entropy:.3f}, "
                f"time={search_time:.1f}s, nodes={result.total_nodes_created})"
            )
            
            # ── 3b. Multi-Agent Epistemic Debate (Counterfactual Check) ──
            # Prevent "Mode Collapse" (logic self-consistency but factually wrong/hallucinated)
            logger.info("  -> Running Epistemic Debate (Counterfactual Simulation)...")
            
            # Build full prompt tensor for CF
            chat = [{"role": "user", "content": query}]
            prompt_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
            
            cf_result = cf_simulator.simulate(
                prompt_ids=prompt_ids,
                kv_cache=None,
                generated_tokens=result.tokens[:min(len(result.tokens), 128)], # Test early premise
                original_entropy=result.best_path_entropy,
            )
            
            if cf_result.is_fragile:
                logger.warning(
                    f"  -> Debate Failed! Path is fragile/confabulated "
                    f"(F={cf_result.fragility_score:.2f}). Rejecting path."
                )
                continue
                
            logger.info(f"  -> Debate Passed. Path is robust (F={cf_result.fragility_score:.2f}).")
            
            # Create a rejected sample using greedy decoding (simulate the failure)
            # In a real setup, we would save the original failing trace.
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
                f"  -> Failed to converge (H={result.best_path_entropy:.3f}). "
                f"Absolute knowledge blind spot detected."
            )
            
            # ── 3c. Tool-Augmented Dreaming (Breaking the Information Cocoon) ──
            # The model fundamentally doesn't know the fact. EGTS cannot invent facts.
            # Trigger external retrieval.
            logger.info("  -> Triggering Tool-Augmented Retrieval...")
            retrieved_context = _simulate_web_search(query)
            
            if retrieved_context:
                logger.info(f"  -> Retrieved external context: {retrieved_context[:60]}...")
                
                # CPT Track: Direct knowledge injection
                cpt_text = f"Question: {query}\nContext: {retrieved_context}\nAnswer: Based on the retrieved context, the correct information is that {retrieved_context}."
                cpt_texts.append(cpt_text)
                
                # Re-run search with context
                augmented_prompt = f"Context: {retrieved_context}\nQuestion: {query}"
                logger.info("  -> Re-running EGTS with augmented context...")
                aug_result = searcher.search(augmented_prompt, max_tokens=1024, chat_template=True)
                
                if aug_result.convergence_achieved:
                    logger.info("  -> Solved via Tool Augmentation!")
                    
                    # Create greedy failure for DPO (WITHOUT context)
                    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
                    with torch.no_grad():
                        bad_outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=False)
                    bad_text = tokenizer.decode(bad_outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    
                    # DPO teaches the model: "When asked this, output the augmented reasoning, not your hallucination"
                    dpo_pairs.append({
                        "prompt": query,
                        "chosen": aug_result.text,
                        "rejected": bad_text,
                    })
                    resolved_queries.append(query)
                else:
                    logger.warning("  -> Still failed to converge even with context. Gap too complex.")
            else:
                logger.warning("  -> Retrieval failed.")

    # ── 4. Offline Fine-tuning (CPT + DPO Dual Track) ──
    if not dpo_pairs and not cpt_texts:
        logger.info("No gaps successfully resolved. Exiting.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
        
    # Track A: Continual Pre-Training (CPT) for factual injection
    if cpt_texts:
        logger.info(f"Starting CPT track: Injecting {len(cpt_texts)} new facts into weights...")
        cpt_file = os.path.join(output_dir, "cpt_corpus.txt")
        with open(cpt_file, "w", encoding="utf-8") as f:
            for text in cpt_texts:
                f.write(text + "\n\n")
        _train_cpt(config, model, tokenizer, cpt_texts, os.path.join(output_dir, "cpt_checkpoint"))
        logger.info("CPT Fact Injection complete.")

    # Track B: DPO for reasoning alignment
    if dpo_pairs:
        logger.info(f"Generated {len(dpo_pairs)} DPO pairs from successful searches.")
        
        # Save pairs for auditing
        pairs_file = os.path.join(output_dir, "dream_pairs.json")
        with open(pairs_file, "w", encoding="utf-8") as f:
            json.dump(dpo_pairs, f, indent=2, ensure_ascii=False)

        logger.info("Starting offline DPO fine-tuning to internalize logic...")
        # Dynamic import to avoid circular dependency
        from metis.pipeline.trainer_phase import _train_dpo
        model_output = os.path.join(output_dir, "model_checkpoint")
        _train_dpo(config, model, tokenizer, dpo_pairs, model_output)
        logger.info("DPO Logic Alignment complete.")

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
