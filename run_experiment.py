#!/usr/bin/env python3
"""
METIS Training Experiment — A/B Comparison (CLI Controller)

Thin entry point that delegates to modular pipeline phases:
    metis/pipeline/config.py           — shared types, prompts, config
    metis/pipeline/generator_phase.py  — Phase 1: Generate & Score
    metis/pipeline/trainer_phase.py    — Phase 2: DPO Training
    metis/pipeline/evaluator_phase.py  — Phase 3: Evaluation + Phase 4: Report

Usage:
    python run_experiment.py --model Qwen/Qwen2.5-0.5B-Instruct --n-prompts 50
    python run_experiment.py --model meta-llama/Llama-3.2-1B-Instruct --device cuda
    python run_experiment.py --phase eval --metis-checkpoint ./output/metis_dpo
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiment")

from metis.pipeline.config import C, ExperimentConfig
from metis.pipeline.generator_phase import phase1_generate
from metis.pipeline.trainer_phase import phase2_train
from metis.pipeline.evaluator_phase import phase3_evaluate, phase4_report


# ═══════════════════════════════════════════════════════
# Main CLI
# ═══════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="METIS Training Experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-72B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cuda / cpu / auto")
    parser.add_argument("--output", type=str, default="./experiment_output",
                        help="Output directory")
    parser.add_argument("--n-prompts", type=int, default=300,
                        help="Number of training prompts")
    parser.add_argument("--n-samples", type=int, default=16,
                        help="Samples per prompt")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Max new tokens per generation")
    parser.add_argument("--dpo-epochs", type=int, default=3,
                        help="DPO training epochs")
    parser.add_argument("--dpo-lr", type=float, default=1e-6,
                        help="DPO learning rate")
    parser.add_argument("--lora-r", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "generate", "train", "eval"],
                        help="Which phase to run")
    parser.add_argument("--metis-checkpoint", type=str, default=None,
                        help="Path to METIS DPO checkpoint (for eval-only)")
    parser.add_argument("--random-checkpoint", type=str, default=None,
                        help="Path to Random DPO checkpoint (for eval-only)")
    parser.add_argument("--vllm", type=str, default=None,
                        help="vLLM server URL (e.g. http://localhost:8000/v1). "
                             "Enables 2-phase generation: vLLM batch gen + teacher-forcing.")
    parser.add_argument("--vllm-data", type=str, default=None,
                        help="Path to pre-generated vLLM samples JSON (from vllm_batch_generate.py). "
                             "Skips generation, only does teacher-forcing for METIS traces.")
    parser.add_argument("--external-dpo", type=str, default=None,
                        help="Path to pre-formatted JSON with {'prompt', 'chosen', 'rejected'} pairs. "
                             "Skips Phase 1 entirely and trains on this external data.")
    parser.add_argument("--no-benchmarks", action="store_true",
                        help="Skip external benchmarks (TruthfulQA + MMLU) in eval phase")
    parser.add_argument("--sft-data", type=str, default=None,
                        help="Path to SFT warmup data JSON [{\"text\": ...}]. "
                             "If not provided but --external-dpo is set, SFT data is auto-generated from chosen responses.")
    parser.add_argument("--no-sft", action="store_true",
                        help="Disable SFT warmup stage (not recommended — causes KL explosion)")
    parser.add_argument("--sft-epochs", type=int, default=1,
                        help="SFT warmup epochs (default: 1)")
    args = parser.parse_args()

    config = ExperimentConfig(
        model_name=args.model,
        device=args.device,
        output_dir=args.output,
        n_train_prompts=args.n_prompts,
        n_samples_per_prompt=args.n_samples,
        max_new_tokens=args.max_tokens,
        dpo_epochs=args.dpo_epochs,
        dpo_learning_rate=args.dpo_lr,
        lora_r=args.lora_r,
        external_dpo_data=args.external_dpo,
        sft_warmup=not args.no_sft,
        sft_data_path=args.sft_data,
        sft_epochs=args.sft_epochs,
        eval_max_tokens=args.max_tokens,
        run_benchmarks=not args.no_benchmarks,
    )
    vllm_url = args.vllm
    vllm_data_path = args.vllm_data

    os.makedirs(config.output_dir, exist_ok=True)

    print(f"""{C.GREEN}
███╗   ███╗███████╗████████╗██╗███████╗
████╗ ████║██╔════╝╚══██╔══╝██║██╔════╝
██╔████╔██║█████╗     ██║   ██║███████╗
██║╚██╔╝██║██╔══╝     ██║   ██║╚════██║
██║ ╚═╝ ██║███████╗   ██║   ██║███████║
╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝╚══════╝
{C.RESET}
 {C.BOLD}[SYSTEM::METIS]{C.RESET} {C.CYAN}Training Experiment{C.RESET}
 {C.DIM}Cognitive Rewards vs Random Baseline{C.RESET}

 > COGNITIVE_LAYER.......[{C.GREEN}ONLINE{C.RESET}]
 > REWARD_COMPUTER.......[{C.GREEN}ACTIVE{C.RESET}]
 > SFT_WARMUP............[{C.GREEN if config.sft_warmup else C.RED}{"ENABLED" if config.sft_warmup else "DISABLED"}{C.RESET}]
 > DPO_TRAINER...........[{C.YELLOW}STANDBY{C.RESET}]
 > EVAL_PIPELINE.........[{C.YELLOW}STANDBY{C.RESET}]

 root@agi:~$ {C.GREEN}Initializing Experiment...{C.RESET}

  Model:    {config.model_name}
  Device:   {config.device}
  Prompts:  {config.n_train_prompts} train + {config.n_eval_prompts} eval
  Samples:  {config.n_samples_per_prompt} per prompt
  Output:   {config.output_dir}
""")

    start = time.time()

    if args.phase in ("all", "generate"):
        if config.external_dpo_data:
            logger.info(f"Skipping Phase 1 Generation. Using external DPO dataset: {config.external_dpo_data}")
            scored_data = [] # Not needed, handled in Phase 2
        else:
            scored_data, model, tokenizer = phase1_generate(
                config, vllm_url=vllm_url, vllm_data_path=vllm_data_path,
            )

        if args.phase == "generate":
            logger.info("Phase 1 complete. Use --phase train to continue.")
            return
    else:
        # Load model for later phases
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if device == "cuda":
            model_kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs).to(device)
        model.eval()

        if not config.external_dpo_data:
            # Load scored data
            data_path = os.path.join(config.output_dir, "phase1_scored_data.json")
            with open(data_path, "r", encoding="utf-8") as f:
                scored_data = json.load(f)
        else:
            scored_data = [] # Not used, Phase 2 will load directly

    if args.phase in ("all", "train"):
        metis_path, random_path = phase2_train(config, scored_data, model, tokenizer)
    else:
        metis_path = args.metis_checkpoint or os.path.join(config.output_dir, "metis_dpo")
        random_path = args.random_checkpoint or os.path.join(config.output_dir, "random_dpo")

    if args.phase in ("all", "eval"):
        base_metrics, metis_metrics, random_metrics = phase3_evaluate(
            config, model, tokenizer, metis_path, random_path,
        )
        phase4_report(config, base_metrics, metis_metrics, random_metrics)

    elapsed = time.time() - start
    logger.info(f"Experiment completed in {elapsed:.1f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
