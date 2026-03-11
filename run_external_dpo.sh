#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METIS_VENV="${METIS_VENV:-$(python3 -c 'import sys; print(sys.prefix)' 2>/dev/null || echo "$SCRIPT_DIR/.venv")}"
[ -f "$METIS_VENV/bin/activate" ] && source "$METIS_VENV/bin/activate"
export TORCHDYNAMO_DISABLE=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "Starting Web Monitor..."
nohup python3 monitor/app.py > monitor.log 2>&1 &

echo "Starting Phase 2 DPO with External Dataset (Orca)..."
# We skip phase 1 and go straight to train and eval
# We use 2000 external pairs. We will output to a new directory to not overwrite METIS.
python3 run_experiment.py \
    --phase train \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output experiment_output_7B_orca_2000 \
    --external-dpo data/external_dpo_pairs.json \
    --dpo-epochs 3 \
    --dpo-lr 1e-6 \
    --lora-r 64

echo "Training finished. Starting Evaluation..."
python3 run_experiment.py \
    --phase eval \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output experiment_output_7B_orca_2000 \
    --external-dpo data/external_dpo_pairs.json \
    --no-benchmarks

echo "Pipeline Finished."
