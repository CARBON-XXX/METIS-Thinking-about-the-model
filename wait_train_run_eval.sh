#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METIS_VENV="${METIS_VENV:-$(python3 -c 'import sys; print(sys.prefix)' 2>/dev/null || echo "$SCRIPT_DIR/.venv")}"
[ -f "$METIS_VENV/bin/activate" ] && source "$METIS_VENV/bin/activate"
export TORCHDYNAMO_DISABLE=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "Waiting for Phase 2 (Train) to finish before starting Evaluation..."
while pgrep -f "python3 run_experiment.py --phase train" > /dev/null; do
    sleep 60
done

echo "Training finished. Starting Evaluation (Phase 3 & 4)..."
python3 run_experiment.py --phase eval --model Qwen/Qwen2.5-7B-Instruct --output experiment_output_7B_1000 --n-prompts 1000 --n-samples 8 --max-tokens 512 >> pipeline_1000_restart.log 2>&1

echo "All Pipeline Finished."
