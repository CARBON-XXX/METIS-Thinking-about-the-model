#!/bin/bash
TRAIN_PID=$(pgrep -f "run_experiment.py --phase train")

if [ -z "$TRAIN_PID" ]; then
    echo "Training process not found. It might have already finished or crashed."
else
    echo "Waiting for training process $TRAIN_PID to finish..."
    while kill -0 $TRAIN_PID 2>/dev/null; do
        sleep 60
    done
    echo "Training process finished at $(date)."
fi

echo "Starting evaluation phase..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METIS_VENV="${METIS_VENV:-$(python3 -c 'import sys; print(sys.prefix)' 2>/dev/null || echo "$SCRIPT_DIR/.venv")}"
[ -f "$METIS_VENV/bin/activate" ] && source "$METIS_VENV/bin/activate"
export TORCHDYNAMO_DISABLE=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

python3 run_experiment.py --phase eval --model Qwen/Qwen2.5-7B-Instruct --output experiment_output_7B_full --n-prompts 144 --n-samples 8 --max-tokens 512 > experiment_output_7B_full/eval_auto.log 2>&1

echo "Evaluation finished at $(date)."
