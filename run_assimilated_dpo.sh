#!/bin/bash
# METIS Architecture Fix v3: SFT(resume) → DPO(1ep) → Generative Eval
# Time budget: ~13h on GB10 (122GB unified memory)
#
# Fixes applied:
#   1. Generative parsing for MMLU/TruthfulQA (no logit blind spot)
#   2. SFT warmup to teach π_ref the <thinking> format (KL stability)
#   3. Ground-truth correctness in reward (anti-eloquent-nonsense)
#   4. DPO batch=2/grad_accum=8/max_len=1024 (GB10 OOM-safe)
#   5. DPO 1 epoch (standard, prevents overfitting)
#
# Time estimate:
#   SFT: ~5s (resume from checkpoint)
#   METIS DPO 1ep: ~6.5h (1000 micro-batches × 23s)
#   Random DPO 1ep: ~6.5h
#   Eval + Benchmarks: ~1h
#   Total: ~14h

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METIS_VENV="${METIS_VENV:-$(python3 -c 'import sys; print(sys.prefix)' 2>/dev/null || echo "$SCRIPT_DIR/.venv")}"
[ -f "$METIS_VENV/bin/activate" ] && source "$METIS_VENV/bin/activate"
export TORCHDYNAMO_DISABLE=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

OUTPUT_DIR="experiment_output_7B_orca_v2_sft_dpo"
LOG="pipeline_final.log"

echo "$(date '+%Y-%m-%d %H:%M:%S') === METIS Full Pipeline Start ===" | tee "$LOG"

# Only clean DPO dirs, preserve SFT checkpoint
rm -rf "$OUTPUT_DIR/metis_dpo" "$OUTPUT_DIR/random_dpo"
echo "$(date '+%Y-%m-%d %H:%M:%S') Cleaned DPO dirs, SFT checkpoint preserved" | tee -a "$LOG"

echo "$(date '+%Y-%m-%d %H:%M:%S') === Phase 1: SFT(resume) + DPO Training ===" | tee -a "$LOG"
python3 run_experiment.py \
    --phase train \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output "$OUTPUT_DIR" \
    --external-dpo data/external_dpo_pairs_metis.json \
    --sft-data data/sft_warmup_orca.json \
    --sft-epochs 1 \
    --dpo-epochs 1 \
    --dpo-lr 1e-6 \
    --lora-r 32 \
    2>&1 | tee -a "$LOG"

TRAIN_EXIT=$?
echo "$(date '+%Y-%m-%d %H:%M:%S') Train exit code: $TRAIN_EXIT" | tee -a "$LOG"

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') !!! Training FAILED — aborting pipeline !!!" | tee -a "$LOG"
    exit 1
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') === Phase 2: Evaluation with Generative Benchmarks ===" | tee -a "$LOG"
python3 run_experiment.py \
    --phase eval \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output "$OUTPUT_DIR" \
    --external-dpo data/external_dpo_pairs_metis.json \
    2>&1 | tee -a "$LOG"

EVAL_EXIT=$?
echo "$(date '+%Y-%m-%d %H:%M:%S') Eval exit code: $EVAL_EXIT" | tee -a "$LOG"

echo "$(date '+%Y-%m-%d %H:%M:%S') === Pipeline Complete ===" | tee -a "$LOG"
echo "Results: $OUTPUT_DIR/experiment_report.json" | tee -a "$LOG"

# Quick summary
if [ -f "$OUTPUT_DIR/experiment_report.json" ]; then
    echo "--- Report Preview ---" | tee -a "$LOG"
    python3 -c "
import json
with open('$OUTPUT_DIR/experiment_report.json') as f:
    r = json.load(f)
print(json.dumps(r, indent=2, ensure_ascii=False)[:2000])
" 2>&1 | tee -a "$LOG"
fi
