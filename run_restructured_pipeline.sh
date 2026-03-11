#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# METIS Architecture Restructure — Full Pipeline v4
# ═══════════════════════════════════════════════════════════════
#
# Fixes applied (vs run_assimilated_dpo.sh):
#   1. Special Token injection: <thinking>, [COGNITIVE_STATE: *] etc.
#      registered as single tokens + model.resize_token_embeddings()
#   2. Ground-truth veto: wrong answer → R_total = -1.0 absolute override
#   3. SFT warmup saves full merged base (Qwen-7B-METIS-SFT)
#   4. DPO batch=2/grad_accum=8/max_len=1024/β=0.25 (GB10 OOM-safe)
#   5. MMLU + TruthfulQA use generative parsing with FINAL ANSWER regex
#   6. Random DPO skipped — 100% compute on METIS main line
#
# Time estimate (GB10, 122GB unified):
#   SFT: ~5s (resume from checkpoint) or ~45min (fresh)
#   METIS DPO 1ep: ~6.5h (1000 micro-batches × 23s)
#   Eval + Benchmarks: ~1h
#   Total: ~8h (resume) or ~8.5h (fresh SFT)
# ═══════════════════════════════════════════════════════════════

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METIS_VENV="${METIS_VENV:-$(python3 -c 'import sys; print(sys.prefix)' 2>/dev/null || echo "$SCRIPT_DIR/.venv")}"
[ -f "$METIS_VENV/bin/activate" ] && source "$METIS_VENV/bin/activate"
export TORCHDYNAMO_DISABLE=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

OUTPUT_DIR="experiment_output_7B_restructured"
LOG="${OUTPUT_DIR}/pipeline_restructured.log"

mkdir -p "$OUTPUT_DIR"

echo "$(date '+%Y-%m-%d %H:%M:%S') === METIS Restructured Pipeline Start ===" | tee "$LOG"

# ── Phase 0: Clean DPO dirs, preserve SFT checkpoint ──
rm -rf "$OUTPUT_DIR/metis_dpo" "$OUTPUT_DIR/random_dpo"
echo "$(date '+%Y-%m-%d %H:%M:%S') [Phase 0] Cleaned DPO dirs, SFT checkpoint preserved" | tee -a "$LOG"

# ── Phase 1+2+3: SFT(resume) → DPO Training ──
echo "$(date '+%Y-%m-%d %H:%M:%S') === Phase 1-3: SFT + DPO Training ===" | tee -a "$LOG"
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

# ── Phase 4: Evaluation with Generative Benchmarks ──
echo "$(date '+%Y-%m-%d %H:%M:%S') === Phase 4: Generative Evaluation ===" | tee -a "$LOG"
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
s = r.get('summary', {})
print(f\"METIS vs Base:   {s.get('metis_vs_base', 'N/A')}\")
print(f\"TruthfulQA Δ:    {s.get('truthfulqa_mc1_delta', 'N/A')}\")
print(f\"MMLU Δ:          {s.get('mmlu_delta', 'N/A')}\")
" 2>&1 | tee -a "$LOG"
fi
