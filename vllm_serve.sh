#!/bin/bash
# METIS vLLM Server — run in WSL2
# Usage: wsl -e bash -c "cd /mnt/g/SEDACV9.0\ PRO && bash vllm_serve.sh"
#
# Serves Qwen2.5-1.5B-Instruct via OpenAI-compatible API on port 8000.
# RTX 4060 8GB: gpu_memory_utilization=0.85 leaves ~1.2GB headroom.

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sedac_dev

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
PORT=8000
GPU_MEM=0.85

echo "Starting vLLM server: ${MODEL} on port ${PORT}"
echo "GPU memory utilization: ${GPU_MEM}"

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port "${PORT}" \
    --host 0.0.0.0 \
    --gpu-memory-utilization "${GPU_MEM}" \
    --max-model-len 1024 \
    --dtype float16 \
    --trust-remote-code \
    --disable-log-requests
