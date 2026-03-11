#!/bin/bash
VLLM_PID=$(pgrep -f "vllm_batch_generate.py")
if [ -n "$VLLM_PID" ]; then
    echo "Waiting for vLLM generation (PID $VLLM_PID) to finish..."
    while kill -0 $VLLM_PID 2>/dev/null; do
        sleep 60
    done
    echo "vLLM generation finished at $(date)."
else
    echo "vLLM generation not found. Assuming finished."
fi

./run_full_1000.sh
