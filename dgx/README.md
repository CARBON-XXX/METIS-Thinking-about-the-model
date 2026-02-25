# DGX Spark — METIS Deployment Programs

Hardware target: **NVIDIA DGX Spark** (Grace Blackwell, 128GB unified memory)

## Programs

### 1. `grpo_train.py` — GRPO Online Training Loop
End-to-end self-improving LLM pipeline:
- vLLM generates G completions per prompt (fast batch decode)
- Teacher-forcing extracts METIS cognitive traces
- CognitiveRewardComputer scores each completion
- Policy gradient update via TRL DPO or manual GRPO loss

```bash
# DGX Spark (70B, full precision)
python dgx/grpo_train.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --dataset metis_prompts.jsonl \
    --num-generations 8 \
    --epochs 3

# RTX 4060 (1.5B, testing)
python dgx/grpo_train.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset metis_prompts.jsonl \
    --num-generations 4 \
    --lora-rank 16 \
    --bf16 false
```

### 2. `vllm_processor.py` — vLLM Native LogitsProcessor
CUSUM monitoring + CoT boost inside vLLM's inference loop:
- **MetisCUSUMProcessor**: Observes entropy per-token, maintains CUSUM state
- **CoTBoostProcessor**: Boosts thinking-token logits when CUSUM fires
- **MetisCompositeProcessor**: Combined observation + intervention

```python
from dgx.vllm_processor import MetisCompositeProcessor
from vllm import LLM, SamplingParams

processor = MetisCompositeProcessor(tokenizer)
params = SamplingParams(logits_processors=[processor], max_tokens=1024)
outputs = llm.generate(prompts, params)
```

### 3. `parallel_search.py` — o1-style Parallel Thinking
Multi-path reasoning with entropy-based pruning:
- Generate 50-100 candidate paths via vLLM
- CUSUM monitors cognitive quality in real-time
- Prune low-quality paths early to save compute
- **MetisIterativeSearch**: Multi-round progressive refinement

```python
from dgx.parallel_search import MetisParallelSearch

searcher = MetisParallelSearch()
result = searcher.search("Complex logic puzzle here...")
print(result.best_response)
print(result.cognitive_report)
```

### 4. `triton_config.pbtxt` — Triton Server Template
Production serving config for Triton + vLLM backend.

```bash
# Setup model repository
mkdir -p model_repository/metis_llm/1
cp dgx/triton_config.pbtxt model_repository/metis_llm/config.pbtxt

# Start server
tritonserver --model-repository=model_repository
```

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  DGX Spark (128GB)                │
│                                                    │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    │
│  │  vLLM    │───>│  METIS   │───>│ Cognitive │    │
│  │ Generate  │    │ Teacher  │    │  Reward   │    │
│  │ (batch)   │    │  Force   │    │ Computer  │    │
│  └──────────┘    └──────────┘    └──────────┘    │
│       │                               │           │
│       v                               v           │
│  ┌──────────────────────────────────────┐        │
│  │     GRPO Policy Gradient Update      │        │
│  │  Loss = -E[A_i * log π(a_i|s_i)]    │        │
│  └──────────────────────────────────────┘        │
│                                                    │
│  ┌──────────────────────────────────────┐        │
│  │     Parallel Search (Inference)      │        │
│  │  50-100 paths × entropy pruning      │        │
│  └──────────────────────────────────────┘        │
│                                                    │
│  ┌──────────────────────────────────────┐        │
│  │     Triton Server (Production)       │        │
│  │  gRPC/HTTP + vLLM + LogitsProcessor  │        │
│  └──────────────────────────────────────┘        │
└──────────────────────────────────────────────────┘
```

## Dependencies

```
# Core
torch >= 2.1
transformers >= 4.40
vllm >= 0.4
trl >= 0.12
peft >= 0.10
datasets >= 2.0

# Optional
tritonclient  # For Triton serving
```

## Day 1 Checklist (DGX Arrives)

1. Install vLLM + TRL + METIS
2. Download Qwen2.5-72B-Instruct
3. Test `demo_metis.py` with 72B (should be instant)
4. Run `grpo_train.py` with 300 prompts × 8 samples
5. Benchmark parallel search on Alice-Bob-Charlie puzzle
6. Deploy Triton server for production inference
