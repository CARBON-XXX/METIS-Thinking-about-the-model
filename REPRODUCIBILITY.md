# METIS — Reproducibility Guide

## Quick Start (3 commands)

```bash
# 1. Clone and install
git clone https://github.com/CARBON-XXX/METIS-Thinking-about-the-model.git
cd METIS-Thinking-about-the-model
pip install -r requirements.txt && pip install -e .

# 2. (Optional) Build Rust native accelerators (~6-75x speedup)
cd metis/_native && pip install maturin && maturin develop --release && cd ../..

# 3. Run the benchmark
python benchmarks/evaluate.py --model Qwen/Qwen2.5-7B-Instruct --output results/
```

## Environment

| Component        | Tested Version        | Minimum            |
|------------------|-----------------------|--------------------|
| Python           | 3.12                  | 3.9                |
| PyTorch          | 2.9.1                 | 2.0                |
| Transformers     | 4.52+                 | 4.36               |
| CUDA             | 13.0 (Blackwell)      | 11.8               |
| Rust (optional)  | 1.82                  | 1.70               |

## Hardware Profiles

| Profile          | GPU                     | VRAM    | Notes                         |
|------------------|-------------------------|---------|-------------------------------|
| Dev              | Any CUDA GPU            | 8 GB    | 1.5B model, LoRA, bf16       |
| Paper (default)  | NVIDIA GB10 / A100      | 24+ GB  | 7B model, full bf16          |
| Production       | DGX Spark / H100        | 80+ GB  | 72B model, bf16              |

## Reproducing Paper Results

### 1. Cognitive DPO Training

```bash
python run_experiment.py \
    --phase train \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output experiment_output/ \
    --n-prompts 1000 --n-samples 8 --max-tokens 512
```

### 2. Evaluation (TruthfulQA, MMLU, GSM8K)

```bash
python run_experiment.py \
    --phase eval \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output experiment_output/
```

### 3. Academic Benchmark (Table 1 in paper)

```bash
python tools/metis_academic_benchmark.py \
    --n-complex 50 --n-simple 50
```

## Rust Native Accelerators

The Rust extensions are **optional** — METIS falls back to pure-Python
implementations automatically. To build:

```bash
cd metis/_native
pip install maturin
maturin develop --release
cd ../..
python -c "import metis_native; print('Native OK')"
```

## Configuration

Experiment configs are in `configs/`. Load with:

```python
from metis.pipeline.yaml_config import load_config
config = load_config("configs/dgx_full.yaml")
```

## Checksum Verification

Model checkpoints are saved with `safetensors` format.
Verify integrity:

```bash
python -c "
from safetensors.torch import load_file
w = load_file('experiment_output/metis_dpo/adapter_model.safetensors')
print(f'Adapter keys: {len(w)}, total params: {sum(v.numel() for v in w.values()):,}')
"
```
