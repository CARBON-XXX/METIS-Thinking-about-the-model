<p align="center">
  <img src="https://img.shields.io/badge/METIS-Preprint-0969da?style=for-the-badge" alt="Status">
  <img src="https://img.shields.io/badge/python-≥3.10-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-≥2.0-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Rust-native_FFI-dea584?style=for-the-badge&logo=rust&logoColor=white" alt="Rust">
  <img src="https://img.shields.io/badge/license-Apache_2.0-green?style=for-the-badge" alt="License">
</p>

<h1 align="center">METIS</h1>
<h3 align="center">Metacognitive Entropy-Triggered Inference Scaling</h3>

<p align="center">
  <b>Adaptive compute allocation via real-time entropy monitoring with Rust-accelerated decision boundaries.</b><br>
  <sub>Attach to any HuggingFace causal LM. Zero model modification. O(1) per token.</sub>
</p>

<p align="center">
  <a href="#key-results">Key Results</a> &nbsp;·&nbsp;
  <a href="#method">Method</a> &nbsp;·&nbsp;
  <a href="#architecture">Architecture</a> &nbsp;·&nbsp;
  <a href="#quick-start">Quick Start</a> &nbsp;·&nbsp;
  <a href="#training-pipeline">Training</a> &nbsp;·&nbsp;
  <a href="#reproducibility">Reproducibility</a> &nbsp;·&nbsp;
  <a href="#paper">Paper</a>
</p>

---

> *Named after Μῆτις (Metis) — the Greek Titaness of wisdom and deep thought.*

## TL;DR

METIS dynamically routes each LLM query through difficulty-appropriate computational pathways (FAST / NORMAL / DEEP) based on real-time token-level entropy monitoring. On a 100-question mixed benchmark:

- **96% accuracy** at only **53.4 avg tokens/request** — Pareto-dominant over all baselines
- **6x latency speedup** via Rust-native entropy computation (1.3 μs at p50)
- **Bimodal token distribution** (ΔBIC = 40.6) mathematically refutes the "short-answer cheating" hypothesis

---

## Key Results

### Pareto Frontier (100-question mixed math/logic benchmark)

| Strategy | Accuracy | Avg Tokens | Wall Time | Throughput |
|:---|:---:|:---:|:---:|:---:|
| Zero-Shot | 90% | 12.4 | 24.3s | 51.1 tok/s |
| Forced CoT | 91% | 156.0 | 118.3s | 131.9 tok/s |
| Self-Consistency (k=5) | 94% | 864.5 | 313.9s | 275.4 tok/s |
| **METIS Dynamic** | **96%** | **53.4** | **51.2s** | **104.3 tok/s** |

METIS achieves the highest accuracy at 4.3x fewer tokens than Forced CoT and 15.8x fewer than Self-Consistency.

### Bimodal Distribution Proof

The token length distribution under METIS routing is **bimodal**, not truncated:

| Test | Statistic | Threshold | Verdict |
|:---|:---:|:---:|:---:|
| ΔBIC (Kass & Raftery) | 40.6 | >10 (very strong) | **Bimodal** |
| Bimodality Coefficient | 0.754 | >0.555 | **Bimodal** |

- **Peak 1 (FAST route):** μ = 22.8 tokens, weight = 69%
- **Peak 2 (DEEP route):** μ = 116.5 tokens, weight = 31%

### Rust Native Acceleration

| Operation | Python p50 | Rust p50 | Speedup |
|:---|:---:|:---:|:---:|
| `update_decide` | 7.8 μs | 1.3 μs | **6.0x** |
| Decision agreement | — | — | **99.1%** |

---

## Method

METIS monitors the entropy of the model's next-token probability distribution in real time:

```
H_t = -Σ p_t(v) log p_t(v)    (token-level entropy)
H̄_t = α · H_t + (1-α) · H̄_{t-1}    (EWMA smoothing, α=0.3)
```

A CUSUM change-point detector identifies sustained entropy shifts. The routing decision:

| Condition | Route | Behavior |
|:---|:---|:---|
| H̄ < τ_low | **FAST** | Direct answer, no reasoning overhead |
| CUSUM alarm or H̄ > τ_high | **DEEP** | Full `<thinking>` deliberation |
| Otherwise | **NORMAL** | Standard generation |

The entropy computation and CUSUM update are implemented as a compiled **Rust module** (via PyO3), eliminating per-token Python interpreter overhead.

### KL-Sentinel Training Guard

During DPO fine-tuning, a sentinel monitors KL divergence between the policy and reference model. If D_KL > 0.15 nats, training automatically rolls back to the previous checkpoint, preventing catastrophic forgetting.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    METIS Framework                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Entropy  │───▶│  Adaptive    │───▶│  Cognitive   │  │
│  │  Monitor  │    │  Controller  │    │  Router      │  │
│  │ (Rust FFI)│    │  (CUSUM+     │    │ (FAST/NORMAL │  │
│  │           │    │   EWMA)      │    │  /DEEP)      │  │
│  └──────────┘    └──────────────┘    └──────┬───────┘  │
│                                             │          │
│  ┌──────────┐    ┌──────────────┐    ┌──────▼───────┐  │
│  │ Boundary  │    │  Curiosity   │    │  Inference   │  │
│  │  Probe    │◀──▶│  Driver      │    │  Engine      │  │
│  │(Semantic  │    │ (Knowledge   │    │ (CoT inject, │  │
│  │ Entropy)  │    │  Gap Track)  │    │  sampling)   │  │
│  └──────────┘    └──────────────┘    └──────────────┘  │
│                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Sentinel  │    │  Dreaming    │    │  Cognitive   │  │
│  │  (KL      │    │  Daemon      │    │  Reward      │  │
│  │  Guard)   │    │ (Night Train)│    │  (DPO/GRPO)  │  │
│  └──────────┘    └──────────────┘    └──────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Core Modules

| Module | Path | Description |
|:---|:---|:---|
| **Entropy Monitor** | `metis/core/controller.py` | EWMA + CUSUM + Cornish-Fisher adaptive thresholds |
| **Rust Native** | `metis/_native/src/` | O(1) statistics, 6x faster decision boundary |
| **Cognitive Router** | `metis/cognitive/switch.py` | FAST/NORMAL/DEEP routing logic |
| **CoT Injector** | `metis/cognitive/cot.py` | Dynamic `<thinking>` block injection at sentence boundaries |
| **Boundary Probe** | `metis/cognitive/boundary.py` | Semantic entropy via multi-sample clustering |
| **Curiosity Driver** | `metis/cognitive/curiosity.py` | Knowledge gap recording for self-evolution |
| **Inference Engine** | `metis/inference.py` | Full generation pipeline with RAG integration |
| **Sentinel** | `metis/sentinel.py` | KL-divergence gating for safe training |
| **Daemon** | `metis/daemon.py` | Idle-time autonomous training loop |
| **Serve** | `metis/serve.py` | OpenAI-compatible HTTP API with SSE streaming |

---

## Quick Start

### Install

```bash
git clone https://github.com/CARBON-XXX/METIS-Thinking-about-the-model.git
cd METIS-Thinking-about-the-model
pip install -r requirements.txt
```

### Attach to Any Model

```python
from metis import Metis, MetisInference

metis = Metis.attach(model, tokenizer)          # non-invasive, zero model modification
engine = MetisInference(metis)
result = engine.generate_cognitive("What is dark matter?", max_tokens=512)

print(result.text)
print(f"Route: {result.cognitive_route}")
print(f"Confidence: {result.avg_confidence:.0%}")
```

### Token-Level Monitoring

```python
metis.start_session("Explain quantum entanglement")
signal = metis.step(logits)       # CognitiveSignal per token

if signal.decision == Decision.DEEP:
    print(f"System 2 activated — H={signal.semantic_entropy:.2f}")

judgment = metis.introspect()     # post-generation cognitive assessment
gaps = metis.end_session()        # knowledge gaps for self-improvement
```

### CLI

```bash
python -m metis info              # system diagnostics
python -m metis attach MODEL_PATH # interactive REPL with telemetry
python -m metis serve MODEL_PATH  # OpenAI-compatible API server
```

---

## Training Pipeline

METIS uses a 3-stage training pipeline with information-theoretic rewards (no LLM-as-judge):

```
Stage 1: SFT Warmup
  └─ 1000 cognitively-structured examples (500 FAST + 500 DEEP)
  └─ Teaches <thinking> tag format and route compliance

Stage 2: DPO Alignment
  └─ 800 balanced preference pairs (400 FAST + 400 DEEP)
  └─ β=0.2, lr=2e-6, LoRA r=16
  └─ KL-Sentinel auto-rollback at D_KL > 0.15 nats

Stage 3: Evolutionary Self-Improvement (Optional)
  └─ Dreaming Daemon: idle-time training on recorded knowledge gaps
  └─ Sentinel gate: promote only if canary accuracy ≥ baseline
```

### Cognitive Reward Function

```
R(y) = R_format + R_route + R_correctness + R_efficiency
     = (tag_validity) + (route_alignment) + (answer_check) + (token_budget)
```

Ground-truth veto: incorrect answer → R_total = -1.0 regardless of cognitive quality.

---

## Reproducibility

All experiments run on a single **NVIDIA GB10** (Blackwell, 128 GB unified memory) using **vLLM** with PagedAttention.

### Reproduce the Paper Results

```bash
# Vector 1: Ablation (Rust vs Python latency + KL-Sentinel)
python tools/phase24_academic_benchmarks.py --vector 1

# Vector 2: Pareto Frontier (4 strategies × 100 questions)
python tools/phase24_academic_benchmarks.py --vector 2

# Vector 3: Generalization (GSM8K + TruthfulQA)
python tools/phase24_academic_benchmarks.py --vector 3

# Phase 24.2: Bimodal Distribution Proof
python tools/render_token_distribution.py

# Render Figures
python tools/render_pareto_frontier.py

# Compile Paper
bash tools/compile_paper.sh
```

### Data Artifacts

| File | Description |
|:---|:---|
| `paper/data/pareto.json` | Pareto frontier: 4 strategies, accuracy + tokens + timing |
| `paper/data/ablation.json` | Rust vs Python latency, KL-Sentinel simulation |
| `paper/data/generalization.json` | GSM8K + TruthfulQA cross-benchmark results |
| `paper/data/token_distribution.json` | Per-request token lengths + GMM fit + bimodality tests |
| `paper/figures/fig1_pareto_frontier.pdf` | Pareto frontier visualization |
| `paper/figures/fig2_bimodal_distribution.pdf` | Bimodal KDE + histogram with GMM decomposition |

---

## Paper

The preprint is available at [`paper/main.pdf`](paper/main.pdf) (7 pages).

To recompile:

```bash
bash tools/compile_paper.sh
# Output: paper/main.pdf
```

---

## Project Structure

```
metis/                    # Core framework
├── core/                 # Entropy monitor, adaptive controller, types
├── cognitive/            # Router, CoT injector, boundary probe, curiosity
├── integrations/         # LangChain, LlamaIndex, RAG adapter
├── pipeline/             # SFT/DPO trainer, evaluator, config
├── training/             # Rewards, benchmarks, tokenizer utils
├── search/               # Counterfactual search, entropy-guided tree
├── _native/              # Rust FFI (PyO3) — O(1) statistics engine
├── inference.py          # Production inference engine
├── sentinel.py           # KL-divergence training guard
├── daemon.py             # Autonomous dreaming training daemon
└── serve.py              # OpenAI-compatible HTTP API

tools/                    # Scripts for training, evaluation, paper figures
tests/                    # Unit and integration tests
paper/                    # LaTeX source, data JSONs, PDF figures
benchmarks/               # Benchmark evaluation harness
configs/                  # DGX training configs
docs/                     # Technical documentation
```

---

## Citation

```bibtex
@article{metis2026,
  title   = {METIS: Metacognitive Entropy-Triggered Inference Scaling},
  author  = {[AUTHOR_NAME_PENDING]},
  year    = {2026},
  url     = {https://github.com/CARBON-XXX/METIS-Thinking-about-the-model}
}
```

## License

[Apache-2.0](LICENSE)

---

<p align="center">
  <b>METIS</b> — Adaptive inference scaling through real-time entropy monitoring.
</p>

