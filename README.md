# METIS: Metacognitive Entropy-Triggered Inference Scaling

**YiRui Li** · Shanghai Qingpu World Foreign Language Senior High School · jasonuzi12@gmail.com

[![arXiv](https://img.shields.io/badge/arXiv-preprint-b31b1b?style=flat-square)](paper/main.pdf)
[![Python](https://img.shields.io/badge/python-≥3.10-3776ab?style=flat-square)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache_2.0-green?style=flat-square)](LICENSE)

---

## Abstract

METIS is a metacognitive inference-scaling framework that dynamically allocates computational depth based on real-time token-level entropy monitoring. Unlike fixed chain-of-thought prompting or uniform self-consistency sampling, METIS routes each query through one of four cognitive pathways — **FAST**, **NORMAL**, **DEEP**, or **SEEK** — determined by an EWMA-smoothed entropy signal processed through a Rust-native O(1) CUSUM change-point detector.

On a 100-question mixed math/logic benchmark, METIS achieves **96% accuracy at 53.4 average tokens**, Pareto-dominating zero-shot (90%, 12.4 tok), forced CoT (91%, 156.0 tok), and self-consistency *k*=5 (94%, 864.5 tok). The framework introduces a continuous self-improvement mechanism (the *Dreaming Daemon*) that archives high-entropy epistemic gaps during inference and performs nightly DPO updates, bounded by a KL-Sentinel safety valve (D_KL < 0.15 nats).

The full paper is available at [`paper/main.pdf`](paper/main.pdf).

---

## Quickstart (3 Steps)

### Step 1 — Clone and install dependencies

```bash
git clone https://github.com/CARBON-XXX/METIS-Thinking-about-the-model.git
cd METIS-Thinking-about-the-model
pip install -r requirements.txt
```

### Step 2 — Reproduce paper results

```bash
# Pareto frontier (Table 1): 4 strategies × 100 questions
python tools/phase24_academic_benchmarks.py --vector 2

# Generalization (Table 5): GSM8K + TruthfulQA
python tools/phase24_academic_benchmarks.py --vector 3

# Ablation (Table 2–4): Rust vs Python latency, KL-Sentinel
python tools/phase24_academic_benchmarks.py --vector 1
```

### Step 3 — Compile the paper

```bash
bash tools/compile_paper.sh    # → paper/main.pdf
```

> **Hardware note.** All experiments were conducted on a single NVIDIA GB10 (128 GB unified memory) with vLLM PagedAttention. A consumer GPU with ≥24 GB VRAM is sufficient for inference-only reproduction.

---

## Main Results

| Strategy | Accuracy | Avg Tokens | Speedup vs. SC |
|:---|:---:|:---:|:---:|
| Zero-Shot | 90% | 12.4 | — |
| Forced CoT | 91% | 156.0 | — |
| Self-Consistency (*k*=5) | 94% | 864.5 | 1.0× |
| **METIS** | **96%** | **53.4** | **16.2×** |

**Bimodal distribution proof:** ΔBIC = 40.6 (Kass & Raftery threshold: >10), Bimodality Coefficient = 0.754 (threshold: >0.555). METIS routing produces a genuine bimodal token distribution (μ₁ = 22.8 tok FAST, μ₂ = 116.5 tok DEEP), not a truncation artifact.

**Rust acceleration:** 6.0× latency reduction (7.8 μs → 1.3 μs per decision at p50), 99.1% decision agreement with the Python reference implementation.

---

## Method Overview

METIS computes token-level Shannon entropy H(t) from the next-token logits at each decoding step, applies an EWMA low-pass filter (α = 0.3) to extract the underlying epistemic signal, and feeds this into a CUSUM change-point detector. The four-tier routing logic:

| Condition | Route | Behavior |
|:---|:---|:---|
| H̄ < τ_low | **FAST** | Direct answer, no deliberation |
| CUSUM alarm or H̄ > τ_high | **DEEP** | Full `<thinking>` chain-of-thought |
| H̄ > τ_crit after DEEP | **SEEK** | External retrieval (RAG injection) |
| Otherwise | **NORMAL** | Standard generation |

The entropy monitor and CUSUM detector are implemented as a compiled Rust module (via PyO3), achieving O(1) amortized cost per token with zero Python interpreter overhead.

### Continuous Evolution

High-entropy queries (epistemic gaps) are archived during inference. A background *Dreaming Daemon* compiles these during idle cycles, generates reasoning trajectories, and performs nightly DPO updates using a 20/80 blend (20% gap-derived, 80% golden anchor). A *KL-Sentinel* halts training if D_KL > 0.15 nats, preventing catastrophic forgetting.

---

## Repository Structure

```
metis/                      Core framework
├── core/                   Entropy monitor, EWMA/CUSUM controller, type definitions
├── cognitive/              Cognitive router, CoT injector, boundary probe, curiosity driver
├── _native/                Rust FFI module (PyO3) — O(1) statistics engine
├── inference.py            Production inference pipeline with RAG integration
├── sentinel.py             KL-divergence training guard
├── daemon.py               Dreaming Daemon: autonomous nightly training loop
└── serve.py                OpenAI-compatible HTTP API with SSE streaming

tools/                      Evaluation scripts, figure rendering, paper compilation
tests/                      Unit and integration tests
paper/                      LaTeX source, experimental data (JSON), compiled PDF
benchmarks/                 Benchmark evaluation harness
```

### Key Source Files

| Component | Path | Description |
|:---|:---|:---|
| Entropy Controller | `metis/core/controller.py` | EWMA + CUSUM + Cornish-Fisher adaptive thresholds |
| Rust Engine | `metis/_native/src/lib.rs` | O(1) Welford statistics with periodic recalibration |
| Cognitive Router | `metis/cognitive/switch.py` | FAST / NORMAL / DEEP / SEEK routing logic |
| Inference Engine | `metis/inference.py` | End-to-end generation with entropy monitoring |
| Dreaming Daemon | `metis/daemon.py` | Gap archival → trajectory generation → DPO update |
| KL-Sentinel | `metis/sentinel.py` | KL-divergence gating with automatic rollback |
| Paper Benchmarks | `tools/phase24_academic_benchmarks.py` | All experimental vectors (Pareto, ablation, generalization) |

---

## Programmatic Usage

```python
from metis import Metis, MetisInference

metis = Metis.attach(model, tokenizer)
engine = MetisInference(metis)
result = engine.generate_cognitive("What is dark matter?", max_tokens=512)

print(result.text)                          # generated response
print(f"Route: {result.cognitive_route}")   # FAST / NORMAL / DEEP / SEEK
print(f"Confidence: {result.avg_confidence:.0%}")
```

```bash
python -m metis info              # system diagnostics
python -m metis attach MODEL_PATH # interactive REPL with entropy telemetry
python -m metis serve MODEL_PATH  # OpenAI-compatible API server
```

---

## Data Artifacts

All experimental data is stored as JSON for independent verification:

| Artifact | Path |
|:---|:---|
| Pareto frontier (4 strategies) | `paper/data/pareto.json` |
| Rust vs. Python latency ablation | `paper/data/ablation.json` |
| GSM8K + TruthfulQA generalization | `paper/data/generalization.json` |
| Token distribution + GMM fit | `paper/data/token_distribution.json` |
| Pareto frontier figure | `paper/figures/fig1_pareto_frontier.pdf` |
| Bimodal distribution figure | `paper/figures/fig2_bimodal_distribution.pdf` |

---

## Citation

```bibtex
@article{li2026metis,
  title   = {METIS: Metacognitive Entropy-Triggered Inference Scaling},
  author  = {Li, YiRui},
  year    = {2026},
  url     = {https://github.com/CARBON-XXX/METIS-Thinking-about-the-model}
}
```

## License

[Apache-2.0](LICENSE)
