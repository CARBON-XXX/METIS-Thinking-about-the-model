<div align="center">

# METIS

### Metacognitive Entropy-Triggered Inference Scaling

**1.3 μs per routing decision · O(1) amortized · Rust-native critical path**

[![Python](https://img.shields.io/badge/python-≥3.10-3776ab?style=flat-square)](https://python.org)
[![Rust](https://img.shields.io/badge/rust-PyO3_FFI-dea584?style=flat-square)](metis/_native/)
[![License](https://img.shields.io/badge/license-Apache_2.0-green?style=flat-square)](LICENSE)

**YiRui Li** · jasonuzi12@gmail.com

</div>

---

## System Architecture

```
                              ┌─────────────────────────────────────────────────────┐
                              │                    vLLM Engine                       │
                              │         PagedAttention · Continuous Batching         │
                              └──────────────────────┬──────────────────────────────┘
                                                     │  logits p(v|x₁..xₜ)
                                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         METIS CONTROL PLANE  (1.3 μs / decision)                   │
│                                                                                     │
│   ┌──────────────┐     ┌──────────────────┐     ┌──────────────────────────────┐   │
│   │  Shannon H(t) │────▶│  EWMA Low-Pass   │────▶│  Siegmund CUSUM Detector    │   │
│   │  -Σ p log p   │     │  α = 0.3         │     │  drift δ, threshold τ      │   │
│   └──────────────┘     └──────────────────┘     └─────────────┬────────────────┘   │
│          ▲                                                     │                    │
│          │                    Rust FFI (PyO3)                  │                    │
│          │              ┌──────────────────────┐               │                    │
│          │              │  O(1) Welford Stats   │               │                    │
│          │              │  Hybrid Recalibration │               │                    │
│          │              └──────────────────────┘               │                    │
│          │                                                     ▼                    │
│   ───────┴─────────────────────────────────────────────────────────────────────     │
│                              ROUTING DECISION                                       │
│          ┌──────────┬──────────────┬──────────────┬────────────────┐                │
│          │   FAST   │    NORMAL    │     DEEP     │      SEEK      │                │
│          │  H̄ < τ_L │  otherwise   │ CUSUM alarm  │ H̄ > τ_crit    │                │
│          │  ≤30 tok │  standard    │ + <thinking> │ after DEEP     │                │
│          │  direct  │  generation  │ full CoT     │ → external RAG │                │
│          └──────────┴──────────────┴──────────────┴────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                 ┌───────────────────────┬┴┬───────────────────────┐
                 ▼                       ▼ ▼                       ▼
          ┌────────────┐        ┌────────────────┐        ┌──────────────┐
          │  Gap Ledger │        │ KV-Cache Release│        │ KL-Sentinel  │
          │  epistemic  │        │ FAST path frees │        │ D_KL < 0.15  │
          │  gap archive│        │ blocks early    │        │ auto-rollback│
          └──────┬─────┘        └────────────────┘        └──────────────┘
                 │
                 ▼
          ┌────────────┐     ┌─────────────────┐
          │  Dreaming   │────▶│  Nightly DPO    │
          │  Daemon     │     │  20/80 blend    │
          │  GPU-idle   │     │  gap + anchor   │
          └────────────┘     └─────────────────┘
```

---

## Why METIS

现有推理扩展方案的核心矛盾：**简单问题浪费算力，复杂问题算力不足。**

Chain-of-Thought 对所有 query 强制推理。Self-Consistency 再乘以 *k* 倍。Semantic Entropy 需要 *K* 次完整生成 + O(K²) NLI。没有一个方案在生成的**同时**做决策。

METIS 在 decode 的**每一个 token** 上提取 Shannon 熵，经 EWMA 滤波 + CUSUM 变点检测，在 **1.3 μs** 内（p50）完成路由——不打断生成流水线，不额外调用模型，不增加端到端延迟。

---

## Hard Numbers

### Accuracy × Token Cost (Pareto Frontier)

| Strategy | Accuracy | Avg Tokens | Token Reduction vs SC |
|:---|:---:|:---:|:---:|
| Zero-Shot | 90.0% | 12.4 | — |
| Forced CoT | 91.0% | 156.0 | — |
| Self-Consistency *k*=5 | 94.0% | 864.5 | 1.0× |
| **METIS Dynamic** | **96.0%** | **53.4** | **16.2×** |

> 100-question mixed benchmark (50 GSM8K + 50 factual QA). METIS 路由 48 请求走 FAST (≤30 tok), 52 请求走 NORMAL/DEEP (50–149 tok).

### Routing Decision Latency

| Operation | Rust (p50) | Python (p50) | Speedup |
|:---|:---:|:---:|:---:|
| `update_decide` (per-token critical path) | **1.3 μs** | 7.8 μs | **6.0×** |
| `get_stats` (10K window) | 1.9 μs | 1.2 μs | — |

> 99.1% decision agreement. Rust: {NORMAL: 771, FAST: 222, DEEP: 7} vs Python: {NORMAL: 778, FAST: 220, DEEP: 2}.

### System-Level Tail Latency (ShareGPT, Poisson Arrival)

| QPS | Vanilla P99 TTFT | METIS P99 TTFT | Compression |
|:---|:---:|:---:|:---:|
| 1 | 1,705 ms | 1,269 ms | 1.3× |
| 4 | 1,967 ms | 1,575 ms | 1.2× |
| **8** | **5,372 ms** | **1,962 ms** | **2.7×** |
| 16 | 2,232 ms | 2,082 ms | 1.1× |

> QPS=8 时 Vanilla 因 KV-cache 饱和出现 P99 尾部暴涨。METIS 提前释放 FAST 路径的 PagedAttention blocks，将尾部延迟压缩 2.7×。

### Bimodal Distribution (非截断证明)

| Test | Statistic | Threshold | Verdict |
|:---|:---:|:---:|:---:|
| ΔBIC (Kass & Raftery) | 40.6 | >10 | **Bimodal** |
| Bimodality Coefficient | 0.754 | >0.555 | **Bimodal** |

> GMM 拟合：Peak 1 (FAST) μ₁=22.8 tok, σ₁=23.5, w₁=0.69 · Peak 2 (DEEP) μ₂=116.5 tok, σ₂=13.6, w₂=0.31

---

## The O(1) Pipeline

METIS 的核心信号链全部运行在 **O(1) 摊还时间**：

```
Token logits ──▶ Shannon entropy H(t)              O(|V|), 单次 softmax
           ──▶ EWMA H̄(t) = αH(t) + (1-α)H̄(t-1)  O(1), 一次乘加
           ──▶ CUSUM S⁺(t) = max(0, S⁺ + H̄ - μ₀ - δ)  O(1), 一次比较
           ──▶ Route decision                       O(1), 阈值判定
```

统计窗口使用 **Welford 在线算法**（mean/variance）+ **周期性 O(N) 重校准**（skew/kurtosis，每 100 步一次），避免浮点漂移的同时保持每步 O(1)。

整条链路编译为 Rust dylib，通过 PyO3 FFI 暴露，**零 Python 解释器开销**。

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/CARBON-XXX/METIS-Thinking-about-the-model.git
cd METIS-Thinking-about-the-model
pip install -r requirements.txt

# 2. Reproduce benchmarks
python tools/phase24_academic_benchmarks.py --vector 2   # Pareto frontier
python tools/phase24_academic_benchmarks.py --vector 3   # GSM8K + TruthfulQA generalization
python tools/phase24_academic_benchmarks.py --vector 1   # Rust vs Python ablation
```

> **Hardware:** All experiments ran on a single NVIDIA GB10 (128 GB unified). A consumer GPU ≥24 GB is sufficient for inference-only runs.

---

## Programmatic Usage

```python
from metis import Metis, MetisInference

metis = Metis.attach(model, tokenizer)
engine = MetisInference(metis)
result = engine.generate_cognitive("Solve: 3x + 2y = 16, x - y = 2", max_tokens=512)

print(result.text)                          # generated response
print(f"Route: {result.cognitive_route}")   # FAST / NORMAL / DEEP / SEEK
print(f"Entropy: {result.avg_entropy:.3f}") # mean EWMA entropy
```

```bash
python -m metis info              # system diagnostics
python -m metis attach MODEL_PATH # interactive REPL with entropy telemetry
python -m metis serve MODEL_PATH  # OpenAI-compatible API server (SSE streaming)
```

---

## Repository Structure

```
metis/
├── core/
│   ├── controller.py        EWMA + CUSUM + Cornish-Fisher adaptive thresholds
│   ├── entropy.py           Token-level Shannon entropy extraction
│   ├── statistics.py        Online statistical accumulators
│   └── types.py             InferenceResult, CognitiveSignal dataclasses
├── cognitive/
│   ├── switch.py            FAST / NORMAL / DEEP / SEEK routing state machine
│   ├── metacognition.py     Metacognitive orchestrator with lazy SE probe
│   ├── curiosity.py         Epistemic gap detection & archival
│   └── boundary.py          Decision boundary with semantic entropy fallback
├── _native/
│   ├── src/lib.rs           O(1) Welford stats, hybrid recalibration
│   ├── src/controller.rs    Rust-native EWMA + CUSUM mirror
│   └── Cargo.toml           PyO3 FFI build config
├── inference.py             Production pipeline: generate_cognitive() + RAG injection
├── sentinel.py              KL-divergence guard (D_KL > 0.15 → rollback)
├── daemon.py                Dreaming Daemon: GPU-idle detection → nightly DPO
└── serve.py                 OpenAI-compatible HTTP API with SSE streaming

tools/                       Benchmarks, stress tests, figure renderers
tests/                       Unit + integration tests
benchmarks/                  Evaluation harness
```

### Key Components

| Component | File | Complexity | What It Does |
|:---|:---|:---:|:---|
| Entropy Controller | `core/controller.py` | O(1)/tok | EWMA filter + CUSUM detector + route emission |
| Rust Engine | `_native/src/lib.rs` | O(1)/tok | Compiled mirror of controller, 6× faster |
| Cognitive Router | `cognitive/switch.py` | O(1) | 4-tier deterministic state machine |
| RAG Adapter | `integrations/rag_adapter.py` | — | SEEK route → web search → context injection |
| Dreaming Daemon | `daemon.py` | — | Nightly gap→trajectory→DPO with 20/80 blend |
| KL-Sentinel | `sentinel.py` | O(n) | 4-prompt canary KL check, auto-rollback |

---

## Cross-Model Scaling

METIS 的 CUSUM 检测器基于**熵漂移率**（一阶导）而非绝对熵值，因此天然适配不同规模的模型：

| Scale | Category | Mean H̄ | FAST% | DEEP% |
|:---|:---|:---:|:---:|:---:|
| 7B | Simple | 0.10 | 100% | 0% |
| 7B | Complex | 0.61 | 12% | 88% |
| 32B | Simple | 0.39 | 100% | 0% |
| 32B | Complex | 0.31 | 100% | 0% |

> 32B 模型的基线熵大幅下降（epistemic sharpening），所有复杂任务也变成低熵直答。部署时需按模型规模重校准阈值 τ_F / τ_D。

---

## SEEK Route: When Compute Is Not Enough

当 DEEP 路由后 H̄ 仍超过 τ_crit = 2.8（模型缺少回答所需的事实知识），METIS 中断生成并触发外部检索：

```
t=0─12   Normal decode      H̄ < 1.0    ───────────────▶  继续
t=13     "BASE" (实验名)     H̄ = 1.42   ───────────────▶  继续
t=19     "measured"          H̄ = 2.04   ───────────────▶  → DEEP
t=22     "approximately"     H̄ = 2.98   ▶ τ_crit breach ▶  → SEEK
t=23     [HALT] CUSUM S⁺ = 2.18 > τ = 2.0
         ├── Topic extraction → "proton magnetic moment Schneider 2022 BASE CERN"
         ├── Tool dispatch    → DuckDuckGo API → Nature (2022) result
         ├── Context inject   → <grounding_context> → KV-cache rebuild
         └── Resume decode    → H̄ drops 2.98 → 0.68 (−77%), completes in 89 tok
```

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
