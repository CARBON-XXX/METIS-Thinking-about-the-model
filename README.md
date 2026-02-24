<p align="center">
  <img src="https://img.shields.io/badge/METIS-v10.0.0-0969da?style=for-the-badge" alt="Version">
  <img src="https://img.shields.io/badge/python-≥3.9-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-≥2.0-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/license-Apache_2.0-green?style=for-the-badge" alt="License">
</p>

<h1 align="center">METIS</h1>

<p align="center">
  <b>Metacognitive Entropy-driven Thinking & Introspection System</b><br>
  <sub>A real-time cognitive layer that gives any LLM the ability to <i>know what it knows</i> — and what it doesn't.</sub>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> &nbsp;·&nbsp;
  <a href="#-core-capabilities">Capabilities</a> &nbsp;·&nbsp;
  <a href="#-integration-examples">Integration</a> &nbsp;·&nbsp;
  <a href="#-architecture">Architecture</a> &nbsp;·&nbsp;
  <a href="#-how-it-works">How It Works</a> &nbsp;·&nbsp;
  <a href="#-cognitive-reward-training">Training</a> &nbsp;·&nbsp;
  <a href="#-api-reference">API</a>
</p>

---

## What is METIS?

METIS is **not** a new model, a fine-tuning method, or a prompt trick.

It is a **metacognitive operating system** — a non-invasive signal-processing layer that attaches to any HuggingFace causal LM via a single forward hook and delivers real-time cognitive awareness:

| Capability | One-liner |
|:---|:---|
| **Dual-System Cognition** | System 1 / System 2 switching — greedy when confident, exploratory when uncertain |
| **Epistemic Boundary Guard** | CUSUM-based anti-hallucination: GENERATE → HEDGE → SEEK → REFUSE |
| **Dynamic Chain-of-Thought** | Inject `<thinking>` blocks when entropy signals demand deeper reasoning |
| **5-Phase Detection** | Classify tokens into FLUENT → RECALL → REASONING → EXPLORATION → CONFUSION |
| **Introspection** | Post-generation self-assessment: confidence, cognitive load, hallucination risk |
| **Curiosity Driver** | Record knowledge gaps at runtime for autonomous self-improvement |
| **Cognitive Rewards** | Information-theoretic DPO/GRPO/KTO training — no LLM-as-judge needed |

Every signal is computed **per-token in O(1)** from the model's own logit distribution. Zero extra inference. Zero model modification.

> *Named after Μῆτις — the Greek Titaness of wisdom and deep thought.*
> *"To know what you know and what you do not know — that is true knowledge." — Confucius*

---

## 🚀 Quick Start

### Install

```bash
git clone https://github.com/CARBON-XXX/METIS-Thinking-about-the-model.git
cd METIS-Thinking-about-the-model
pip install -r requirements.txt
```

### Minimal Example

```python
from metis import Metis, MetisInference

metis = Metis.attach(model, tokenizer)   # non-invasive, zero model modification
engine = MetisInference(metis)
result = engine.generate("What is dark matter?", max_tokens=512)

print(result.text)
print(f"Confidence: {result.avg_confidence:.0%}  |  System 2: {result.system2_ratio:.0%}")
print(f"Hedged: {result.was_hedged}  |  Refused: {result.was_refused}")
```

### Step-by-Step Monitoring

```python
from metis import Decision, BoundaryAction

metis.start_session("Explain quantum entanglement")
signal = metis.step(logits)   # returns CognitiveSignal with 13+ dimensions

if signal.decision == Decision.DEEP:
    print(f"System 2 — H={signal.semantic_entropy:.2f}, z={signal.z_score:+.2f}")

if signal.boundary_action == BoundaryAction.REFUSE:
    print("Knowledge boundary reached.")

judgment = metis.introspect()   # MetaJudgment
gaps = metis.end_session()      # list[KnowledgeGap]
```

### CLI

```bash
python -m metis info                                         # diagnostics
python -m metis attach --model Qwen/Qwen2.5-1.5B-Instruct   # interactive session
python -m metis experiment --n-prompts 300                   # full training run
```

### Interactive Demo

```bash
python demo_metis.py
```

```
[METIS think=OFF max=200]> What is 2+2?
  [  1] F FLU H=0.03 z=+0.00 ########## greedy  GENERATE 'The'
  [  2] F FLU H=0.01 z=+0.00 ########## greedy  GENERATE ' answer'
  [  3] F FLU H=0.00 z=+0.00 ########## greedy  GENERATE ' is'
  [  4] F FLU H=0.00 z=+0.00 ########## greedy  GENERATE ' 4'
```

| Command | Description |
|:---|:---|
| `/think` | Toggle Thinking Protocol |
| `/tokens N` | Set max generation tokens |
| `/examples` | Show built-in test questions |
| `/quit` | Exit |

---

## ✨ Core Capabilities

| # | Module | Method | Scope |
|:---:|:---|:---|:---|
| 1 | **Cognitive Switch** | Adaptive entropy thresholds + Cornish-Fisher + Bonferroni | LLM, Agent |
| 2 | **Boundary Guard** | sd-weighted CUSUM with surprise feedback | LLM, RAG |
| 3 | **Dynamic CoT** | CUSUM + momentum early-warning, 4 strategy modes | LLM, Agent |
| 4 | **Thinking Protocol** | Anti-Lazy enforcement, 64-token minimum | LLM |
| 5 | **Cognitive Sampling** | Greedy / Normal / Explore + adaptive repetition penalty | LLM |
| 6 | **Phase Detection** | Sliding-window z-score, self-calibrating thresholds | LLM, Training |
| 7 | **Predictive Signals** | Token surprise (−log₂ p), entropy gradient, momentum (EMA) | LLM, Agent |
| 8 | **Self-Correction** | Draft-Critique-Refine via 3-signal risk detection | LLM, Agent |
| 9 | **Introspection** | MetaJudgment: confidence, load, risk, stability, action | LLM, Agent |
| 10 | **Curiosity Driver** | Confusion → categorize → Dreaming Phase → targeted learning | Self-Improvement |
| 11 | **Cognitive Rewards** | 5-component reward: coherence + calibration + phase + epistemic + efficiency | Training |
| 12 | **Trace Export** | Per-token structured JSON, 13+ signal dimensions | Audit |

### Signal Stack

| Signal | Formula | Cost |
|:---|:---|:---:|
| Semantic Entropy (Sys 1) | $H_{\text{sem}} = H_{\text{shannon}} \times (1 + \lambda \cdot D_{\text{emb}})$ | O(1) |
| Semantic Entropy (Sys 2) | Kuhn et al. NLI clustering | O(N²) |
| Z-Score | $z = (H - \mu) / \max(\sigma, 0.15)$ | O(1) |
| Token Surprise | $S = -\log_2 p(\text{sampled})$ | O(1) |
| CUSUM | $S(t) = \max(0,\; S(t\!-\!1) + (z - k) \times sd)$ | O(1) |
| Cornish-Fisher | $z_p + \tfrac{1}{6}(z_p^2\!-\!1)\gamma_1 + \tfrac{1}{24}(z_p^3\!-\!3z_p)\gamma_2$ | O(1) |

---

## 🔌 Integration Examples

<details>
<summary><b>Standalone LLM</b></summary>

```python
from metis import Metis, MetisInference

metis = Metis.attach(model, tokenizer)
engine = MetisInference(metis)
result = engine.generate("What is dark matter?", max_tokens=512)
```
</details>

<details>
<summary><b>AI Agent (LangChain / CrewAI / Custom)</b></summary>

```python
metis = Metis.attach(agent.llm, agent.tokenizer)

for step in agent.plan:
    result = engine.generate(step.prompt)
    judgment = metis.introspect()

    if judgment.hallucination_risk > 0.3:
        agent.escalate_to_human(step)
    elif judgment.suggested_action == "verify":
        agent.use_tool("search", step.query)
    elif judgment.cognitive_load > 0.8:
        agent.decompose(step)
    else:
        agent.execute(result.text)
```
</details>

<details>
<summary><b>RAG — Intelligent Retrieval Gating</b></summary>

```python
from metis import EpistemicState

signal = metis.step(logits)

if signal.epistemic_state in (EpistemicState.UNCERTAIN, EpistemicState.UNKNOWN):
    docs = retriever.search(query)              # genuinely uncertain → retrieve
    result = engine.generate(f"{docs}\n{query}")
else:
    result = engine.generate(query)             # already knows → skip retrieval
```
</details>

<details>
<summary><b>Multi-Agent Epistemic Negotiation</b></summary>

```python
judgment_a = metis_a.introspect()
judgment_b = metis_b.introspect()

if abs(judgment_a.epistemic_confidence - judgment_b.epistemic_confidence) > 0.15:
    final = result_a if judgment_a.epistemic_confidence > judgment_b.epistemic_confidence else result_b
else:
    final = orchestrator.debate(result_a, result_b)
```
</details>

<details>
<summary><b>DPO / GRPO Cognitive Training</b></summary>

```python
from metis.training import CognitiveGRPO, PreferencePairGenerator

grpo = CognitiveGRPO(inference_fn=my_inference_fn)

for prompt in training_prompts:
    group = grpo.generate_group(prompt, n_samples=8)

pairs = PreferencePairGenerator().from_groups(groups)
pairs.export_dpo("cognitive_dpo_train.jsonl")   # TRL-compatible
```
</details>

---

## 🏗 Architecture

```
metis/
├── metis.py                  # Unified metacognitive core orchestrator
├── inference.py              # Cognitive-aware generation pipeline
│
├── core/                     # ── Signal Processing ──
│   ├── entropy.py            # Token-level semantic entropy (System 1, O(1)/token)
│   ├── semantic_entropy.py   # Generation-level SE (Kuhn et al., System 2)
│   ├── statistics.py         # Online statistics (mean/var/skew/kurtosis)
│   ├── controller.py         # Adaptive controller (AFF + CUSUM + Cornish-Fisher)
│   └── types.py              # 20+ data types
│
├── cognitive/                # ── Decision Layer ──
│   ├── switch.py             # System 1/2 mode switch
│   ├── boundary.py           # Epistemic boundary guard
│   ├── cot.py                # Dynamic CoT injection
│   ├── phase.py              # 5-phase cognitive detector
│   ├── curiosity.py          # Curiosity driver
│   └── metacognition.py      # Introspection & self-correction
│
├── training/                 # ── Reward & Training ──
│   ├── rewards.py            # 5-component cognitive reward
│   ├── grpo.py               # Cognitive GRPO
│   ├── dataset.py            # DPO/KTO pair generator
│   ├── generator.py          # METIS-instrumented generator
│   └── trl_adapter.py        # TRL adapter
│
├── integrations/hook.py      # Non-invasive PyTorch forward hook
└── _native/                  # Rust/PyO3 acceleration (WIP)
```

### Data Flow

```
                    ┌─────────────────────────────────────────────────┐
                    │          METIS Cognitive Operating System        │
                    │                                                 │
 LLM logits ──────▶│  SemanticEntropy ──▶ OnlineStatistics           │
                    │        │                    │                   │
                    │        └──────┬─────────────┘                   │
                    │               ▼                                 │
                    │       AdaptiveController                        │
                    │    (AFF + CUSUM + Cornish-Fisher)               │
                    │        │       │       │        │               │
                    │        ▼       ▼       ▼        ▼               │
                    │    Switch  Boundary   CoT    Phase              │
                    │    (S1/S2) (Guard)  (Inject) (Detect)           │
                    │        └───────┴───────┴────────┘               │
                    │                    │                            │
                    │         CognitiveSignal (13+ dim)               │
                    └────────────────────┼────────────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
             MetisInference      MetacognitiveCore     CuriosityDriver
             (sampling+CoT)      (introspection)       (knowledge gaps)
                    │                    │                    │
                    ▼                    ▼                    ▼
            InferenceResult        MetaJudgment         KnowledgeGap[]
```

---

## 🔬 How It Works

### 1. Dual-System Cognition

Inspired by Kahneman's *Thinking, Fast and Slow* (2011). The controller evaluates **4 independent criteria** with Bonferroni correction — DEEP requires ≥2 to fire:

1. **z-score test** — Is entropy statistically anomalous?
2. **Cornish-Fisher quantile** — Is entropy in the non-Gaussian tail?
3. **CUSUM alarm** — Has a distributional shift occurred?
4. **Trend analysis** — Is entropy consistently rising?

| System | Trigger | Sampling |
|:---:|:---|:---|
| **System 1 (FAST)** | Low entropy, high confidence | Greedy (argmax) |
| **System 2 (DEEP)** | ≥2 of 4 criteria fire | Exploratory + CoT |

### 2. Epistemic Boundary Guard

sd-weighted CUSUM control chart — the same method used in semiconductor manufacturing:

$$S(t) = \max\bigl(0,\; S(t\!-\!1) + (z - k) \times sd\bigr)$$

| Threshold | Action |
|:---:|:---|
| $S(t) \geq H_{\text{hedge}}$ | **HEDGE** — add uncertainty disclaimer |
| $S(t) \geq H_{\text{refuse}}$, conf > 0.3 | **SEEK** — trigger RAG |
| $S(t) \geq H_{\text{refuse}}$, conf < 0.3 | **REFUSE** — stop generation |

**Surprise feedback**: when token surprise > baseline, CUSUM receives a boost — catching confident hallucination.

### 3. Dynamic Chain-of-Thought

Two trigger paths:

- **Path A (Reactive)**: CUSUM on entropy z-scores → trigger `<thinking>` block
- **Path B (Predictive)**: Entropy momentum accumulation → early-warning before the main spike

Four strategies selected by cognitive state: **REFLECTION** (oscillation) · **DECOMPOSITION** (sustained difficulty) · **CLARIFICATION** (high diversity) · **STANDARD** (default).

### 4. Cognitive Phase Detection

Five phases via sliding-window z-score statistics, **self-calibrating** to each session:

```
FLUENT ──▶ RECALL ──▶ REASONING ──▶ EXPLORATION ──▶ CONFUSION
 (easy)    (memory)    (working)     (searching)     (stuck)
```

### 5. Metacognitive Introspection

Post-generation **MetaJudgment**:

| Metric | Range | Description |
|:---|:---:|:---|
| `epistemic_confidence` | [0, 1] | Weighted confidence + KNOWN/LIKELY ratio |
| `cognitive_load` | [0, 1] | DEEP ratio + z-score normalization |
| `hallucination_risk` | [0, 1] | High confidence × high z-score contradiction |
| `stability` | — | stable / volatile / chaotic |
| `suggested_action` | — | continue / verify / hedge / abort |

When `hallucination_risk > 0.3` → **Draft-Critique-Refine** pipeline activates automatically.

### 6. Curiosity Driver

Closed-loop self-improvement:

```
Runtime Confusion → KnowledgeGap Record → Dreaming Phase → Targeted Fine-tune → Verify
        ▲                                                            │
        └────────────────────────────────────────────────────────────┘
```

| Gap Category | Condition | Priority |
|:---|:---|:---|
| `complete_unknown` | Peak z > 3.0 | Critical |
| `sustained_confusion` | High-z ratio > 50% | High |
| `local_spike` | Brief z > 2.0 | Medium |

---

## 🎓 Cognitive Reward Training

### Why Not LLM-as-Judge?

| | Traditional RLHF | METIS Cognitive Rewards |
|:---|:---|:---|
| **Source** | Human / LLM judge | Information-theoretic signals |
| **Cost** | Extra LLM inference | Free (from existing trace) |
| **Determinism** | Stochastic | Same trace → same reward |
| **Interpretability** | Black box | 5 decomposable components |

### 5-Component Reward

$$R = w_1 R_{\text{coherence}} + w_2 R_{\text{calibration}} + w_3 R_{\text{phase}} + w_4 R_{\text{epistemic}} + w_5 R_{\text{efficiency}}$$

| Component | Weight | Measures |
|:---|:---:|:---|
| **R_coherence** | 0.20 | Entropy stability (smooth reasoning vs. erratic swings) |
| **R_calibration** | 0.30 | Confidence-surprise alignment |
| **R_phase** | 0.20 | Cognitive arc quality, penalize CONFUSION |
| **R_epistemic** | 0.15 | Appropriate uncertainty expression |
| **R_efficiency** | 0.15 | Don't overthink easy tasks |

### Anti-Reward-Hacking

- **Completeness bonus** — prevents early-EOS gaming
- **Length factor** — `min(1, n/40)` prevents ultra-short exploitation
- **Quality veto** — filters out "less bad as good" pairs
- **Homogeneous pairing** — same-temperature samples only

### Supported Frameworks

| Framework | Method | Export |
|:---|:---|:---|
| **GRPO** | Group Relative Policy Optimization | Ranked groups |
| **DPO** | Direct Preference Optimization | TRL JSONL |
| **KTO** | Kahneman-Tversky Optimization | Threshold labels |

---

## 📖 API Reference

### CognitiveSignal

Returned by `metis.step(logits)`:

```python
signal.semantic_entropy      # float — combined entropy (bits)
signal.token_entropy         # float — raw Shannon entropy
signal.semantic_diversity    # float — top-k embedding dispersion [0, 1]
signal.confidence            # float — max softmax probability [0, 1]
signal.decision              # Decision.FAST | NORMAL | DEEP
signal.epistemic_state       # EpistemicState.KNOWN | LIKELY | UNCERTAIN | UNKNOWN
signal.boundary_action       # BoundaryAction.GENERATE | HEDGE | SEEK | REFUSE
signal.z_score               # float — standardized entropy deviation
signal.token_surprise        # float — −log₂ p(sampled)
signal.entropy_gradient      # float — dH/dt
signal.entropy_momentum      # float — EMA of gradient
signal.cognitive_phase       # str — fluent/recall/reasoning/exploration/confusion
signal.cusum_alarm           # bool
signal.adaptive_thresholds   # tuple[float, float]
signal.introspection         # str — natural language explanation
```

### InferenceResult

Returned by `engine.generate()`:

```python
result.text                     # str
result.tokens_generated         # int
result.avg_confidence           # float
result.system2_ratio            # float
result.was_hedged               # bool
result.was_refused              # bool
result.boundary_interventions   # int
result.semantic_entropy_result  # Optional[SemanticEntropyResult]
```

### MetaJudgment

Returned by `metis.introspect()`:

```python
judgment.epistemic_confidence   # float [0, 1]
judgment.cognitive_load         # float [0, 1]
judgment.hallucination_risk     # float [0, 1]
judgment.stability              # "stable" | "volatile" | "chaotic"
judgment.suggested_action       # "continue" | "verify" | "hedge" | "abort"
judgment.reasoning              # str
```

---

## ⚙️ Configuration

### Key Thresholds

| Parameter | Default | Description |
|:---|:---:|:---|
| `SAFE_ENTROPY_THRESHOLD` | 0.6 | Entropy below this → always FAST |
| `Z_SCORE_STD_FLOOR` | 0.15 | Minimum σ for numerical stability |
| `CUSUM_K` / `CUSUM_HEDGE_H` | 0.5 / 8.0 | Boundary guard sensitivity |
| `COT_CUSUM_K` / `COT_CUSUM_H` | 0.3 / 4.0 | CoT trigger sensitivity |
| `COT_COOLDOWN_STEPS` | 40 | Min tokens between CoT injections |
| `MAX_COT_INJECTIONS` | 3 | Per-session CoT limit |
| `MIN_THINKING_TOKENS` | 64 | Anti-Lazy minimum |

### Training Defaults

| Parameter | Default | Description |
|:---|:---:|:---|
| `max_new_tokens` | 512 | Generation length per sample |
| `dpo_beta` | 0.1 | KL penalty |
| `lora_r` | 16 | LoRA rank |
| `n_samples` | 8 | Samples per prompt (GRPO) |
| `metis_stride` | 4 | Observation frequency (1 = every token) |

---

## 📊 Benchmarks

**Setup**: Qwen/Qwen2.5-1.5B-Instruct · 300 prompts × 8 samples × 3 DPO epochs · Consumer GPU

| Metric | Result |
|:---|:---|
| Confusion Ratio | 0.07–0.08 (CUSUM detector functional) |
| All 5 reward components | Active and differentiating |
| Latency | ~13–15s per sample (512 tokens) |
| Reward hacking | None detected (anti-gaming defenses verified) |

---

## 📚 References

- **Kuhn et al.** (ICLR 2023) — *Semantic Uncertainty*
- **Kahneman** (2011) — *Thinking, Fast and Slow*
- **Page** (1954) — *Continuous Inspection Schemes* (CUSUM)
- **Cornish & Fisher** (1938) — *Moments and Cumulants*
- **Li et al.** (2022) — *Contrastive Decoding*
- **Rafailov et al.** (NeurIPS 2023) — *Direct Preference Optimization*
- **Shao et al.** (2024) — *DeepSeek-R1* (GRPO)

---

## 🗂 Project Layout

```
METIS-Thinking-about-the-model/
├── metis/                    Core package
│   ├── core/                 Signal processing (entropy, statistics, controller)
│   ├── cognitive/            Decisions (switch, boundary, CoT, phase, curiosity)
│   ├── training/             Rewards, GRPO, DPO/KTO, generator
│   ├── integrations/         PyTorch forward hooks
│   └── _native/              Rust/PyO3 accelerators
├── run_experiment.py         Full 3-phase experiment runner
├── demo_metis.py             Interactive cognitive visualization
├── demo_reward.py            Cognitive reward demo
├── docs/                     Philosophy & technical docs
├── tests/                    Test suite
└── benchmarks/               Evaluation scripts
```

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

[Apache-2.0](LICENSE)

---

<p align="center">
  <b>METIS</b> — <i>Not making AI faster. Making AI wiser.</i><br>
  <sub>The first step toward AGI is not more parameters — it's self-awareness.</sub>
</p>
