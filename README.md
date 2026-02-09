<p align="center">
  <img src="https://img.shields.io/badge/METIS-v10.0-blue?style=for-the-badge" alt="Version">
  <img src="https://img.shields.io/badge/python-≥3.10-green?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-≥2.0-red?style=for-the-badge&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/license-MIT-orange?style=for-the-badge" alt="License">
</p>

<h1 align="center">METIS</h1>

<p align="center">
  <b>Metacognitive Entropy-driven Thinking & Introspection System</b><br>
  <i>Named after Μῆτις — the Greek Titaness of wisdom, cunning counsel, and deep thought</i>
</p>

<p align="center">
  <i>"To know what you know and what you do not know — that is true knowledge."</i> — Confucius
</p>

<p align="center">
  <a href="#-the-problem">The Problem</a> •
  <a href="#-key-innovations">Key Innovations</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-interactive-demo">Demo</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-how-it-works">How It Works</a> •
  <a href="#-signal-processing">Signal Processing</a> •
  <a href="#-api-reference">API</a>
</p>

---

## 🧠 The Problem

Modern Large Language Models are **blindly confident**. They answer *"What is the capital of France?"* and *"Who was the third mayor of Atlantis?"* with equal conviction. This **metacognitive deficit** leads to:

- **Hallucinations** — Generating plausible-sounding but factually wrong content with no hesitation.
- **Unreliability** — Users cannot distinguish when the model is "recalling facts" vs. "probabilistically completing text."
- **Lack of Boundaries** — Models don't know the edges of their own knowledge and will confidently answer questions far beyond their competence.

**METIS is not a new LLM.** It is a **cognitive layer** — a real-time metacognitive system that attaches to any existing language model and gives it the ability to *know what it's doing*. It is the difference between a machine that produces text and one that *thinks about what it's producing*.

---

## ✨ Key Innovations

| Capability | Description | Mechanism |
|:---|:---|:---|
| **Dual-System Cognition** | Kahneman's System 1/2 switching — know when to speak vs. think | Adaptive entropy thresholds with Cornish-Fisher calibration |
| **Epistemic Boundary Guard** | Real-time hallucination prevention — detect what the model doesn't know | Multi-signal z-score analysis with Bonferroni correction |
| **Dynamic Chain-of-Thought** | Context-aware reasoning injection — not template CoT, but adaptive thinking | CoTManager with strategy selection (Standard / Clarification / Decomposition / Reflection) |
| **Thinking Protocol** | Force deep `<thinking>...</thinking>` internal monologue with Anti-Lazy enforcement | System prompt engineering + premature closure detection + continuation injection |
| **Cognitive-Aware Sampling** | Every token's sampling strategy adapts to its cognitive state | Greedy for confident tokens, exploration for uncertain ones |
| **Hallucination Self-Correction** | Draft-Critique-Refine pipeline triggered by metacognitive risk assessment | MetacognitiveCore introspection → verification re-generation |
| **Curiosity Driver** | Autonomous knowledge gap recording for self-evolution | Runtime confusion detection → gap logging → targeted learning |
| **Metacognitive Introspection** | Post-generation self-assessment with actionable judgments | Full trace analysis producing epistemic confidence, cognitive load, hallucination risk |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/CARBON-XXX/METIS-Know-what-you-are-doing.git
cd METIS-Know-what-you-are-doing
pip install -r requirements.txt
```

### Basic Usage

```python
from metis import Metis, MetisInference, Decision, BoundaryAction

# Attach METIS to any HuggingFace model (non-invasive, zero model modification)
metis = Metis.attach(model, tokenizer)

# === Option A: Step-by-step cognitive monitoring ===
metis.start_session("Explain quantum entanglement")
signal = metis.step(logits)

if signal.decision == Decision.DEEP:
    # System 2 activated — model is uncertain, trigger deeper reasoning
    print(f"Entropy: {signal.semantic_entropy:.2f}, z-score: {signal.z_score:+.2f}")

if signal.boundary_action == BoundaryAction.REFUSE:
    # Model has hit its knowledge boundary — don't generate
    print("Knowledge boundary detected. Refusing to hallucinate.")

judgment = metis.introspect()   # Metacognitive self-assessment
gap = metis.end_session()       # Record knowledge gaps

# === Option B: Full inference pipeline (recommended) ===
engine = MetisInference(metis, on_token=my_streaming_callback)
result = engine.generate(
    "What is dark matter?",
    max_tokens=256,
    use_thinking_protocol=True,  # Enable <thinking> deep reasoning
)

print(result.text)
print(f"Hedged: {result.was_hedged}, Refused: {result.was_refused}")
print(f"Confidence: {result.avg_confidence:.1%}, Uncertainty: {result.uncertainty_score:.2f}")
```

---

## 🎮 Interactive Demo

```bash
python demo_metis.py
```

The interactive demo provides real-time visualization of METIS cognitive processing:

```
[METIS think=OFF max=200]> What is 2+2?

  >> METIS Cognitive Monitoring...
  [  1] F H=0.03 z=+0.00 ########## greedy GENERATE 'The'
  [  2] F H=0.01 z=+0.00 ########## greedy GENERATE ' answer'
  [  3] F H=0.00 z=+0.00 ########## greedy GENERATE ' is'
  [  4] F H=0.00 z=+0.00 ########## greedy GENERATE ' 4'
```

**Demo Commands:**

| Command | Description |
|:---|:---|
| `/think` | Toggle Thinking Protocol (enables `<thinking>` deep reasoning) |
| `/tokens N` | Set max generation tokens |
| `/examples` | Show built-in example questions |
| `/quit` | Exit |

You can also type a number (1-5) to run a built-in example, or type any question directly.

---

## 🏗 Architecture

```
metis/
├── __init__.py                    # Public API exports
├── metis.py                       # Metis — unified metacognitive core
├── inference.py                   # MetisInference — cognitive-aware generation pipeline
│
├── core/                          # Signal Processing Layer
│   ├── entropy.py                 # Token-level semantic entropy heuristic (System 1)
│   ├── semantic_entropy.py        # Generation-level SE (Kuhn et al. 2023, System 2)
│   ├── statistics.py              # Sliding-window online statistics (unbiased moments)
│   ├── controller.py              # Adaptive threshold controller (AFF + CUSUM + Cornish-Fisher)
│   └── types.py                   # Core data types and enums
│
├── cognitive/                     # Cognitive Layer
│   ├── switch.py                  # System 1/2 cognitive mode switch
│   ├── boundary.py                # Epistemic boundary guard (anti-hallucination)
│   ├── cot.py                     # Dynamic Chain-of-Thought injection manager
│   ├── curiosity.py               # Curiosity driver (knowledge gap recording)
│   └── metacognition.py           # MetacognitiveCore (introspection & regulation)
│
└── integrations/                  # Integration Layer
    └── hook.py                    # Non-invasive PyTorch forward hook
```

### Data Flow

```
Input Prompt
     │
     ▼
┌─────────────┐    logits     ┌──────────────────────────────────────────────┐
│  LLM Model  │──────────────▶│  METIS Cognitive Layer                      │
│  (any HF)   │               │                                              │
└─────────────┘               │  SemanticEntropyComputer ──▶ AdaptiveController
     ▲                        │         │                          │          │
     │                        │         ▼                          ▼          │
     │ sampling               │  CognitiveSwitch          BoundaryGuard      │
     │ intervention            │         │                          │          │
     │                        │         ▼                          ▼          │
     │                        │    CognitiveSignal { decision, boundary,     │
     │                        │                      entropy, confidence,    │
     │                        │                      z_score, introspection }│
     │                        └──────────────┬───────────────────────────────┘
     │                                       │
     │                                       ▼
     │                              ┌─────────────────┐
     └──────────────────────────────│ MetisInference   │
                                    │  • Cognitive      │
                                    │    Sampling       │
                                    │  • CoT Injection  │
                                    │  • Anti-Lazy      │
                                    │    Thinking       │
                                    │  • Boundary       │
                                    │    Actions        │
                                    └─────────────────┘
                                           │
                                           ▼
                                    InferenceResult
                                    { text, was_hedged,
                                      was_refused,
                                      introspection, ... }
```

---

## 🔬 How It Works

### 1. Cognitive-Aware Sampling

METIS **actively changes how each token is sampled** based on real-time cognitive signals. This is the core intervention point:

| Decision | Sampling Strategy | Rationale |
|:---:|:---|:---|
| **FAST** (System 1) | `greedy` (argmax) | Model is highly confident — take the best token directly |
| **NORMAL** | User-configured temp/top_p | Standard sampling, no intervention |
| **DEEP** (System 2) | `explore` (↑temp ×1.3, ↑top_p +0.1) | Model is uncertain — widen the search space |

Additionally, **entropy-aware logit sharpening** (inspired by contrastive decoding; Li et al., 2022) suppresses noisy long-tail tokens when `z_score > 1.0` and confidence is low:

```python
if signal.z_score > 1.0 and signal.confidence < 0.5:
    sharpness = 1.0 + 0.15 * min(signal.z_score - 1.0, 3.0)
    logits = logits * sharpness  # Sharpen distribution, reduce random walk
```

### 2. Dynamic Chain-of-Thought Injection

When METIS detects sustained uncertainty (3+ consecutive DEEP decisions, or ≥5/12 high z-score steps), it injects reasoning tokens directly into the KV cache. Unlike static CoT templates, METIS constructs **context-aware** prompts:

```
Model output: "...the speed of light is invariant in all reference frames"
                                    ↓ CoT triggered
Injected: "Wait — I said 'is invariant in all reference frames',
           but does that actually follow? Let me re-check."
```

**Strategy Selection Matrix:**

| Strategy | Trigger | Example Injection |
|:---|:---|:---|
| **STANDARD** | Generic high entropy | *"I'm not fully confident about '{context}'. Let me reconsider."* |
| **CLARIFICATION** | High semantic diversity + low confidence | *"What exactly does '{context}' mean in this context?"* |
| **DECOMPOSITION** | 5+ consecutive DEEP decisions | *"'{context}' is complex. Let me break it down step by step."* |
| **REFLECTION** | Decision oscillation (FAST↔DEEP switching) | *"I said '{context}', but does that actually follow?"* |

CoT injections update the model's KV cache and logits, ensuring the injected reasoning **actually influences** subsequent token predictions.

### 3. Thinking Protocol

When enabled (`use_thinking_protocol=True`), METIS forces the model into a deep reasoning mode using `<thinking>...</thinking>` tags:

- A system prompt enforces the thinking protocol
- `<thinking>` is injected at generation start
- **Anti-Lazy Thinking**: If the model tries to close `</thinking>` before generating at least 64 tokens of reasoning, METIS rolls back the closure and injects a continuation prompt
- A visualizer buffer hides internal rollback artifacts from the streaming output

### 4. Epistemic Boundary Guard

Prevents hallucination by detecting when the model is **outside its knowledge boundary**:

```
High confidence + Low entropy   →  KNOWN         →  GENERATE
Low confidence + High entropy   →  UNKNOWN       →  SEEK / REFUSE
Accumulated uncertainty         →  UNCERTAIN     →  HEDGE
High z + contradictory signals  →  HALLUCINATION →  HEDGE + flag
```

**REFUSE uses a grace period + consecutive threshold:**
- Early tokens (within first 8): REFUSE triggers immediately
- After the model has committed to an answer: requires 3 consecutive REFUSE signals
- This prevents false positives when the model is legitimately correcting a false premise (entropy spikes during correction are normal, not signs of ignorance)

### 5. Hallucination Self-Correction (G4)

When MetacognitiveCore detects `hallucination_risk > 0.3`:

1. **Draft-Critique**: Re-generate with a verification prompt that asks the model to scrutinize its own answer
2. **Compare**: Measure average confidence of original vs. corrected version
3. **Adopt**: Use the higher-confidence version (with a 10% relative improvement threshold)
4. **Budget**: Correction generation is capped at `min(max_correction_tokens, max_tokens)` to prevent runaway cost

### 6. Curiosity Driver

Records **knowledge gaps** for autonomous self-improvement:

```
Runtime Detection:
  high entropy event → record KnowledgeGap { query, entropy, timestamp }

SE Verification:
  SE confirms uncertainty → mark gap as verified (high priority)

Future Use:
  Dreaming Phase → targeted fine-tuning on recorded gaps
```

### 7. MetacognitiveCore — Introspection

After generation, MetacognitiveCore analyzes the full cognitive trace and produces a `MetaJudgment`:

| Metric | Computation | Range |
|:---|:---|:---|
| `epistemic_confidence` | Weighted blend of avg confidence + KNOWN/LIKELY ratio | 0.0 – 1.0 |
| `cognitive_load` | Proportion of DEEP decisions + mean z-score | 0.0 – 1.0 |
| `hallucination_risk` | Detection of contradictory signals (high confidence + high z-score simultaneously) | 0.0 – 1.0 |
| `stability` | Frequency of entropy trend changes | `"stable"` / `"volatile"` / `"chaotic"` |
| `suggested_action` | Decision-theoretic recommendation | `"continue"` / `"verify"` / `"hedge"` / `"abort"` |

---

## 📊 Signal Processing

### Token-Level Semantic Entropy (System 1)

A fast, per-token heuristic combining Shannon entropy with embedding-space diversity:

$$H_{\text{semantic}} = H_{\text{shannon}} \times (1 + \lambda \cdot D_{\text{embedding}})$$

Where $D_{\text{embedding}}$ measures top-k token dispersion in embedding space. This runs at O(1) per token and serves as the primary signal for System 1 decisions.

### Generation-Level Semantic Entropy (System 2)

The rigorous implementation of **Kuhn et al. (ICLR 2023)**:

1. Sample N complete generations for the same prompt
2. Compute pairwise bidirectional entailment (NLI)
3. Cluster into K semantic equivalence classes
4. Compute: $SE = -\sum_k p(C_k) \log_2 p(C_k)$

SE = 0 → all generations are semantically consistent (model is certain)
SE > 0 → semantic disagreement exists (model is genuinely uncertain)

### Adaptive Controller

| Component | Method | Purpose |
|:---|:---|:---|
| **Forgetting Factor** | $\lambda_t = \lambda_{\text{base}} / (1 + \alpha \cdot \|e\| / \sigma)$ | Adapt learning rate to prediction error magnitude |
| **Change Detection** | Siegmund-corrected CUSUM | Detect distributional shifts in entropy stream |
| **Threshold Calibration** | Cornish-Fisher expansion (skewness + kurtosis) | Non-Gaussian quantile estimation for dynamic thresholds |
| **Decision Logic** | Bonferroni-corrected multi-hypothesis (4 criteria, ≥2 required) | Control false positive rate across multiple signal tests |
| **Prior Estimation** | Empirical Bayes with Beta conjugate | Online prior updates for Bayesian decision framework |

---

## 📖 API Reference

### CognitiveSignal

Every `metis.step(logits)` returns a `CognitiveSignal`:

```python
signal.semantic_entropy     # float — combined semantic entropy (bits)
signal.token_entropy        # float — raw Shannon entropy (bits)
signal.semantic_diversity   # float — top-k embedding diversity [0, 1]
signal.confidence           # float — max softmax probability [0, 1]
signal.decision             # Decision.FAST / NORMAL / DEEP
signal.epistemic_state      # EpistemicState.KNOWN / LIKELY / UNCERTAIN / UNKNOWN
signal.boundary_action      # BoundaryAction.GENERATE / HEDGE / SEEK / REFUSE
signal.entropy_trend        # "rising" / "falling" / "stable" / "oscillating"
signal.introspection        # str — natural language self-explanation
signal.z_score              # float — standardized entropy deviation
signal.cusum_alarm          # bool — CUSUM change-point detection alarm
signal.adaptive_thresholds  # Tuple[float, float] — current (z_uncertain, z_unknown)
```

### InferenceResult

`engine.generate(...)` returns an `InferenceResult`:

```python
result.text                   # str — final generated text
result.tokens_generated       # int — total tokens generated
result.latency_ms             # float — generation time in milliseconds
result.avg_token_entropy      # float — mean entropy across all tokens
result.avg_confidence         # float — mean confidence across all tokens
result.uncertainty_score      # float — cumulative uncertainty metric
result.system1_ratio          # float — fraction of FAST decisions
result.system2_ratio          # float — fraction of DEEP decisions
result.was_hedged             # bool — whether uncertainty disclaimer was added
result.was_refused            # bool — whether generation was refused (knowledge boundary)
result.was_verified           # bool — whether System 2 SE verification ran
result.boundary_interventions # int — number of boundary events
result.introspection          # str — full cognitive introspection summary
result.semantic_entropy_result # Optional[SemanticEntropyResult] — System 2 SE details
```

### MetaJudgment

`metis.introspect()` returns a `MetaJudgment`:

```python
judgment.epistemic_confidence  # float — overall confidence [0, 1]
judgment.cognitive_load        # float — System 2 utilization [0, 1]
judgment.hallucination_risk    # float — contradictory signal score [0, 1]
judgment.stability             # str — "stable" / "volatile" / "chaotic"
judgment.boundary_status       # str — boundary guard status
judgment.suggested_action      # str — "continue" / "verify" / "hedge" / "abort"
judgment.reasoning             # str — natural language explanation
```

---

## ⚙️ Configuration

### Key Thresholds

| Parameter | Default | Description |
|:---|:---:|:---|
| `SAFE_ENTROPY_THRESHOLD` | 0.6 | Entropy below this is always FAST (prevents false positives) |
| `Z_SCORE_STD_FLOOR` | 0.15 | Minimum std for z-score calculation (numerical stability) |
| `COT_COOLDOWN_STEPS` | 15 | Minimum steps between CoT injections |
| `MAX_COT_INJECTIONS` | 3 | Maximum CoT injections per session |
| `REFUSE_GRACE_PERIOD` | 8 | Tokens before REFUSE can trigger immediately |
| `MIN_THINKING_TOKENS` | 64 | Minimum tokens before Anti-Lazy allows `</thinking>` closure |

---

## 📚 Academic References

- **Kuhn et al.** (ICLR 2023) — *"Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation"* — Foundation for generation-level semantic entropy
- **Kahneman, D.** (2011) — *"Thinking, Fast and Slow"* — Dual-process theory inspiring System 1/2 architecture
- **Li et al.** (2022) — *"Contrastive Decoding"* — Inspiration for entropy-aware logit sharpening
- **Page, E.S.** (1954) — *"Continuous Inspection Schemes"* — CUSUM change-point detection
- **Cornish & Fisher** (1938) — *"Moments and Cumulants"* — Non-Gaussian quantile approximation

---

## 🗂 Project Structure

```
METIS-Know-what-you-are-doing/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── demo_metis.py                  # Interactive cognitive visualization demo
├── metis/                         # Core METIS package
│   ├── core/                      # Signal processing layer
│   ├── cognitive/                 # Cognitive decision layer
│   ├── integrations/              # LLM integration hooks
│   └── inference.py               # Full inference pipeline
├── docs/                          # Design philosophy & technical docs
└── skills/                        # Windsurf skill definitions
```

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>METIS</b> — <i>Know what you are doing.</i>
</p>
