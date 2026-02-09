# METIS: Technical Overview

> **Metacognitive Entropy-driven Thinking & Introspection System**
> Why it exists, what it solves, and how it works

---

## Table of Contents

1. [The Problem: Why METIS Exists](#1-the-problem-why-metis-exists)
2. [What METIS Solves](#2-what-metis-solves)
3. [Core Architecture](#3-core-architecture)
4. [Core Principle 1: Semantic Entropy as Cognitive Signal](#4-core-principle-1-semantic-entropy-as-cognitive-signal)
5. [Core Principle 2: Dual-Process Cognition (System 1 / System 2)](#5-core-principle-2-dual-process-cognition-system-1--system-2)
6. [Core Principle 3: Epistemic Boundary Guard](#6-core-principle-3-epistemic-boundary-guard)
7. [Core Principle 4: Adaptive Statistical Control](#7-core-principle-4-adaptive-statistical-control)
8. [Core Principle 5: Thinking Protocol with Metacognitive Regulation](#8-core-principle-5-thinking-protocol-with-metacognitive-regulation)
9. [Core Principle 6: Curiosity-Driven Self-Evolution](#9-core-principle-6-curiosity-driven-self-evolution)
10. [Core Principle 7: Metacognitive Introspection](#10-core-principle-7-metacognitive-introspection)
11. [System Integration: The Cognitive Pipeline](#11-system-integration-the-cognitive-pipeline)
12. [Theoretical Foundations](#12-theoretical-foundations)
13. [Positioning in the AI Landscape](#13-positioning-in-the-ai-landscape)

---

## 1. The Problem: Why METIS Exists

### 1.1 The Metacognitive Deficit of Large Language Models

Modern Large Language Models (LLMs) represent one of the most significant achievements in artificial intelligence. Models like GPT-4, Qwen, and LLaMA can generate fluent text, answer complex questions, write code, and reason about abstract problems. Yet they share a fundamental flaw: **they do not know what they do not know.**

When a human is asked "What is the capital of France?", they answer instantly and confidently — Paris. When asked "What was the third mayor of Atlantis?", they recognize the question is nonsensical and say so. This ability — **knowing the boundaries of one's own knowledge** — is called **metacognition**, literally "thinking about thinking."

LLMs lack this ability entirely. They answer both questions with equal statistical confidence. The model's internal probability distribution may assign 95% to "Paris" for the first question and, say, 30% to some fabricated name for the second — but the model has no mechanism to recognize that the second answer is qualitatively different from the first. The 30% probability is treated as "somewhat less certain" rather than "I am generating fiction."

This metacognitive deficit leads to three critical failure modes:

| Failure Mode | Description | Example |
|:---|:---|:---|
| **Hallucination** | Generating plausible but factually incorrect content with no hesitation | "Einstein won the Nobel Prize in Chemistry in 1975" |
| **Unreliability** | Users cannot distinguish model recall from probabilistic completion | "The study found that..." (fabricated citation) |
| **Boundary Blindness** | Model forces answers to questions beyond its competence | Answering future-prediction questions with equal confidence as historical facts |

### 1.2 Why Existing Solutions Fall Short

Several approaches have been proposed to address hallucination:

**Retrieval-Augmented Generation (RAG):** Provides external knowledge, but does not help the model recognize *when* it needs external knowledge. Without a metacognitive trigger, RAG systems either retrieve on every query (wasteful) or miss critical cases.

**RLHF / Constitutional AI:** Trains models to refuse harmful queries, but the refusal patterns are learned heuristics, not principled epistemic reasoning. Models learn "refuse questions about X" rather than "recognize when I lack knowledge."

**Confidence calibration:** Post-hoc calibration (e.g., Platt scaling) adjusts output probabilities, but operates on the final distribution, not on the model's internal cognitive state. It cannot distinguish "low confidence because I'm choosing between synonyms" from "low confidence because I'm fabricating."

### 1.3 The METIS Insight

METIS starts from a different premise: **the model's internal entropy signal already contains the information needed for metacognition — it just needs to be extracted and interpreted correctly.**

During generation, the softmax distribution over the vocabulary at each token position encodes the model's uncertainty. But raw token entropy is not sufficient — it conflates lexical diversity (multiple valid word choices) with epistemic uncertainty (the model genuinely does not know). METIS introduces **semantic entropy**: entropy weighted by the semantic diversity of candidate tokens, measured in the model's own embedding space.

This transforms the generation process from:

```
Input → Model → Output (blind)
```

To:

```
Input → Model → METIS Cognitive Layer → Output (self-aware)
         ↑              │
         └──── Regulation (think deeper / hedge / refuse)
```

METIS is not a new model. It is a **cognitive layer** — a non-invasive system that attaches to any existing LLM and gives it the ability to monitor, evaluate, and regulate its own generation process in real time.

---

## 2. What METIS Solves

METIS addresses five concrete problems:

### 2.1 Real-Time Hallucination Detection

At every token generation step, METIS computes a cognitive signal that quantifies how certain the model is about what it is generating. When the signal indicates the model is operating beyond its knowledge boundary, METIS can:

- **HEDGE**: Annotate the output with uncertainty markers ("I believe...", "approximately...")
- **SEEK**: Trigger external tools (RAG, web search, calculator) to verify or supplement
- **REFUSE**: Decline to answer when confidence is critically low

### 2.2 Adaptive Compute Allocation

Not all tokens require the same cognitive effort. METIS implements Kahneman's dual-process theory:

- **System 1 (FAST)**: For tokens where the model is confident → greedy decoding, minimal compute
- **System 2 (DEEP)**: For tokens where the model is uncertain → increased temperature, chain-of-thought reasoning, verification

This mirrors how humans think: answering "1+1=?" requires no deliberation, while proving a mathematical theorem requires sustained focus.

### 2.3 Structured Internal Reasoning

METIS can inject a `<thinking>...</thinking>` protocol that forces the model into an internal reasoning process before producing a final answer. Unlike static chain-of-thought prompting ("Let's think step by step"), METIS's thinking protocol is:

- **Dynamically triggered**: Only activated when cognitive signals indicate the query requires deep reasoning
- **Bounded**: Enforced token budget prevents infinite thinking loops
- **Monitored**: Repetition detection identifies when thinking is unproductive and force-closes it

### 2.4 Knowledge Gap Recording

The Curiosity Driver module records every instance where the model encounters high uncertainty. These "knowledge gaps" are logged with full context (query, entropy peak, surrounding tokens) for future targeted learning. This creates a closed-loop self-improvement cycle:

```
Runtime confusion → Record gap → Targeted training → Eliminate confusion
```

### 2.5 Post-Generation Metacognitive Assessment

After generation completes, the MetacognitiveCore module performs a full-trace analysis, producing a MetaJudgment report:

- **Epistemic Confidence**: How confident was the model across the entire response?
- **Cognitive Load**: How much System 2 reasoning was required?
- **Hallucination Risk**: Were there contradictory signals (high confidence + high entropy)?
- **Stability**: Was the generation process smooth or oscillating?

This report enables downstream systems (agents, UIs) to make informed decisions about how to present the output.

---

## 3. Core Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           METIS Cognitive Layer                             │
│                                                                             │
│  ┌───────────────────────┐                                                  │
│  │   SemanticEntropy     │  Per-token: H_sem = H_shannon × (1 + λ·D)       │
│  │   Computer            │  Measures: token entropy + semantic diversity    │
│  │   (core/entropy.py)   │  Output: (semantic_entropy, confidence, sd)     │
│  └───────────┬───────────┘                                                  │
│              │                                                              │
│              ▼                                                              │
│  ┌───────────────────────┐                                                  │
│  │   AdaptiveController  │  Signal processing: AFF, CUSUM, Cornish-Fisher  │
│  │   (core/controller.py)│  Output: Decision (FAST / NORMAL / DEEP)        │
│  │                       │  + z-score + dynamic thresholds                 │
│  └───────────┬───────────┘                                                  │
│              │                                                              │
│      ┌───────┼───────┬────────────────────────┐                             │
│      ▼       ▼       ▼                        ▼                             │
│  ┌────────┐ ┌──────────────┐ ┌─────────────┐ ┌──────────────┐              │
│  │Boundary│ │  Cognitive   │ │  Curiosity  │ │ Metacognitive│              │
│  │ Guard  │ │   Switch     │ │   Driver    │ │    Core      │              │
│  │        │ │              │ │             │ │              │              │
│  │GENERATE│ │ System 1/2   │ │ Knowledge   │ │ Post-gen     │              │
│  │HEDGE   │ │ allocation   │ │ gap logging │ │ introspection│              │
│  │SEEK    │ │              │ │             │ │              │              │
│  │REFUSE  │ │              │ │             │ │              │              │
│  └────────┘ └──────────────┘ └─────────────┘ └──────────────┘              │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                       MetisInference Pipeline                               │
│  Token-by-token generation with cognitive sampling, repetition detection,   │
│  thinking protocol, and boundary intervention                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module Responsibilities

| Module | File | Role |
|:---|:---|:---|
| **Metis** (Orchestrator) | `metis/metis.py` | Central coordinator — routes logits through entropy → controller → boundary → curiosity → switch → metacognition |
| **SemanticEntropyComputer** | `metis/core/entropy.py` | Computes per-token semantic entropy using Shannon entropy + embedding-space diversity |
| **AdaptiveController** | `metis/core/controller.py` | Transforms continuous entropy into discrete decisions via AFF, CUSUM, Cornish-Fisher |
| **SlidingWindowStats** | `metis/core/statistics.py` | Numerically stable online statistics (mean, std, skewness, kurtosis) over sliding window |
| **EpistemicBoundaryGuard** | `metis/cognitive/boundary.py` | Classifies epistemic state and decides boundary action (GENERATE / HEDGE / SEEK / REFUSE) |
| **CognitiveSwitch** | `metis/cognitive/switch.py` | Implements Kahneman's System 1/2 switching with oscillation detection |
| **CuriosityDriver** | `metis/cognitive/curiosity.py` | Records knowledge gaps for targeted offline learning |
| **MetacognitiveCore** | `metis/cognitive/metacognition.py` | Post-generation introspection producing MetaJudgment |
| **CoTManager** | `metis/cognitive/cot.py` | Context-aware chain-of-thought injection with strategy selection |
| **MetisInference** | `metis/inference.py` | Full generation pipeline integrating all cognitive components |

---

## 4. Core Principle 1: Semantic Entropy as Cognitive Signal

### 4.1 Beyond Shannon Entropy

Shannon entropy of the next-token distribution measures the "surprise" of the prediction:

$$H_{token} = -\sum_{i} p_i \log_2 p_i$$

But Shannon entropy is **semantically blind**. Consider two distributions:

| Scenario | Top candidates | Shannon H | Meaning |
|:---|:---|:---|:---|
| A | "happy" (50%), "glad" (50%) | 1.0 bit | Model **knows**: the emotion is positive |
| B | "cat" (50%), "table" (50%) | 1.0 bit | Model **confused**: two unrelated concepts |

Both have identical Shannon entropy, but Scenario A represents lexical choice (synonyms), while Scenario B represents genuine uncertainty (the model has no idea what comes next).

### 4.2 Semantic Diversity

METIS introduces a **semantic diversity** metric that measures the dispersion of top-k candidate tokens in the model's own embedding space:

$$D_{semantic} = 1 - \frac{\sum_{i<j} w_{ij} \cdot \cos(\mathbf{e}_i, \mathbf{e}_j)}{\sum_{i<j} w_{ij}}$$

Where:
- $\mathbf{e}_i$ is the normalized embedding of the i-th top token
- $w_{ij} = p_i \cdot p_j$ is the probability-weighted pair importance
- The sum is over all pairs of top-k tokens

Key design: **probability weighting** ensures that high-probability candidate pairs dominate the diversity score, while low-probability noise tokens contribute minimally.

### 4.3 Combined Semantic Entropy

The final cognitive signal combines both:

$$H_{semantic} = H_{token} \times (1 + \lambda \cdot D_{semantic})$$

Where $\lambda = 0.15$ is the semantic weight. This formula:
- **Amplifies** entropy when top candidates are semantically diverse (genuine confusion)
- **Suppresses** entropy when top candidates are synonyms (lexical choice)

### 4.4 Confidence

Alongside entropy, METIS tracks **confidence** as the maximum token probability:

$$c = \max_i p_i$$

Confidence is a complementary signal: high entropy with high confidence is unusual and may indicate a problematic distribution. Low entropy with low confidence is contradictory and may indicate quantization artifacts.

---

## 5. Core Principle 2: Dual-Process Cognition (System 1 / System 2)

### 5.1 Theoretical Basis

Daniel Kahneman's *Thinking, Fast and Slow* (2011) describes two cognitive systems in the human brain:

| | System 1 | System 2 |
|:---|:---|:---|
| **Nature** | Fast, automatic, intuitive | Slow, deliberate, analytical |
| **Effort** | Low | High |
| **Trigger** | Default mode | Activated by difficulty or surprise |
| **Examples** | "2+2=?" → 4 | "17×24=?" → (need to calculate) |

### 5.2 METIS Implementation

METIS maps this to three cognitive decisions based on real-time entropy signals:

```python
class Decision(Enum):
    FAST = "fast"       # System 1: low entropy → greedy decoding
    NORMAL = "normal"   # Standard: moderate entropy → standard sampling
    DEEP = "deep"       # System 2: high entropy → exploration + verification
```

The decision is made by the **AdaptiveController**, which uses:

1. **Z-score**: How unusual is the current entropy relative to the running distribution?
2. **CUSUM alarm**: Has a change-point been detected (entropy regime shift)?
3. **Empirical Bayes posterior**: What is the posterior probability of being in a "high-entropy state"?
4. **Consecutive run length**: How many consecutive tokens have been high-entropy?

### 5.3 Cognitive-Aware Sampling

Each decision maps to a different sampling strategy:

| Decision | Temperature | Behavior |
|:---|:---|:---|
| **FAST** | 0.0 | Pure greedy (argmax) — no exploration needed |
| **NORMAL** | user-set | Standard top-p sampling |
| **DEEP** | ≥ 0.4 (autonomous) | Elevated temperature for exploration + widened top-p |

Critically, DEEP mode temperature is **autonomous** — it does not depend on the user's base temperature setting. This ensures that the cognitive system's internal exploration is always active, even when the user requests greedy decoding.

---

## 6. Core Principle 3: Epistemic Boundary Guard

### 6.1 The Four Epistemic States

METIS classifies each token into one of four epistemic states:

```
KNOWN ─────── LIKELY ─────── UNCERTAIN ─────── UNKNOWN
  z < -0.5       -0.5 < z < θ₁      θ₁ < z < θ₂         z > θ₂
  c > 0.7        moderate c          low c                very low c
```

Where θ₁ (z_uncertain) and θ₂ (z_unknown) are **dynamically computed** thresholds that adapt to the model's entropy distribution using Cornish-Fisher expansion (accounting for skewness and kurtosis).

### 6.2 Boundary Actions

Each epistemic state maps to an action:

| State | Action | Meaning |
|:---|:---|:---|
| KNOWN | **GENERATE** | Proceed with normal generation |
| LIKELY | **GENERATE** | Proceed, minor uncertainty acceptable |
| UNCERTAIN | **HEDGE** | Generate but annotate with uncertainty |
| UNKNOWN | **SEEK** or **REFUSE** | Request external verification or decline to answer |

### 6.3 Multi-Signal Gating

To prevent false positives, the boundary guard uses multiple gates before triggering non-GENERATE actions:

1. **Warm-up protection**: First 20 tokens are always GENERATE (z-score unreliable with insufficient samples)
2. **Streak requirement**: HEDGE requires ≥3 consecutive high-z tokens (prevents single-token spikes at sentence boundaries)
3. **Confidence gate**: HEDGE in the uncertain zone requires confidence < 0.3 (higher confidence = structural uncertainty, not epistemic)
4. **Semantic diversity gate**: If top-k candidates are semantically similar (sd < 0.75), high entropy reflects lexical choice, not confusion
5. **Exponential decay accumulator**: Prevents early high-entropy tokens from permanently biasing the guard

---

## 7. Core Principle 4: Adaptive Statistical Control

### 7.1 The Challenge

Entropy distributions vary dramatically across models, quantization levels, languages, and query types. A fixed threshold (e.g., "entropy > 3.0 = uncertain") would fail across these variations. METIS requires thresholds that **self-calibrate** to any model's behavior.

### 7.2 Adaptive Forgetting Factor (AFF)

The controller maintains exponential moving averages (EMA) of entropy with an adaptive decay rate:

$$\lambda_t = \frac{\lambda_{base}}{1 + \alpha \cdot |e_t| / \sigma}$$

Where $e_t$ is the prediction error (observed entropy minus predicted). When the model's entropy deviates significantly from expectations, the forgetting factor decreases, allowing the EMA to adapt faster. During stable periods, the factor returns to the base value for smooth tracking.

### 7.3 Siegmund's Corrected CUSUM

For change-point detection (detecting when the model transitions between "confident" and "uncertain" regimes), METIS uses Siegmund's corrected CUSUM:

$$S^+_t = \max(0, S^+_{t-1} + z_t - k)$$

Alarm when $S^+ > h$, where:
$$h = \frac{\ln(2k^2 \cdot ARL_0 + 1)}{2k}$$

$ARL_0$ is the target average run length before false alarm (set to 200 tokens), providing theoretical false-alarm rate guarantees.

### 7.4 Cornish-Fisher Expansion

Standard z-score thresholds assume Gaussian-distributed entropy. In practice, entropy distributions are **skewed** (right-tailed) and **heavy-tailed** (occasional extreme values). METIS uses Cornish-Fisher expansion to compute non-Gaussian quantiles:

$$z_{adj} = z + \frac{(z^2 - 1)}{6}S + \frac{(z^3 - 3z)}{24}K - \frac{(2z^3 - 5z)}{36}S^2$$

Where $S$ is Fisher skewness and $K$ is Fisher excess kurtosis, both computed from a sliding window of recent entropy values using numerically stable Bessel-corrected estimators.

---

## 8. Core Principle 5: Thinking Protocol with Metacognitive Regulation

### 8.1 Design

The thinking protocol gives the model an explicit "internal monologue" space:

```
User: What is the integral of x²?

<thinking>
The integral of x² with respect to x...
Using the power rule: ∫x^n dx = x^(n+1)/(n+1) + C
So ∫x² dx = x³/3 + C
</thinking>

The integral of x² is x³/3 + C.
```

Content inside `<thinking>` tags is internal reasoning — displayed to the user in a dedicated frame but not part of the "official" answer.

### 8.2 Metacognitive Regulation

Unlike simple chain-of-thought prompting, METIS actively regulates the thinking process:

| Regulation | Mechanism | Purpose |
|:---|:---|:---|
| **Token Budget** | `max_thinking_tokens = 512` | Prevents infinite thinking loops |
| **Anti-Lazy Enforcement** | Suppress premature `</thinking>` if < 64 tokens | Ensures minimum reasoning depth |
| **Repetition Detection** | Jaccard + positional matching inside thinking | Detects semantic loops |
| **Forced Closure** | Inject `</thinking>` on budget exhaustion or repetition | Guarantees thinking block closure |
| **Single-Attempt Limit** | `thinking_failed` flag | Prevents recursive thinking trigger loops |

### 8.3 State Machine

The thinking protocol is a state machine with exhaustive exit coverage:

```
         ┌─────────────┐
         │   NORMAL     │ ──── repetition ────▶ inject <thinking>
         │  GENERATION  │                              │
         └──────────────┘                              ▼
                ▲                              ┌───────────────┐
                │                              │   THINKING     │
          close </thinking>                    │   (bounded)    │
                │                              └───────┬───────┘
                │                                      │
         ┌──────┴──────┐               repetition / budget / EOS
         │  THINKING   │                               │
         │  CLOSED     │ ◀─── force close ─────────────┘
         └─────────────┘
                │
          thinking_failed = True
                │
         ┌──────▼──────┐
         │  NORMAL     │ ──── repetition ────▶ FORCE STOP
         │  (no retry) │       (no thinking)    (circuit breaker)
         └─────────────┘
```

---

## 9. Core Principle 6: Curiosity-Driven Self-Evolution

### 9.1 Design Philosophy

An intelligent system should not only perform tasks but also **learn from its failures**. The CuriosityDriver implements this by recording "knowledge gaps" — instances where the model exhibited high uncertainty.

### 9.2 Gap Detection

During generation, the CuriosityDriver observes the z-score signal:

```python
def observe(self, entropy: float, z_score: float):
    self._observation_buffer.append((entropy, z_score))
    if z_score > self._gap_z_threshold * PEAK_Z_MULTIPLIER:
        # Peak confusion detected — record knowledge gap
```

### 9.3 Gap Categorization

Knowledge gaps are categorized by severity:

| Category | Criterion | Interpretation |
|:---|:---|:---|
| **Complete Unknown** | peak z > 3.0 | Model has no relevant knowledge |
| **Sustained Confusion** | high-z ratio > 50% | Model is consistently uncertain |
| **Local Spike** | peak z > 2.0, ratio < 50% | Specific sub-topic is unknown |

### 9.4 Self-Evolution Loop

```
     Runtime                              Offline ("Dreaming Phase")
     ───────                              ─────────────────────────
     Generate response                    Load knowledge gap records
          │                                        │
     Detect high entropy ────record────▶  Prioritize by severity
          │                                        │
     Continue generation                  Targeted fine-tuning
          │                                        │
     End session                          Update model weights
                                                   │
                                          Deploy updated model
                                                   │
     Next session ◀────────────────────────────────┘
```

---

## 10. Core Principle 7: Metacognitive Introspection

### 10.1 Post-Generation Analysis

After generation completes, the MetacognitiveCore consumes the full CognitiveTrace (all per-token signals) and produces a MetaJudgment:

```python
@dataclass
class MetaJudgment:
    epistemic_confidence: float   # Overall confidence (0-1)
    cognitive_load: float         # Processing difficulty (0-1)
    hallucination_risk: float     # Risk of fabrication (0-1)
    stability: str                # "stable" / "oscillating" / "degrading"
    boundary_status: str          # "safe" / "edge" / "violated"
    suggested_action: str         # "continue" / "verify" / "refuse" / "seek"
    reasoning: str                # Human-readable explanation
```

### 10.2 Key Metrics

All metrics are computed from **observed data**, with zero hardcoded thresholds:

- **Epistemic Confidence**: Based on the distribution of per-token confidence values and the ratio of KNOWN/LIKELY epistemic states
- **Cognitive Load**: Based on the ratio of DEEP decisions and the z-score distribution's upper tail
- **Hallucination Risk**: Detects contradictory signals — tokens where confidence is high but entropy is also high (the model is "confidently wrong")
- **Stability**: Analyzes the frequency of entropy_trend changes (frequent oscillation = unstable generation)

---

## 11. System Integration: The Cognitive Pipeline

### 11.1 Per-Token Processing Flow

For each generated token, the following pipeline executes in ~1ms on GPU:

```
1. Model forward pass → logits
                           │
2. SemanticEntropyComputer.compute(logits)
   → (semantic_entropy, token_entropy, semantic_diversity, confidence)
                           │
3. AdaptiveController.update(entropy, confidence)
   AdaptiveController.decide(entropy, confidence)
   → Decision (FAST / NORMAL / DEEP), z_score, dynamic thresholds
                           │
4. EpistemicBoundaryGuard.evaluate(signal, thresholds)
   → (EpistemicState, BoundaryAction, explanation)
                           │
5. CognitiveSwitch.process(signal)
   → SwitchResult (mode, compute_budget, strategy)
                           │
6. CuriosityDriver.observe(entropy, z_score)
   → (records knowledge gaps if detected)
                           │
7. CognitiveSignal assembled → returned to inference pipeline
                           │
8. Cognitive-aware sampling: temperature/top_p adjusted by Decision
   → next_token_id
```

### 11.2 Intervention Points

The inference pipeline can intervene at several points based on cognitive signals:

| Intervention | Trigger | Action |
|:---|:---|:---|
| **Cognitive Sampling** | Every token | Adjust temperature/top_p based on FAST/NORMAL/DEEP |
| **Thinking Injection** | Repetition detected | Inject `<thinking>` to break semantic loop |
| **Thinking Closure** | Budget / repetition / EOS | Force-inject `</thinking>` |
| **HEDGE annotation** | boundary_action = HEDGE | Mark output as uncertain |
| **REFUSE** | boundary_action = REFUSE | Decline to answer |
| **SEEK** | boundary_action = SEEK | Trigger external tool/RAG |
| **Force Stop** | 3+ repetition events | Terminate generation |

---

## 12. Theoretical Foundations

### 12.1 Information Theory

| Concept | Application in METIS |
|:---|:---|
| **Shannon Entropy** (Shannon, 1948) | Base measurement of next-token uncertainty |
| **Semantic Entropy** (Kuhn et al., ICLR 2023) | Generation-level uncertainty via entailment clustering — METIS uses a fast per-token approximation |
| **Mutual Information** | Implicit in semantic diversity: low MI between top-k tokens = synonyms |

### 12.2 Cognitive Science

| Theory | Application in METIS |
|:---|:---|
| **Dual-Process Theory** (Kahneman, 2011) | System 1 / System 2 cognitive switching |
| **Metacognition** (Flavell, 1979) | Self-monitoring of cognitive processes |
| **Epistemic Logic** | KNOWN / UNCERTAIN / UNKNOWN classification |
| **Curiosity-Driven Learning** (Schmidhuber, 1991) | Knowledge gap detection for self-improvement |

### 12.3 Signal Processing & Statistics

| Technique | Application in METIS |
|:---|:---|
| **Adaptive Forgetting Factor** (Fortescue et al., 1981) | Self-tuning EMA for non-stationary entropy tracking |
| **CUSUM** (Page, 1954; Siegmund, 1985) | Change-point detection for entropy regime shifts |
| **Cornish-Fisher Expansion** (1938) | Non-Gaussian quantile estimation for adaptive thresholds |
| **Fisher-Pearson Skewness** | Higher-order moment computation for distribution characterization |
| **Bessel Correction** | Unbiased variance estimation in sliding windows |
| **Bonferroni Correction** | Multiple-testing correction for multi-criteria DEEP decisions |

### 12.4 High-Dimensional Geometry

| Phenomenon | Relevance |
|:---|:---|
| **Concentration of Measure** | Cosine similarity degenerates in high-dim embedding spaces → requires probability-weighted aggregation |
| **Johnson-Lindenstrauss Lemma** | Theoretical basis for random projection as dimension reduction (future work) |

---

## 13. Positioning in the AI Landscape

### 13.1 METIS vs. Existing Approaches

| System | Approach | Limitation | METIS Advantage |
|:---|:---|:---|:---|
| **Standard LLM** | Generate without monitoring | No self-awareness | Real-time cognitive signals |
| **RAG** | Always retrieve | No trigger mechanism | Entropy-driven retrieval trigger |
| **RLHF** | Trained refusal patterns | Heuristic, not principled | Statistical epistemic reasoning |
| **Conformal Prediction** | Post-hoc confidence sets | Offline, not real-time | Per-token online monitoring |
| **Self-Consistency** (Wang et al.) | Sample multiple outputs, vote | O(N) compute cost | Single-pass, O(1) per token |
| **Thinking Models** (DeepSeek-R1, QwQ) | Trained thinking behavior | Requires specialized training | Works with ANY model, zero training |

### 13.2 Key Differentiators

1. **Model-Agnostic**: Attaches to any autoregressive LLM without modifying weights or training
2. **Real-Time**: Per-token processing in ~1ms, no multi-sample overhead
3. **Principled**: Based on information theory and statistical process control, not heuristics
4. **Self-Calibrating**: All thresholds adapt automatically to any model's entropy distribution
5. **Comprehensive**: Monitors, intervenes, records, and introspects — a complete cognitive layer

### 13.3 Toward AGI

METIS provides a miniature prototype of the cognitive architecture needed for AGI:

| AGI Requirement | METIS Implementation |
|:---|:---|
| **Self-Awareness** | Real-time entropy monitoring = "knowing what I'm doing" |
| **Self-Regulation** | Boundary guard + cognitive switching = "adjusting my behavior" |
| **Self-Improvement** | Curiosity driver = "learning from my mistakes" |
| **Self-Assessment** | Metacognitive core = "evaluating my performance" |

In this sense, METIS is not just an engineering tool — it is a step toward giving machines the cognitive self-awareness that defines intelligence.

---

*METIS — Named after Μῆτις, the Greek Titaness of wisdom, cunning counsel, and deep thought.*
*"To know what you know and what you do not know — that is true knowledge." — Confucius*
