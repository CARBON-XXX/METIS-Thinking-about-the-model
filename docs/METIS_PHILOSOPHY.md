# METIS: Giving Machines the Self-Awareness of Cognition
> **Metacognitive Entropy-driven Thinking & Introspection System**  
> *"Know thyself" — applied to artificial intelligence*

## 1. Why Do We Need METIS?

Despite the remarkable capabilities of modern Large Language Models (LLMs), they still suffer from a fatal Achilles' heel: **Overconfidence**.

At their core, traditional LLMs are statistical "prediction machines." They exhibit the same level of confidence when answering "What is the capital of France?" and "Who was the third mayor of the fictional city of Atlantis?" This **Metacognitive Deficit** leads to:
- **Hallucinations**: Confidently generating fabricated information.
- **Unreliability**: Users cannot distinguish when the model is "recalling facts" versus "probabilistic completion."
- **Lack of Boundaries**: The model does not know its own knowledge cutoff, and forces answers to questions beyond its capabilities.

METIS was created to solve this problem. It is not a new LLM, but rather a **Cognitive Layer** — a "conscience" and "judge" that attaches to the soul of an LLM.

---

## 2. The Name: From SEDAC to METIS

The project was originally called **SEDAC** (Semantic Entropy-driven Adaptive Control) — an engineering-focused description. Accurate, but incomplete.

**METIS (Μῆτις)** is the Greek Titaness of wisdom and mother of Athena. She embodies:
- **Cunning Intelligence**: Flexible, adaptive wisdom.
- **Forethought**: Deliberate, strategic thinking.
- **Transformative Power**: The ability to adapt and evolve.

The evolution from SEDAC to METIS marks an upgrade in our goal: from **"optimizing generation parameters"** to **"building a cognitive architecture."** We no longer focus solely on the probability distribution of tokens, but on the model's **Epistemic State** during generation.

---

## 3. Core Philosophy

### 3.1. Cognition over Generation
The traditional generation pipeline is `Input -> Model -> Output`.
METIS introduces an **Introspection Loop**:
`Input -> Cognition (System 1) -> [Metacognitive Judgment] -> (System 2 / Hedge / Refuse) -> Output`

In METIS's worldview, **"not answering" is sometimes more valuable than "answering."** When cognitive signals indicate extreme uncertainty, METIS chooses `REFUSE` or `HEDGE` rather than providing a wrong answer.

### 3.2. Honesty over Compliance
Models trained with RLHF (Reinforcement Learning from Human Feedback) tend toward sycophancy — telling users what they want to hear. METIS is the guardian of **Epistemic Honesty**.
It judges whether the model is fabricating or speculating by computing **Semantic Entropy** in real time — unlike simple token entropy, semantic entropy measures uncertainty of *meaning*.

### 3.3. Dynamic Compute Paths
The human brain does not spend the same energy thinking about "1+1=?" and "proving the Riemann hypothesis."
METIS implements Daniel Kahneman's **Dual Process Theory**:
- **System 1 (Fast Thinking)**: Intuition-based rapid generation, low entropy, low latency. (`Decision.FAST`)
- **System 2 (Slow Thinking)**: When high risk or high entropy is detected, deep reasoning is triggered. (`Decision.DEEP`)
    - Monte Carlo sampling (Semantic Entropy Sampling)
    - Chain-of-Thought injection (CoT Injection)
    - External tool invocation (Curiosity-driven Seek)

---

## 4. Architectural Significance: A Key Piece Toward AGI

AGI (Artificial General Intelligence) is not just about more parameters. It requires **Autonomy** and **Self-Correction**.

METIS provides a miniature AGI cognitive prototype:
1.  **Perception**: `SemanticEntropyComputer` senses the model's internal uncertainty.
2.  **Decision**: `AdaptiveController` dynamically determines the depth of reasoning.
3.  **Action**: `EpistemicBoundaryGuard` executes generation, refusal, or help-seeking.
4.  **Memory**: `CuriosityDriver` records knowledge gaps, enabling long-term cognitive evolution.

In this sense, METIS is the LLM's **Prefrontal Cortex** — responsible for monitoring, inhibiting, and regulating otherwise impulsive generation behavior.

---

## 5. Closing Thoughts

> *"Know thyself."* — Socrates

METIS is not about making models more powerful (that is the job of pretraining). It is about making models **more self-aware**. On the road to superintelligence, a god that does not know its own ignorance is more dangerous than a devil. METIS exists to prevent that future.
