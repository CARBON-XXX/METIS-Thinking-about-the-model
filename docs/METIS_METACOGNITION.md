# METIS: Metacognitive Architecture

> **METIS is not a tool — it is an extension of the LLM's self-awareness**

---

## Core Paradigm Shift

```
V9  (Tool view):       LLM → METIS(monitor) → User
V10 (Metacognition):   LLM ⟺ METIS(self)    → Cognitive Behavior
```

### What is Metacognition?

Metacognition = "Thinking about thinking"
- **Cognitive Monitoring**: Knowing what you know
- **Cognitive Regulation**: Adjusting strategies based on state
- **Cognitive Boundaries**: Recognizing capability limits

### METIS as AGI Metacognitive Layer

```
┌─────────────────────────────────────────────────────────┐
│               METIS Metacognitive Layer                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────┐ │
│  │  State Aware   │  │  Regulation   │  │  Boundary    │ │
│  │  (Perception)  │  │  (Control)    │  │  (Limits)    │ │
│  └───────┬───────┘  └───────┬───────┘  └──────┬──────┘ │
│          │                  │                  │        │
│          └──────────────────┼──────────────────┘        │
│                             ▼                           │
│                    ┌─────────────────┐                  │
│                    │  Meta-Decision  │                  │
│                    │                 │                  │
│                    └────────┬────────┘                  │
│                             │                           │
└─────────────────────────────┼───────────────────────────┘
                              ▼
                    ┌─────────────────┐
                    │  LLM Inference   │
                    └─────────────────┘
```

---

## Metacognitive State Space

No longer simple EXIT/NORM/O1, but a continuous cognitive state space:

### 1. Epistemic Confidence
```python
class EpistemicState:
    CERTAIN = "I know this for sure"     # High confidence, low entropy
    PROBABLE = "I believe this is right"  # Moderate confidence
    UNCERTAIN = "I'm not sure"            # Low confidence, high entropy
    UNKNOWN = "I don't know"              # Acknowledging ignorance
    CONFUSED = "The question is ambiguous" # Meta-confusion
```

### 2. Cognitive Load
```python
class CognitiveLoad:
    TRIVIAL = "This is simple"           # Low load
    MODERATE = "Requires some thought"   # Moderate load
    DEMANDING = "This is complex"        # High load
    OVERLOAD = "Beyond processing capacity" # Cognitive overload
```

### 3. Cognitive Boundary
```python
class CognitiveBoundary:
    WITHIN = "Within my capabilities"
    EDGE = "At the edge of my knowledge"
    BEYOND = "Beyond my capabilities"
    NEED_HELP = "Need external assistance"  # Triggers tool/retrieval
```

---

## Metacognitive Signals

### Signal Sources

1. **Entropy Dynamics**
   - Not absolute entropy values, but entropy *change patterns*
   - Entropy spike → Cognitive boundary
   - Entropy oscillation → Internal conflict
   - Entropy stable → Cognitive homeostasis

2. **Semantic Distance**
   - Distribution of top-k tokens in embedding space
   - Clustered → Confident (multiple similar options)
   - Dispersed → Uncertain (diverse semantic directions)

3. **Temporal Patterns**
   - Entropy autocorrelation
   - Periodic → Structured output (code)
   - Random → Creative/exploratory
   - Monotonically rising → Loss of control / hallucination

---

## Metacognitive Actions

Metacognition is not just monitoring — it must produce **behavior**:

### 1. Introspection
```
"Let me check my reasoning..."
"I need to reconsider this problem..."
```

### 2. Acknowledging Limits
```
"I'm not sure if this answer is correct"
"This is beyond my training data"
```

### 3. Seeking Help
```
Trigger: Retrieval Augmented Generation (RAG)
Trigger: Tool invocation
Trigger: Ask user for clarification
```

### 4. Strategy Switching
```
Fast thinking → Slow thinking (System 1 → System 2)
Generation → Verification
Exploration → Exploitation
```

---

## Implementation Architecture

```python
class MetacognitiveCore:
    """
    METIS Metacognitive Core
    
    Not an external tool, but an intrinsic component of the cognitive process.
    """
    
    def __init__(self):
        # State perception
        self.state_monitor = CognitiveStateMonitor()
        
        # Boundary detection
        self.boundary_detector = CognitiveBoundaryDetector()
        
        # Self-regulation
        self.regulator = CognitiveRegulator()
        
        # Meta-memory
        self.meta_memory = MetaMemory()
    
    def introspect(self, cognitive_trace: CognitiveTrace) -> MetaJudgment:
        """
        Introspection: Analyze own cognitive process.
        
        Returns:
            MetaJudgment: Metacognitive judgment
                - epistemic_state: Epistemic confidence level
                - cognitive_load: Cognitive load level
                - boundary_status: Boundary status
                - suggested_action: Recommended action
        """
        pass
    
    def regulate(self, judgment: MetaJudgment) -> CognitiveAction:
        """
        Regulation: Adjust behavior based on metacognitive judgment.
        """
        pass
```

---

## Relationship to AGI

METIS is the metacognitive infrastructure on the path to AGI:

| Capability | Current LLMs | + METIS |
|---|---|---|
| Self-awareness | ❌ | ✅ Knows what it doesn't know |
| Cognitive boundaries | ❌ | ✅ Identifies capability limits |
| Strategy adaptation | ❌ | ✅ Dynamic strategy switching |
| Help-seeking | Passive | ✅ Proactive help-seeking |
| Epistemic honesty | Hallucinations | ✅ Acknowledges uncertainty |

---

## Known Limitations & Open Challenges

Intellectual honesty demands that we acknowledge the current boundaries of this work:

### 1. Latency Cost of System 2

System 1 (token-level entropy) operates in real-time with negligible overhead. However, System 2 — generation-level semantic entropy — requires:
- **N forward-pass samples** (typically N=5) to generate diverse completions
- **Bidirectional entailment checking** via an NLI model to cluster semantically equivalent outputs
- **Entropy computation** over the resulting semantic clusters

In high-concurrency production scenarios, this multi-sample pipeline can introduce **5–10× latency** compared to a single greedy decode. The current architecture mitigates this through System 1 → System 2 cascading (only escalating when token-level entropy exceeds a threshold), but a rigorous **latency vs. accuracy Pareto analysis** across different workloads remains an open task. Future work should quantify the exact trade-off curve and explore approximation techniques (e.g., early-exit sampling, cached entailment).

### 2. NLI Model Dependency

The semantic clustering step in System 2 relies on a Natural Language Inference model (e.g., DeBERTa-large-MNLI) to judge whether two generated outputs are semantically equivalent. This introduces a **systemic single point of failure**:
- If the NLI model **misjudges entailment** (e.g., treats contradictory outputs as equivalent), the semantic entropy estimate collapses, and the metacognitive layer makes incorrect confidence assessments.
- The NLI model itself has known biases (lexical overlap heuristics, sensitivity to negation).

This is a fundamental architectural bottleneck. Potential mitigations include ensemble NLI, embedding-space clustering as a fallback, or training a lightweight task-specific entailment head — but none fully eliminate the dependency.

### 3. Lack of Empirical Validation

The current release presents **architecture and theory** but does not yet include quantitative benchmark results. The academic community rightfully prioritizes **empirical evidence** over design documents. Key missing evaluations:

| Benchmark | What it tests | Status |
|---|---|---|
| TruthfulQA | Hallucination detection | Planned |
| HaluEval | Hallucination classification | Planned |
| FactScore | Factual precision | Planned |
| SelfAware | "I don't know" calibration | Planned |

Until these benchmarks are run and reported with proper baselines (token entropy, P(True), verbalized confidence), the claims of this architecture remain **theoretically motivated but empirically unvalidated**. This is the most critical gap to address.

---

## Next Steps

1. Refactor `AdaptiveThresholdController` → `MetacognitiveCore`
2. Implement continuous representation of cognitive states
3. Implement metacognitive behavior triggers
4. Integrate into the inference loop as an intrinsic mechanism

---

*"The unexamined AI is not worth deploying."* — METIS
