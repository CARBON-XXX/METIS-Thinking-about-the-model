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

## Next Steps

1. Refactor `AdaptiveThresholdController` → `MetacognitiveCore`
2. Implement continuous representation of cognitive states
3. Implement metacognitive behavior triggers
4. Integrate into the inference loop as an intrinsic mechanism

---

*"The unexamined AI is not worth deploying."* — METIS
