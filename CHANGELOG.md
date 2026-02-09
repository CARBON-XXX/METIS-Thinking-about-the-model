# Changelog

## V10.0.0 (2025-02)

### Architecture Rewrite

- **Renamed from SEDAC to METIS** (Metacognitive Entropy-driven Thinking & Introspection System)
- Modular package: `metis/core/`, `metis/cognitive/`, `metis/integrations/`
- Full English localization of source code, documentation, and demo

### Core Features

- **Dual-System Reasoning**: Kahneman System 1 (fast) / System 2 (slow) switching via semantic entropy thresholds
- **Generation-Level Semantic Entropy**: Based on Kuhn et al. (ICLR 2023) — measures uncertainty across meaning, not just token probability
- **Epistemic Boundary Guard**: Four-action framework (GENERATE / HEDGE / SEEK / REFUSE)
- **Dynamic Chain-of-Thought**: Adaptive CoT injection with language-aware prompts (English & Chinese)
- **Thinking Protocol**: `<thinking>` block support with separate display of reasoning vs. final answer
- **Anti-Lazy Thinking**: Enforces deep reasoning within thinking blocks, rejects premature closure
- **Metacognitive Introspection**: Hallucination self-correction via entropy spike detection
- **Curiosity Driver**: Records knowledge gaps for autonomous future learning

### Demo & Tooling

- `demo_metis.py`: Interactive cognitive visualization with real-time entropy display
- `pyproject.toml`: Proper Python packaging (Apache-2.0)

---

## Pre-V10 History

Earlier versions (V5–V9) were developed under the name **SEDAC** (Semantic Entropy-Driven Adaptive Control), focusing on early-exit acceleration and cascade entropy probes. V10 marks a paradigm shift from token-level optimization to a full metacognitive architecture.
