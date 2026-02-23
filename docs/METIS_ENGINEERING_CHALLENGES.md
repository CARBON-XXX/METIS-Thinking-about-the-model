 # METIS Engineering Challenges & Solutions

> **Metacognitive Entropy-driven Thinking & Introspection System**
> Technical Problem-Solving Documentation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Challenge 1: Adaptive Temperature Control Failure](#2-challenge-1-adaptive-temperature-control-failure)
3. [Challenge 2: Repetition Detection False Positives on CJK Text](#3-challenge-2-repetition-detection-false-positives-on-cjk-text)
4. [Challenge 3: Logit Sharpening Instability on Quantized Models](#4-challenge-3-logit-sharpening-instability-on-quantized-models)
5. [Challenge 4: Ghost KV Cache After Token Rollback](#5-challenge-4-ghost-kv-cache-after-token-rollback)
6. [Challenge 5: Monotonic Uncertainty Accumulation](#6-challenge-5-monotonic-uncertainty-accumulation)
7. [Challenge 6: Z-Score Inconsistency in Decision Controller](#7-challenge-6-z-score-inconsistency-in-decision-controller)
8. [Challenge 7: Thinking Protocol Stability on Non-Thinking Models](#8-challenge-7-thinking-protocol-stability-on-non-thinking-models)
9. [Challenge 8: Thinking Block Leakage Into Final Output](#9-challenge-8-thinking-block-leakage-into-final-output)
10. [Challenge 9: Recursive Thinking Trigger Loop](#10-challenge-9-recursive-thinking-trigger-loop)
11. [Challenge 10: Token Budget Exhaustion by Thinking](#11-challenge-10-token-budget-exhaustion-by-thinking)
12. [Challenge 11: False Epistemic Uncertainty on Lexical Diversity](#12-challenge-11-false-epistemic-uncertainty-on-lexical-diversity)
13. [Challenge 12: High-Dimensional Cosine Similarity Degeneracy](#13-challenge-12-high-dimensional-cosine-similarity-degeneracy)
14. [Summary of Mathematical & Theoretical Foundations](#14-summary-of-mathematical--theoretical-foundations)
15. [Lessons Learned](#15-lessons-learned)

---

## 1. Project Overview

METIS (Metacognitive Entropy-driven Thinking & Introspection System) is a real-time cognitive monitoring and intervention system that attaches to any autoregressive language model. It provides per-token cognitive signals — semantic entropy, confidence, epistemic state — and makes decisions about when the model should think deeper, hedge its claims, or refuse to answer.

The system was developed for the Qwen 2.5-7B model (4-bit quantized via bitsandbytes) and draws on Kahneman's dual-process theory (System 1 / System 2), signal processing (CUSUM change-point detection, Adaptive Forgetting Factor), and epistemic logic (boundary guard, knowledge gap detection).

### Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────┐
│  MetisInference Pipeline (inference.py)          │
│  Token-by-token generation + cognitive monitoring│
│                                                  │
│  ┌────────────┐  ┌──────────────┐               │
│  │ Metis Core ├──▶ Entropy      │               │
│  │ (metis.py) │  │ Computer     │               │
│  │            │  │(core/entropy)│               │
│  │  ┌────────┐│  └──────────────┘               │
│  │  │Adaptive││                                  │
│  │  │Control ││  ┌──────────────┐               │
│  │  │ler     ├┼──▶ Boundary    │               │
│  │  └────────┘│  │ Guard       │               │
│  │            │  │(cognitive/  │               │
│  │            │  │ boundary)   │               │
│  └────────────┘  └──────────────┘               │
│                                                  │
│  Cognitive Signals: H, z, confidence, decision   │
│  Interventions: HEDGE, SEEK, REFUSE, THINKING    │
└──────────────────────────────────────────────────┘
    │
    ▼
Final Output (with epistemic annotations)
```

### Core Modules

| Module | Responsibility |
|--------|---------------|
| `metis.py` | Central orchestrator — computes entropy, calls controller, evaluates boundary |
| `core/entropy.py` | Token-level semantic entropy heuristic: H = H_shannon × (1 + λ·D_embedding) |
| `core/controller.py` | Adaptive decision controller (AFF, CUSUM, Cornish-Fisher) |
| `cognitive/boundary.py` | Epistemic boundary guard (KNOWN / UNCERTAIN / UNKNOWN classification) |
| `inference.py` | Generation pipeline with cognitive sampling, repetition detection, thinking protocol |

---

## 2. Challenge 1: Adaptive Temperature Control Failure

**File:** `metis/inference.py` — `_cognitive_sample()`

### Problem Description

METIS implements cognitive-aware sampling: the sampling temperature is adjusted based on the cognitive decision (FAST / NORMAL / DEEP). In DEEP mode, the system should increase temperature to allow exploration of alternative generation paths. However, the implementation had a critical dependency:

```python
# BROKEN: DEEP mode depends on base_temperature > 0
elif signal.decision == Decision.DEEP:
    temperature = base_temperature + z_boost  # If base_temperature=0, this barely helps
```

The `generate()` method's default `temperature=0.0` propagated into `_cognitive_sample()` as `base_temperature`. With `base_temperature=0`, the DEEP mode formula produced `temperature = 0 + z_boost`, where `z_boost` was typically 0.0–0.4. This made DEEP mode nearly indistinguishable from FAST mode for most tokens.

### Root Cause Analysis

The design assumed `temperature` would be set by the user to a non-zero value. But the default was 0.0 (greedy decoding), and the demo hardcoded `temperature=0.0`. This meant the adaptive sampling system — a core feature of METIS — was completely disabled by default.

The deeper architectural issue: DEEP mode's exploration temperature should be an **autonomous** system parameter, not derived from a user-facing base temperature. The cognitive decision to "think deeper" is an internal metacognitive process and should not depend on user configuration.

### Solution

Made DEEP mode temperature autonomous — it computes its own exploration temperature based on the z-score signal, with a guaranteed minimum floor:

```python
elif signal.decision == Decision.DEEP:
    z_boost = min(max(signal.z_score - 1.0, 0.0) * 0.1, 0.4)
    temperature = max(0.4 + z_boost, base_temperature)  # Floor of 0.4
    top_p = min(base_top_p + 0.1, 0.95)
```

Key design decisions:
- **Floor of 0.4**: Ensures meaningful exploration even when `base_temperature=0`
- **z_boost scaling**: Higher z-scores (more uncertain) → higher temperature → more exploration
- **`max()` with base_temperature**: Never reduces temperature below user's explicit setting
- **top_p expansion**: Slightly widens the sampling nucleus for DEEP mode

Also removed the hardcoded `temperature=0.0` from the demo's `generate()` call, allowing the adaptive system to work as designed.

### Impact

Before: DEEP mode was functionally identical to greedy decoding (100% of the time for default settings).
After: DEEP mode provides genuine exploration with temperature 0.4–0.8, enabling the model to escape local optima and generate more diverse reasoning paths.

---

## 3. Challenge 2: Repetition Detection False Positives on CJK Text

**File:** `metis/inference.py` — `_detect_repetition_hybrid()`

### Problem Description

The repetition detection system uses Jaccard similarity to detect when the model is stuck in a loop. The original parameters were:
- Jaccard threshold: 0.45
- Minimum window size: 16 tokens

These caused massive false positives on Chinese text. Normal Chinese mathematical reasoning would be flagged as "repetitive," triggering unnecessary thinking injections that disrupted the model's output.

### Root Cause Analysis

Chinese (CJK) text has a much higher rate of shared functional characters compared to English. Characters like `的`, `是`, `了`, `在`, `和`, `不` appear in virtually every sentence. With a 16-token window, these shared functional characters alone could produce Jaccard similarity > 0.45, even when the actual content was completely different.

**Quantitative analysis:**
In a typical 16-token Chinese sentence, approximately 4–6 tokens are functional characters. Between two arbitrary sentences:
- Shared functional tokens: ~4
- Total unique tokens in union: ~24
- Jaccard similarity: 4/24 ≈ 0.17 (below threshold)

But in mathematical text (which naturally reuses variable names and operators):
- Shared tokens: ~8 (variables x, y, =, +, functional chars)
- Union: ~18
- Jaccard: 8/18 ≈ 0.44 → **false positive at 0.45 threshold!**

### Solution

Two adjustments:
1. **Jaccard threshold: 0.45 → 0.7**: Requires genuine token-set overlap, not just shared functional characters
2. **Minimum window: 16 → 32**: Longer windows dilute the impact of shared functional characters

Additionally, implemented a hybrid detection strategy:
- **Short patterns (< 32 tokens):** Strict positional matching (threshold 0.9) — catches exact loops without bag-of-words ambiguity
- **Long patterns (≥ 32 tokens):** Jaccard similarity (threshold 0.7) — catches semantic loops and rephrasing at paragraph level

### Mathematical Justification

For a window of size w, if functional characters appear with frequency f ≈ 0.3 in CJK text:

```
E[|A ∩ B|_functional] ≈ f × V_functional ≈ 0.3 × 20 = 6 tokens
E[|A ∪ B|] ≈ 2w - overlap ≈ 2×32 - 6 = 58
E[Jaccard_noise] ≈ 6/58 ≈ 0.10  (well below 0.7)
```

This ensures that the threshold is unreachable by random functional character overlap alone.

### Impact

False positive rate dropped from ~15% (on Chinese math text) to near 0%. Genuine repetition loops (e.g., model repeating the same paragraph verbatim) are still detected reliably because their Jaccard similarity is typically 0.85+.

---

## 4. Challenge 3: Logit Sharpening Instability on Quantized Models

**File:** `metis/inference.py` — `_cognitive_sample()`

### Problem Description

METIS applies a "logit sharpening" step when `temperature > 0` to make the probability distribution more peaked:

```python
logits = logits / temperature
```

When `temperature = 0` (greedy decoding), this becomes division by zero. The original code handled this with a guard, but a subtle edge case existed: when temperature was _exactly_ 0.0, the code still entered the sharpening path due to a floating-point comparison issue with the adaptive temperature computation.

### Root Cause Analysis

On 4-bit quantized models (bitsandbytes NF4), logit values have reduced precision. The logits tensor contains values like:

```
[12.375, 12.250, 12.125, ...]  # Top tokens very close in logit space
```

When these near-identical logits are divided by a very small temperature (e.g., 0.001 from rounding), the differences are amplified astronomically:

```
[12375.0, 12250.0, 12125.0, ...]  # After division by 0.001
```

After softmax, this produces a nearly one-hot distribution that flips unpredictably between the top candidates at every step — essentially random argmax on quantization noise.

### Solution

Skip logit sharpening entirely when `temperature ≤ 0`:

```python
if temperature > 0:
    logits = logits / temperature
    # ... top-p filtering, sampling
else:
    # Pure greedy: argmax
    next_token_id = logits.argmax(dim=-1).item()
```

### Theoretical Context

This is a manifestation of the **numerical instability of softmax with extreme temperature scaling** on low-precision arithmetic. In full-precision (FP32), the logit gap between top candidates is typically ~0.5–2.0, which is large relative to floating-point epsilon. But in INT4/NF4 quantization, effective precision is ~4 bits, giving a minimum representable gap of ~0.0625. Temperature scaling by 1/T with T→0 amplifies this quantization noise beyond the representable range.

---

## 5. Challenge 4: Ghost KV Cache After Token Rollback

**File:** `metis/inference.py` — Repetition intervention logic

### Problem Description

When METIS detects repetition, it rolls back the generated token sequence by trimming the repeated tail:

```python
generated_tokens = generated_tokens[:-rep_len]
```

However, the **KV cache** (past_key_values) was not updated to match. The KV cache is the transformer's "memory" of all previously processed tokens. After trimming the token list but leaving the KV cache intact, there was a divergence:

- `generated_tokens` reflected the trimmed sequence
- `past_key_values` still contained attention states for the trimmed tokens

This "ghost context" caused the model to hallucinate, producing outputs influenced by tokens that no longer existed in the generated sequence.

### Root Cause Analysis

The KV cache in transformer models is an append-only structure. Each new token's key-value pairs are concatenated to the cache:

```
KV_cache = [kv_token_1, kv_token_2, ..., kv_token_N]
```

After trimming `generated_tokens` from N to N-K, the cache still has N entries. The model's next forward pass uses all N entries for attention, effectively "seeing" tokens that were removed.

The correct approach is to **regenerate** the KV cache from scratch using only the retained tokens:

```python
full_input = torch.cat([prompt_ids, gen_ids_trimmed], dim=1)
clean_out = model(input_ids=full_input, use_cache=True, return_dict=True)
past_key_values = clean_out.past_key_values  # Clean cache
```

### Solution

After every token rollback (repetition trimming, thinking block closure, etc.), the KV cache is rebuilt from the remaining tokens:

```python
generated_tokens = generated_tokens[:-rep_len]  # Trim
# Rebuild KV cache without ghost context
if generated_tokens:
    gen_ids = torch.tensor([generated_tokens], device=model.device)
    full_input = torch.cat([prompt_ids, gen_ids], dim=1)
else:
    full_input = prompt_ids
clean_out = model(input_ids=full_input, use_cache=True, return_dict=True)
past_key_values = clean_out.past_key_values
logits = clean_out.logits[:, -1, :]
```

### Performance Cost

Rebuilding the KV cache requires a full forward pass over all retained tokens — O(N²) in the sequence length due to self-attention. For typical sequences (< 2048 tokens), this takes 50–200ms on GPU. This is acceptable because rollback events are rare (typically 0–2 per generation).

### Impact

Before: After repetition rollback, the model would generate semantically incoherent text influenced by "phantom" removed tokens.
After: Clean context allows the model to generate fresh, coherent continuations.

---

## 6. Challenge 5: Monotonic Uncertainty Accumulation

**File:** `metis/cognitive/boundary.py` — `EpistemicBoundaryGuard`

### Problem Description

The boundary guard maintains an `uncertainty_accumulator` that tracks cumulative uncertainty over the generation. Originally, this accumulator only increased:

```python
if z > z_unc:
    self._uncertainty_accumulator += z
```

This meant that early high-entropy tokens (common during the first 20–50 tokens as the model establishes context) would permanently taint the uncertainty score. After the model settled into confident generation, the accumulated score remained high, causing the boundary guard to trigger HEDGE actions on tokens that were objectively certain.

### Root Cause Analysis

The accumulator violated a fundamental principle of online signal processing: **recent observations should carry more weight than distant ones**. The original design treated all uncertainty events equally regardless of when they occurred, creating a "ratchet effect" where uncertainty could only increase.

In practice, language model generation follows a characteristic entropy trajectory:
1. **High entropy** in the first 10–30 tokens (establishing topic, style)
2. **Moderate entropy** during content generation
3. **Low entropy** at the end (concluding statements)

Without decay, the initial high-entropy phase permanently biased the guard toward HEDGE/REFUSE actions.

### Solution

Added exponential decay to the uncertainty accumulator:

```python
self._uncertainty_decay = 0.995  # ~200-token effective window
# In evaluate():
self._uncertainty_accumulator *= self._uncertainty_decay
if z > z_unc:
    self._uncertainty_accumulator += z
```

### Mathematical Analysis

The decay factor λ = 0.995 creates an effective memory window:

```
Effective window ≈ 1 / (1 - λ) = 1 / 0.005 = 200 tokens
```

After 200 tokens, a past uncertainty event's contribution decays to:

```
0.995^200 ≈ 0.367 (1/e) of its original value
```

After 500 tokens:

```
0.995^500 ≈ 0.082 → effectively forgotten
```

This ensures the boundary guard reflects the model's **current** confidence state, not its entire history.

---

## 7. Challenge 6: Z-Score Inconsistency in Decision Controller

**File:** `metis/core/controller.py` — `AdaptiveController.get_z_score()`

### Problem Description

The `AdaptiveController` computes z-scores internally for its CUSUM and decision logic. An external method `get_z_score()` was provided for the boundary guard and visualization to use. However, the external method applied a **low-entropy clamping** step that the internal logic did not:

```python
def get_z_score(self, entropy):
    # This clamping was NOT present in internal decide()
    if entropy < self._cold_start_entropy_mean * 0.1:
        return 0.0  # Clamp low entropy to "normal"
    # ... normal z-score computation
```

This meant the boundary guard saw different z-scores than the decision controller, leading to contradictory states: the controller might decide DEEP (based on unclamped z), while the boundary guard saw z=0 (clamped) and decided KNOWN.

### Root Cause Analysis

The clamping was originally added to suppress noise during low-entropy tokens (where the z-score formula `(x - μ) / σ` produces large negative values). But it was applied inconsistently — only in the external API, not in the internal decision logic.

### Solution

Removed the clamping and used the same cached z-score that the internal decision logic uses:

```python
def get_z_score(self, entropy):
    # Use the same z-score that decide() computed
    return self._last_z_score
```

This guarantees perfect consistency: the boundary guard, visualization, and decision controller all see the same z-score for each token.

---

## 8. Challenge 7: Thinking Protocol Stability on Non-Thinking Models

**File:** `metis/inference.py` — Thinking protocol implementation

### Problem Description

METIS implements a `<thinking>...</thinking>` protocol that allows the model to reason internally before producing a final answer. However, the Qwen 2.5-7B model was **not trained** with this protocol. When forced into thinking mode, the model exhibited several failure modes:

1. **Repetitive thinking**: The model enters a semantic loop, repeating the same reasoning pattern
2. **Infinite thinking**: Without a token budget, the model never closes the `</thinking>` tag
3. **Lazy thinking**: The model immediately outputs `</thinking>` without any reasoning

### Root Cause Analysis

Models not trained on thinking protocols have no learned representation of the `<thinking>` / `</thinking>` tag semantics. From the model's perspective, these are arbitrary text tokens. The model cannot:

- Understand that content inside `<thinking>` tags is "internal" and shouldn't be verbose
- Learn when to terminate thinking and start answering
- Structure its reasoning within the thinking block

This is fundamentally different from models like DeepSeek-R1 or QwQ, which are specifically trained to use thinking tags as a structured reasoning format.

### Solution

Implemented a multi-layered safety system:

#### Layer 1: Token Budget (`max_thinking_tokens = 512`)
Force-close the thinking block after a maximum number of tokens:

```python
if is_thinking and (step - thinking_start_step) >= self._max_thinking_tokens:
    # Inject </thinking> and rebuild KV cache
    close_tag = "\n</thinking>\n"
    # ... inject into generated_tokens, feed to model, continue
```

#### Layer 2: Anti-Lazy Enforcement (`min_thinking_tokens = 64`)
If the model tries to close thinking before generating enough tokens, suppress the closing tag:

```python
if is_thinking:
    recent = tokenizer.decode(generated_tokens[-12:])
    if "</thinking>" in recent and (step - thinking_start_step) < min_thinking_tokens:
        # Roll back the closing tag, rebuild KV cache, re-sample
```

#### Layer 3: Repetition-Aware Forced Closure
If repetition is detected inside thinking, force-close instead of attempting token-banning (which fails because the model uses synonyms to repeat semantically):

```python
# Escalation 2: Repetition INSIDE thinking
thinking_failed = True
generated_tokens = generated_tokens[:-rep_len]
# Rebuild KV cache, inject </thinking>, let model answer
```

### Impact

Before: Model could loop indefinitely in thinking, consuming the entire token budget.
After: Thinking is bounded, monitored, and forcefully closed when unproductive.

---

## 9. Challenge 8: Thinking Block Leakage Into Final Output

**File:** `metis/inference.py` — Generation loop exit paths

### Problem Description

When the generation loop terminated (by max_tokens, EOS token, or force-stop), the `is_thinking` flag could still be `True`, meaning the `</thinking>` tag was never injected. The downstream `_split_thinking()` function relied on matching `<thinking>...</thinking>` pairs. An unclosed block meant the entire output — thinking content included — leaked into the final answer.

### Root Cause Analysis

There were **four** exit paths from the generation loop, but only one originally handled the thinking state:

1. ✅ **EOS token**: Could happen inside thinking → tag not closed
2. ✅ **max_tokens exhaustion**: Loop ends naturally → tag not closed
3. ✅ **Force-stop (repetition events ≥ 3)**: `break` statement → tag not closed
4. ✅ **Thinking budget exhaustion**: Already handled (closes tag before continuing)

Exit paths 1–3 all left `is_thinking = True` without injecting `</thinking>`.

### Solution

Added a catch-all check at every exit point:

```python
# Force-stop break:
if repetition_events >= _REP_FORCE_STOP:
    if is_thinking:
        # Inject </thinking> before breaking
        close_tag = "\n</thinking>\n"
        close_tag_ids = tokenizer.encode(close_tag, add_special_tokens=False)
        for tid in close_tag_ids:
            generated_tokens.append(tid)
    break

# After generation loop (covers max_tokens and EOS):
if is_thinking:
    close_tag = "\n</thinking>\n"
    close_ids = tokenizer.encode(close_tag, add_special_tokens=False)
    for tid in close_ids:
        generated_tokens.append(tid)
```

### Design Principle

This follows the **RAII (Resource Acquisition Is Initialization)** pattern adapted for generation:
- **Opening** a thinking block is an "acquisition" (allocates cognitive context)
- **Closing** it is a "release" (returns to normal generation)
- Every exit path must guarantee the release, similar to `try/finally` in exception handling

---

## 10. Challenge 9: Recursive Thinking Trigger Loop

**File:** `metis/inference.py` — Escalation logic

### Problem Description

The most critical bug in the system. The escalation logic had three states:

1. **Escalation 1**: Repetition detected outside thinking → inject `<thinking>` to break the loop
2. **Escalation 2**: Repetition detected inside thinking → force-close thinking
3. **Force-stop**: After 3 repetition events → terminate generation

The bug: after Escalation 2 closed thinking, the model returned to normal generation. If it started repeating again (which was likely, since the model was already in a problematic state), Escalation 1 triggered **again**, injecting a new `<thinking>` block. This created a cycle:

```
Repetition → Escalation 1 (inject <thinking>)
    → Repetition in thinking → Escalation 2 (close </thinking>)
        → Repetition again → Escalation 1 (inject <thinking> AGAIN)
            → Repetition in thinking → Escalation 2 (close </thinking>)
                → Repetition → Escalation 1 ... (repeat until force-stop)
```

The model consumed ~362 tokens on 3 failed thinking attempts before force-stop, leaving almost no token budget for the actual answer.

### Root Cause Analysis

The `else` branch in the escalation logic handled both:
- `is_thinking == True` (legitimate: repetition inside thinking)
- `is_thinking == False and thinking_failed == True` (BUG: would try to close a non-existent thinking block)

There was no flag to track whether thinking had already been attempted and failed.

### Solution

Introduced a `thinking_failed` flag and restructured the escalation into three distinct branches:

```python
thinking_failed = False  # Tracks if thinking already failed

# During repetition handling:
if thinking_failed and not is_thinking:
    # Case 3: Thinking already failed, model STILL looping
    # No point continuing — force stop now
    generated_tokens = generated_tokens[:-rep_len]
    break

elif not is_thinking:
    # Escalation 1: First-time thinking injection
    # ... inject <thinking>, rebuild KV cache

else:
    # Escalation 2: Repetition inside thinking
    thinking_failed = True  # Mark thinking as failed
    # ... close </thinking>, rebuild KV cache
```

### Impact

Before: Model wasted ~362 tokens on 3 rounds of failed thinking, producing a truncated answer.
After: Model attempts thinking at most **once**. If it fails, generation stops immediately or continues without thinking, preserving the full token budget for the answer.

### State Machine

```
                    ┌──────────────┐
    repetition ──▶  │  Escalation 1│ ──▶ is_thinking = True
    (outside)       │  (inject     │
                    │  <thinking>) │
                    └──────┬───────┘
                           │
                    repetition (inside)
                           │
                    ┌──────▼───────┐
                    │  Escalation 2│ ──▶ is_thinking = False
                    │  (close      │     thinking_failed = True
                    │  </thinking>)│
                    └──────┬───────┘
                           │
                    repetition (outside, thinking_failed=True)
                           │
                    ┌──────▼───────┐
                    │  Case 3:     │ ──▶ FORCE STOP
                    │  Immediate   │     (no more thinking attempts)
                    │  termination │
                    └──────────────┘
```

---

## 11. Challenge 10: Token Budget Exhaustion by Thinking

**File:** `metis/inference.py`, `demo_metis.py`

### Problem Description

The `generate()` method's default `max_tokens = 256` was shared between thinking and the final answer. With thinking consuming 100–200 tokens, the answer portion was severely truncated. The demo also hardcoded `max_tokens = 200`.

### Root Cause Analysis

Thinking tokens and answer tokens shared the same budget. The generation loop:

```python
for step in range(max_tokens):
    # Both thinking and answer tokens count against this limit
```

With `max_tokens = 256` and thinking consuming 150 tokens, only 106 tokens remained for the answer — often insufficient for a complete response.

### Solution

1. Increased `generate()` default: `max_tokens = 256 → 2048`
2. Increased `run_demo()` default: `max_tokens = 200 → 2048`
3. The thinking protocol has its own `max_thinking_tokens = 512` limit (independent ceiling)

Combined with Challenge 9's fix (thinking attempts at most once), the worst-case thinking overhead is bounded at ~512 tokens, leaving ~1536 tokens for the answer.

---

## 12. Challenge 11: False Epistemic Uncertainty on Lexical Diversity

**File:** `metis/cognitive/boundary.py` — `EpistemicBoundaryGuard.evaluate()`

### Problem Description

For the query "什么是LLM" (What is LLM) — a basic factual question — METIS flagged 6–7 tokens as HEDGE (uncertain), including:

```
[190] HEDGE '包括'  H=4.94 z=+2.35   (should be: GENERATE)
[197] HEDGE '目前'  H=5.33 z=+1.87   (should be: GENERATE)
[212] HEDGE '不是'  H=5.37 z=+1.79   (should be: GENERATE)
```

The model was perfectly confident in its answer. These tokens appeared uncertain because the model had multiple **synonymous** next-token choices: "包括"/"例如"/"有" all mean "including" in different phrasings.

### Root Cause Analysis

The boundary guard used two signals to assess uncertainty:
1. **z-score** (relative entropy): measures how unusual this token's entropy is
2. **confidence** (top-token probability): measures how dominant the top candidate is

Both signals measure the **shape of the probability distribution** — they are fundamentally correlated. When the model chooses between 5 synonyms:
- Entropy is high (5 options → ~2.3 bits)
- Confidence is low (top token ≈ 25%)
- z-score is high (entropy above running average)

All three signals say "uncertain," but the model **knows what it wants to say** — it just has multiple valid phrasings. This is **lexical diversity**, not **epistemic uncertainty**.

The critical missing signal: **semantic diversity** — are the alternative tokens semantically similar (synonyms) or genuinely different (confusion)?

### Solution: Multi-Layered Approach

#### Layer 1: Semantic Diversity Gate

The `SemanticEntropyComputer` already computed a `semantic_diversity` metric (cosine distance of top-k embeddings), but the boundary guard never used it. Added a `SEMANTIC_DIVERSITY_GATE`:

```python
SEMANTIC_DIVERSITY_GATE = 0.75

# In evaluate():
sd = signal.semantic_diversity
if z > z_unk:
    if sd < SEMANTIC_DIVERSITY_GATE:
        return GENERATE  # Synonyms, not uncertainty
    # ... rest of logic
```

#### Layer 2: Streak Requirements

Single-token entropy spikes are common at sentence/topic boundaries and do not indicate sustained uncertainty. Tightened streak requirements:

| Zone | Before | After |
|------|--------|-------|
| z > z_unk → REFUSE/SEEK | streak ≥ 2 | streak ≥ 3 |
| z > z_unk → HEDGE | no requirement | streak ≥ 2 |
| z > z_unc → HEDGE | streak ≥ 2 | streak ≥ 3 |

#### Layer 3: Confidence Gate Tightening

Lowered `UNCERTAIN_CONFIDENCE_GATE` from 0.4 to 0.3. A top-token probability of 30% is common in 3–5 synonym scenarios and should not trigger HEDGE.

### Impact

HEDGE events for "介绍一下LLM": **7 → 2** (a 71% reduction in false positives).
Remaining HEDGEs occur at genuinely sustained high-entropy sequences (3+ consecutive high-z tokens with very low confidence).

---

## 13. Challenge 12: High-Dimensional Cosine Similarity Degeneracy

**File:** `metis/core/entropy.py` — `SemanticEntropyComputer._compute_semantic_diversity()`

### Problem Description

The semantic diversity metric was supposed to distinguish synonyms (low diversity) from genuinely different tokens (high diversity). In practice, **all** tokens showed diversity values of 0.84–0.95, making the signal useless:

```
[  1] sd=0.88  '我是'      (confident, should be low)
[  7] sd=0.91  '阿里巴巴'  (confident, should be low)
[ 28] sd=0.91  '处理'      (HEDGE trigger, should be high)
[ 51] sd=0.74  '？'        (punctuation)
```

The `SEMANTIC_DIVERSITY_GATE` at 0.3 never filtered any content token.

### Root Cause Analysis

This is a manifestation of the **concentration of measure** phenomenon in high-dimensional spaces. The Qwen 2.5-7B model has an embedding dimension of 3584. In such high-dimensional spaces:

**Theorem (Concentration of Cosine Similarity):**
For two independent random unit vectors in ℝ^d, their cosine similarity converges to 0 as d → ∞, with standard deviation O(1/√d).

For d = 3584:
```
E[cos(u, v)] = 0
Std[cos(u, v)] ≈ 1/√3584 ≈ 0.017
```

This means virtually all token pairs have cosine similarity in the range [-0.05, 0.05], making `1 - avg_similarity ≈ 0.95–1.00` for all tokens. Even genuine synonyms, which might have similarity 0.1–0.2, produce diversity values of 0.80–0.90 — barely distinguishable from completely unrelated tokens.

### Original Implementation

```python
# Equal-weight average over all top-k pairs
k = 10
top_embeddings = embedding_matrix[top_indices]  # [batch, 10, 3584]
sim_matrix = cosine_similarity(top_embeddings)   # [batch, 10, 10]
avg_similarity = mean(upper_triangle(sim_matrix))
diversity = 1 - avg_similarity  # Always ≈ 0.9
```

The equal-weight averaging over k=10 tokens further diluted the signal: tokens ranked 4–10 have very low probability and contribute noise.

### Solution: Probability-Weighted Cosine Distance

Redesigned the diversity computation with two key changes:

1. **Reduced k from 10 to 5**: Focus on high-probability candidates
2. **Probability-weighted pairwise distance**: Weight each pair (i,j) by p_i × p_j

```python
k = min(5, self._top_k, logits.shape[-1])
top_values, top_indices = torch.topk(logits, k, dim=-1)
top_probs = F.softmax(top_values.float(), dim=-1)  # Re-normalize

# Probability-weighted similarity
weight_matrix = top_probs.unsqueeze(2) @ top_probs.unsqueeze(1)  # [b, k, k]
mask = torch.triu(torch.ones(k, k), diagonal=1)

weighted_sim = (sim_matrix * weight_matrix * mask).sum() / (weight_matrix * mask).sum()
diversity = 1 - weighted_sim
```

### Mathematical Justification

For the synonym case: "包括"(p=0.40) / "例如"(p=0.35) / "有"(p=0.15) / noise(p=0.10):

```
Pair ("包括","例如"): weight = 0.40 × 0.35 = 0.140, sim ≈ 0.10
Pair ("包括","有"):   weight = 0.40 × 0.15 = 0.060, sim ≈ 0.08
Other pairs:          weight ≈ 0.01–0.03,            sim ≈ 0.02

Weighted sum = 0.140×0.10 + 0.060×0.08 + ... ≈ 0.022
Weight total  = 0.140 + 0.060 + 0.053 + ... ≈ 0.30
Avg weighted sim = 0.022 / 0.30 ≈ 0.073
Diversity = 1 - 0.073 = 0.927
```

While the improvement is modest in absolute terms (from 0.95 to 0.93 for synonyms), the key insight is that the diversity compression into 0.5–0.97 range is inherent to high-dimensional embeddings. The gate was therefore raised to 0.75 to match the actual observed distribution:

- **sd < 0.75**: Punctuation, deterministic tokens, some synonym scenarios → GENERATE
- **sd ≥ 0.75**: Content tokens requiring further streak/confidence analysis

### Broader Lesson

Token-level embedding similarity is a **weak signal** for semantic synonymy in high-dimensional spaces. More effective approaches for future work include:
- **Contextual embeddings**: Use hidden states instead of static token embeddings
- **Entailment-based clustering**: Following Kuhn et al. (ICLR 2023)
- **Adaptive normalization**: Z-normalize diversity values over a running window

---

## 14. Summary of Mathematical & Theoretical Foundations

### Signal Processing

| Technique | Purpose in METIS | Reference |
|-----------|-----------------|-----------|
| **Adaptive Forgetting Factor (AFF)** | Adapts exponential smoothing rate based on prediction error | Fortescue et al., 1981 |
| **Siegmund's Corrected CUSUM** | Change-point detection for entropy regime shifts | Siegmund, 1985 |
| **Cornish-Fisher Expansion** | Non-Gaussian quantile estimation for z-score thresholds | Cornish & Fisher, 1938 |

### Information Theory

| Concept | Formula | Use in METIS |
|---------|---------|-------------|
| **Shannon Entropy** | H = -Σ p_i log₂(p_i) | Base token-level uncertainty |
| **Semantic Entropy** | H_sem = H × (1 + λ·D) | Entropy weighted by semantic diversity |
| **Confidence** | c = max(p_i) | Top-token probability |
| **Z-Score** | z = (H - μ) / σ | Relative entropy (model-adaptive) |

### Cognitive Science

| Theory | Application in METIS |
|--------|---------------------|
| **Kahneman's Dual-Process** | System 1 (FAST) / System 2 (DEEP) cognitive modes |
| **Epistemic States** | KNOWN / LIKELY / UNCERTAIN / UNKNOWN classification |
| **Metacognition** | Self-monitoring of cognitive confidence and load |

### High-Dimensional Geometry

| Phenomenon | Impact | Mitigation |
|------------|--------|------------|
| **Concentration of Measure** | Cosine similarity → 0 in high dims | Probability-weighted aggregation |
| **Curse of Dimensionality** | Embedding distance loses discriminative power | Reduced top-k, adaptive gate |

---

## 15. Lessons Learned

### 1. Default Values Are Design Decisions

The `temperature=0.0` default and `max_tokens=256` default silently disabled core features. In a system where components interact through shared parameters, **every default value is an implicit design decision** that constrains the behavior envelope of the entire system.

**Principle:** Critical system parameters should have defaults that enable (not disable) the system's core functionality. If a parameter controls an internal cognitive process (like DEEP mode temperature), it should be autonomous, not user-facing.

### 2. Quantization Changes the Problem Space

Algorithms designed for FP32 precision may fail silently on quantized models. The logit sharpening bug produced "correct" outputs most of the time, but with subtle argmax instability on close logit values. This class of bug is extremely difficult to detect through output inspection alone.

**Principle:** When targeting quantized models, test all numerical operations at the precision boundary. Logit differences < 0.1 should be treated as noise, not signal.

### 3. KV Cache Is Hidden State

The KV cache is an invisible "memory" that persists across generation steps. Any operation that modifies the token sequence (trimming, injection, rollback) MUST update the KV cache to match, or the model will hallucinate from "phantom" context.

**Principle:** Treat KV cache updates as a mandatory postcondition of any token-sequence mutation. This is analogous to maintaining database consistency after write operations.

### 4. Correlated Signals Create Blind Spots

Entropy and confidence both measure distribution shape — they are not independent signals. Using both to detect uncertainty creates a blind spot for **lexical diversity** (multiple valid phrasings), which produces high entropy AND low confidence despite zero epistemic uncertainty.

**Principle:** When building multi-signal classifiers, verify signal independence. Correlated signals cannot distinguish scenarios where one underlying cause (distribution shape) maps to different semantic meanings (uncertainty vs. lexical choice).

### 5. High-Dimensional Geometry Breaks Intuition

Cosine similarity, a reliable metric in low-dimensional spaces (d < 100), becomes degenerate in high-dimensional embedding spaces (d > 1000). All pairs converge to similarity ≈ 0, destroying the discriminative power that the algorithm relies on.

**Principle:** When adapting algorithms from low-dimensional settings to high-dimensional ones, verify that the core distance metric retains its discriminative power. Use probability-weighted aggregation to focus on the statistically meaningful subset of comparisons.

### 6. State Machines Need Exhaustive Exit Coverage

The thinking protocol had four exit paths, but only one was properly handled. In a stateful generation system, every combination of (exit_condition × internal_state) must be explicitly addressed.

**Principle:** Enumerate all (exit × state) pairs before implementation. Use assertions or runtime checks to catch unhandled combinations early.

### 7. Failed Recovery Attempts Should Not Be Repeated

The recursive thinking loop (Challenge 9) demonstrated that retrying a failed recovery strategy consumes resources without benefit. If thinking failed due to repetition, the model's internal state makes thinking unproductive for this particular input — retrying will produce the same failure.

**Principle:** Track recovery attempt history and implement a "circuit breaker" pattern: after one failed attempt, escalate to a different strategy (or graceful termination) rather than repeating the same intervention.

---

*Document generated from the METIS v9.0 development audit.*
*All code references point to the `metis/` package in the SEDAC v9.0 repository.*
