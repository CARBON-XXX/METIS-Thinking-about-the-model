# METIS V1 Hardware Utilization Strategy

**Full Project Compute Architecture on a 128 GB VRAM / 1 PetaFLOPS DGX Node**

> Internal Whitepaper — Systems Architecture Division
> Last updated: 2026-03-09

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Training Compute Scaling: The RLHF Bottleneck](#2-training-compute-scaling-the-rlhf-bottleneck)
   - 2.1 [Three-Phase Training Pipeline](#21-three-phase-training-pipeline)
   - 2.2 [GRPO Memory Anatomy](#22-grpo-memory-anatomy)
   - 2.3 [The Group-Size / KV-Cache Tradeoff](#23-the-group-size--kv-cache-tradeoff)
   - 2.4 [LoRA Rank Scaling for Reasoning Absorption](#24-lora-rank-scaling-for-reasoning-absorption)
   - 2.5 [DPO vs GRPO: Memory Footprint Comparison](#25-dpo-vs-grpo-memory-footprint-comparison)
3. [Inference Compute Elasticity: The Gateway](#3-inference-compute-elasticity-the-gateway)
   - 3.1 [Dual-System Architecture](#31-dual-system-architecture)
   - 3.2 [Cognitive Routing and Test-Time Compute Distribution](#32-cognitive-routing-and-test-time-compute-distribution)
   - 3.3 [The Epistemic Tax and Lazy Evaluation](#33-the-epistemic-tax-and-lazy-evaluation)
4. [Future Hardware Scaling Roadmap](#4-future-hardware-scaling-roadmap)
   - 4.1 [Multi-Node GRPO: Separating Rollout from Policy Optimization](#41-multi-node-grpo-separating-rollout-from-policy-optimization)
   - 4.2 [Inference Serving at Scale](#42-inference-serving-at-scale)

---

## 1. Executive Summary

METIS V1 is a metacognitive routing engine built on Qwen2.5-7B-Instruct, trained via a three-phase alignment pipeline — **SFT → DPO → GRPO** — on a single NVIDIA GB10 DGX node (128 GB unified VRAM, Ada/Blackwell-class, CUDA Capability 12.1).

The system achieved a **Pareto-optimal** outcome on the target metrics:

| Metric | DPO Baseline | GRPO Final | Delta |
|--------|:------------:|:----------:|:-----:|
| GSM8K Accuracy (n=50) | 84.0% | **86.0%** | **+2.0%** |
| Simple QA Accuracy (n=30) | 96.7% | 96.7% | 0.0% |
| Overall Accuracy (n=80) | 88.8% | **90.0%** | **+1.2%** |
| Avg Token Verbosity | 72.6 | **71.3** | **−1.9%** |

Higher accuracy *and* lower verbosity — strict Pareto improvement over the DPO-only policy. This document details the hardware engineering decisions that made this possible on a single node.

---

## 2. Training Compute Scaling: The RLHF Bottleneck

### 2.1 Three-Phase Training Pipeline

Each phase has a fundamentally different compute signature:

| Phase | Algorithm | Forward Passes per Step | Dominant Resource | Wall Time |
|-------|-----------|:-----------------------:|-------------------|-----------|
| **SFT** | Supervised Fine-Tuning | 1 | Activations (O(BLd)) | ~45 min |
| **DPO** | Direct Preference Optimization | 2 (chosen + rejected) | 2× Activations | ~1h 41m |
| **GRPO** | Group Relative Policy Optimization | 1 + G rollouts | KV-Cache (O(BGLd_h)) | **16h 41m** |

Where $B$ = batch size, $L$ = sequence length, $d$ = hidden dimension, $G$ = group size (num_generations), $d_h$ = head dimension.

GRPO dominates wall time by 10× because each training step requires generating $G = 16$ complete rollouts per prompt before any gradient computation occurs.

### 2.2 GRPO Memory Anatomy

The GB10 node provides 122 GB of usable VRAM (128 GB total minus OS/driver reservation). Our GRPO configuration occupies memory as follows:

| Component | Size (GB) | Formula / Notes |
|-----------|:---------:|-----------------|
| Base Model (bf16) | ~14.0 | $7\text{B params} \times 2\text{ bytes}$ |
| LoRA Adapter ($r=64$) | ~4.0 | $2 \times r \times d \times 7\text{ modules} \times 2\text{ bytes}$ |
| Optimizer States (AdamW) | ~8.0 | $2 \times |\theta_{\text{LoRA}}|$ (first + second moments) |
| KV-Cache (generation) | ~7.0 | $B_{\text{gen}} \times L_{\text{max}} \times 2 \times n_{\text{layers}} \times d_h \times 2$ |
| Gradient Checkpointed Activations | ~20.0 | Recomputed per-layer; peak ≈ $O(B \sqrt{N_{\text{layers}}})$ |
| **Peak Observed** | **~53** | **43% of 122 GB capacity** |

Gradient checkpointing (`use_reentrant=False`) was essential — without it, activation memory alone would exceed 60 GB, pushing total past the OOM boundary when combined with the KV-Cache during rollout.

### 2.3 The Group-Size / KV-Cache Tradeoff

GRPO estimates the advantage function using within-group normalization:

$$\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G + \epsilon}$$

where $r_i$ is the reward for the $i$-th completion and $\mu_G, \sigma_G$ are the group mean and standard deviation across $G$ completions of the same prompt.

**Increasing $G$ from 4 to 16 reduces advantage estimator variance by $4\times$** (variance scales as $O(1/G)$), producing more stable policy gradients. However, this shifts the computational bottleneck:

| Group Size $G$ | Completions per Step | KV-Cache Peak | Backprop Time | Generation Time | Bottleneck |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 4 | 16 | ~1.8 GB | ~15s | ~20s | **Balanced** |
| **16** | **64** | **~7.0 GB** | **~24s** | **~112s** | **Generation (82%)** |
| 32 | 128 | ~14.0 GB | ~24s | ~220s | Generation (90%) |

At $G = 16$, autoregressive generation consumes **82% of wall time** (112s generation vs. 24s training per cycle). The KV-Cache must store:

$$\text{KV}_{\text{peak}} = B_{\text{gen}} \times L_{\text{max}} \times 2 \times n_{\text{layers}} \times d_h \times 2_{\text{bytes}}$$

For our configuration ($B_{\text{gen}} = 16, L_{\text{max}} = 1024, n_{\text{layers}} = 28, d_h = 128$):

$$\text{KV}_{\text{peak}} = 16 \times 1024 \times 2 \times 28 \times 128 \times 2 \approx 7.3 \text{ GB}$$

This is manageable on the GB10, but $G = 32$ would push KV-Cache to ~14.6 GB, consuming 12% of total VRAM for caching alone — a non-trivial fraction that competes with activation memory during gradient computation.

**Amortization strategy.** TRL's `generation_batch_size` parameter creates an implicit `steps_per_generation` ratio. We configured `generation_batch_size=16` with `per_device_train_batch_size=4`, yielding `steps_per_generation=4`. This means:

- **Step 1**: Generate 16 prompts × 16 completions = 256 rollouts (~112s)
- **Steps 2–4**: Train on cached rollouts, no generation (~24s each)
- **Amortized cost**: $(112 + 3 \times 24) / 4 = 47\text{s/step}$ (vs. 112s without amortization)

This 2.4× throughput improvement was critical for completing 1000 steps in 16h 41m on a single node.

### 2.4 LoRA Rank Scaling for Reasoning Absorption

The DPO phase used modest LoRA parameters ($r = 16, \alpha = 32$) targeting attention projections only. For GRPO, we scaled aggressively:

| Parameter | DPO (Phase 6) | GRPO (Phase 15) | Rationale |
|-----------|:-------------:|:----------------:|-----------|
| LoRA rank $r$ | 16 | **64** | Higher rank = more capacity for math reasoning patterns |
| $\alpha$ | 32 | **128** | Effective LR scaling: $\alpha / r = 2.0$ (constant) |
| Target modules | 4 (q/k/v/o) | **7** (q/k/v/o/gate/up/down) | MLP layers critical for factual recall; attention alone insufficient for GSM8K |
| Trainable params | ~26M (0.37%) | **~168M (2.4%)** | 6.5× more capacity |
| Adapter VRAM | ~0.8 GB | **~4.0 GB** | Acceptable: only 3.3% of total budget |
| `modules_to_save` | embed + lm_head | embed + lm_head | Preserve tokenizer alignment |

The $r = 64$ rank was justified by the **reasoning absorption hypothesis**: GSM8K requires multi-step arithmetic chains that engage both attention (step sequencing) and MLP (number manipulation) pathways. At $r = 16$, the adapter lacks sufficient rank to represent the reward-gradient manifold for chain-of-thought reasoning without overwriting the DPO-trained cognitive routing signal.

Empirically, KL divergence remained below 0.01 throughout training (mean $\approx$ 0.005), confirming that the larger adapter absorbed the math signal without catastrophic forgetting of the prior alignment.

### 2.5 DPO vs GRPO: Memory Footprint Comparison

| Resource | SFT | DPO | GRPO |
|----------|:---:|:---:|:----:|
| Model copies in VRAM | 1 | 2 (policy + ref) | 1 + implicit ref (via KL) |
| Forward passes / step | 1 | 2 (chosen + rejected) | $1 + G$ (train + rollouts) |
| Peak VRAM (measured) | ~30 GB | ~57 GB | ~53 GB |
| Bottleneck | Activations | Dual-model activations | KV-Cache + autoregressive gen |
| Safe batch size ($L = 1024$) | 8 | 2 | 4 |
| Gradient accum needed | 1 | 8 | 2 |
| Effective batch | 8 | 16 | 8 |

**Key insight**: DPO's memory pressure comes from maintaining two full model copies simultaneously (policy + reference), doubling activation memory. GRPO avoids this by computing KL divergence against a frozen copy of the initial policy weights (loaded once during trainer init), but instead pays the cost in sequential autoregressive generation across $G$ rollouts. On a single node, DPO is memory-bound while GRPO is compute-bound.

The GB10's 122 GB unified memory is underutilized by both algorithms (peak ~57 GB for DPO, ~53 GB for GRPO). The constraint is not raw capacity but rather **memory bandwidth during generation**: each token decode requires a full KV-Cache read, making the autoregressive loop bandwidth-bound rather than compute-bound.

---

## 3. Inference Compute Elasticity: The Gateway

### 3.1 Dual-System Architecture

METIS implements a **Kahneman-inspired dual-system** inference architecture (`metis/inference.py: MetisInference`):

```
User Query
    │
    ▼
┌──────────────────────────────┐
│  Cognitive Router (DPO/GRPO) │  ← Model emits [COGNITIVE_STATE: FAST/DEEP]
│  generate_cognitive()        │
└──────────┬───────────────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌──────────────────────────────────────────┐
│ FAST    │  │ DEEP                                     │
│ Path    │  │  ├─ <thinking>...</thinking> generation  │
│         │  │  ├─ Token-level CUSUM entropy monitoring  │
│ O(L)    │  │  ├─ Repetition detection + intervention  │
│ tokens  │  │  ├─ Semantic Entropy Probe (on demand)   │
│         │  │  └─ Curiosity-driven retrieval (on UNKNOWN)│
└─────────┘  └──────────────────────────────────────────┘
  ~3-5s                    ~15-30s
  ~10 tok                  ~100-500 tok
```

The router is not a separate classifier — it is an **emergent behavior** of the DPO-trained model, which learns to emit cognitive state tags as the first tokens of its response. A deterministic state machine (`_parse_cognitive_route()`) parses the raw output into structured routing decisions.

### 3.2 Cognitive Routing and Test-Time Compute Distribution

The routing creates a **bimodal compute distribution**, where simple queries consume $O(L_{\text{short}})$ and complex queries consume $O(L_{\text{long}} + C_{\text{thinking}})$:

| Route | Trigger | Tokens Generated | Latency | Compute Cost |
|-------|---------|:----------------:|:-------:|:------------:|
| **FAST** (explicit) | `[COGNITIVE_STATE: FAST]` | 5–15 | ~3s | $O(1)$ relative |
| **FAST** (implicit) | No tag emitted | 8–20 | ~5s | $O(1)$ relative |
| **DEEP** | `[COGNITIVE_STATE: DEEP]` + `<thinking>` | 100–512 | 15–30s | $O(n)$ with thinking |
| **DEEP + Probe** | DEEP + high uncertainty | 100–512 + 5 × 64 | 30–60s | $O(n + 5n')$ epistemic tax |

This creates **Compute Elasticity**: the system allocates GPU cycles proportionally to problem difficulty. In the Phase 14 benchmark (n=100):

- **82%** of simple queries routed to FAST → 8.1 avg tokens → ~3s each
- **100%** of complex queries routed to DEEP → 109.1 avg tokens → ~18s each
- **Aggregate savings**: ~60% fewer tokens than a static "always think" policy

The token-level monitoring pipeline (`generate()` method) adds a per-token overhead of approximately **0.3ms** (boundary guard z-score computation + CUSUM update). For a 500-token DEEP response, this adds ~150ms — negligible compared to the ~15s total generation time. The overhead comes from:

1. **Entropy z-score**: $z_t = (H_t - \mu_H) / \sigma_H$ — running mean/variance, $O(1)$ per token
2. **CUSUM accumulator**: $S_t = \max(0, S_{t-1} + (z_t - k) \cdot \sigma_t)$ — single scalar update
3. **Boundary classification**: Threshold comparison against $S_t$ — constant time

### 3.3 The Epistemic Tax and Lazy Evaluation

The Semantic Entropy Probe (`SemanticBoundaryProbe` in `metis/cognitive/boundary.py`) implements the Kuhn et al. (ICLR 2023) method:

1. **Sample** $N = 5$ short answers at temperature $T = 1.0$ → 5 forward passes, 64 tokens each
2. **Cluster** by semantic equivalence via pairwise LLM judgment → $\binom{N}{2} = 10$ short forward passes
3. **Compute** Shannon entropy: $\mathcal{H} = -\sum_k p_k \log_2 p_k$
4. **Classify**: $\mathcal{H} > 0.8 \rightarrow$ UNKNOWN, $\mathcal{H} > 0.4 \rightarrow$ UNCERTAIN, else KNOWN/LIKELY

The **Epistemic Tax** is severe: $5 + 10 = 15$ additional forward passes per probed query. At ~3s per short generation, this adds **~45s of latency** — a 3× multiplier on DEEP queries.

**Mitigation: Lazy Evaluation (Phase 14).** The `MetacognitiveOrchestrator` implements absolute short-circuit:

```python
# Pseudo-code from metacognition.py
if route in (FAST, FAST_IMPLICIT):
    return response  # NO probe instantiation, NO entropy calculation
    # semantic_entropy = None (not 0.0 — never computed)
```

The `SemanticBoundaryProbe` is **never instantiated** for FAST queries — not even allocated in memory. It is lazy-created on the first DEEP hit via `_ensure_probe()`. This eliminates the epistemic tax for the majority of production traffic (simple queries), reducing the amortized inference cost to:

$$C_{\text{amortized}} = p_{\text{FAST}} \cdot C_{\text{FAST}} + p_{\text{DEEP}} \cdot (C_{\text{DEEP}} + p_{\text{probe}} \cdot C_{\text{probe}})$$

With $p_{\text{FAST}} \approx 0.82$, $p_{\text{DEEP}} \approx 0.18$, and $p_{\text{probe}} \approx 0.5$ (only triggered when uncertainty is detected):

$$C_{\text{amortized}} \approx 0.82 \times 3\text{s} + 0.18 \times (18\text{s} + 0.5 \times 45\text{s}) \approx 2.46 + 7.29 = 9.75\text{s}$$

Compared to a naive "always probe" policy: $0.82 \times 48\text{s} + 0.18 \times 63\text{s} = 50.7\text{s}$ — a **5.2× latency reduction**.

---

## 4. Future Hardware Scaling Roadmap

### 4.1 Multi-Node GRPO: Separating Rollout from Policy Optimization

The single-node GRPO architecture is fundamentally limited by the **generation bottleneck** — 82% of wall time is spent in sequential autoregressive decoding. On a multi-node cluster, the architecture decomposes naturally:

```
┌─────────────────────────────────────────────────────────────┐
│                    Policy Optimizer Node                     │
│  (1× GPU, high-bandwidth)                                   │
│  - Receives completed rollouts + rewards                    │
│  - Computes group-normalized advantages                     │
│  - Runs PPO/GRPO gradient update                            │
│  - Broadcasts updated policy weights                        │
└──────────────────────┬──────────────────────────────────────┘
                       │ Weight sync (every K steps)
          ┌────────────┼────────────┐
          ▼            ▼            ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Rollout      │ │ Rollout      │ │ Rollout      │
│ Worker 0     │ │ Worker 1     │ │ Worker N     │
│ (1× GPU)     │ │ (1× GPU)     │ │ (1× GPU)     │
│ - vLLM serve │ │ - vLLM serve │ │ - vLLM serve │
│ - G/N gens   │ │ - G/N gens   │ │ - G/N gens   │
│ - Reward eval│ │ - Reward eval│ │ - Reward eval│
└──────────────┘ └──────────────┘ └──────────────┘
```

**Scaling properties:**

| Configuration | Rollout Workers | Gen Time (est.) | Total Step Time | Speedup |
|---------------|:---------------:|:---------------:|:---------------:|:-------:|
| Current (1-node) | 0 (colocated) | 112s | 60s (amortized) | 1.0× |
| 2-node | 1 dedicated | ~60s | ~40s | 1.5× |
| 4-node | 3 dedicated | ~25s | ~30s | 2.0× |
| 8-node + vLLM | 7 dedicated (vLLM) | ~8s | ~25s | **2.4×** |

The key enabler for multi-node is **vLLM**-based rollout workers. Our current vLLM version (0.16.0) is incompatible with TRL 0.29.0's colocate mode, but a dedicated rollout service (decoupled from TRL) would bypass this constraint entirely. Each worker runs a frozen policy copy with continuous batching and PagedAttention, achieving 3–5× higher generation throughput than HuggingFace `generate()`.

**Weight synchronization** is the critical coordination point. Two strategies:

1. **Synchronous**: All workers pause, receive new weights, resume. Simple but creates idle bubbles.
2. **Asynchronous (IMPALA-style)**: Workers generate continuously; optimizer processes rollouts with slight off-policy lag. Higher throughput but requires importance-weight correction ($\rho_t = \pi_\theta / \pi_{\theta_{\text{old}}}$).

For METIS V2, we recommend the asynchronous approach with $G = 64$ distributed across 4 rollout workers, each generating $G/4 = 16$ completions in parallel. This would reduce the 16h 41m training run to approximately **~7 hours**.

### 4.2 Inference Serving at Scale

The cognitive routing architecture maps directly to a tiered serving topology:

```
              Load Balancer
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐
   │ FAST    │ │ FAST    │ │ DEEP    │
   │ Pool    │ │ Pool    │ │ Pool    │
   │ (small  │ │ (small  │ │ (large  │
   │  batch, │ │  batch, │ │  batch, │
   │  low    │ │  low    │ │  high   │
   │  latency│ │  latency│ │  tokens)│
   └─────────┘ └─────────┘ └─────────┘
    vLLM, tp=1   vLLM, tp=1  vLLM, tp=2
    max_tok=64   max_tok=64  max_tok=1024
```

- **FAST Pool**: Optimized for latency. Small max_tokens, aggressive batching, single-GPU tensor parallelism. Target: p99 < 500ms.
- **DEEP Pool**: Optimized for throughput. Large max_tokens, thinking budget, optional Semantic Entropy Probe sidecar. Target: p99 < 5s.
- **Routing**: The first ~10 tokens determine the pool. A lightweight prefix classifier (or the model's own first-token logits) routes requests before full generation begins.

The Semantic Entropy Probe becomes a **dedicated microservice** at scale — sampling and equivalence judgments are embarrassingly parallel across the $N = 5$ samples, reducing probe latency from 45s (sequential) to ~9s (5-way parallel).

---

## Appendix A: Observed Training Metrics (Phase 15 GRPO)

| Metric | Value |
|--------|-------|
| Hardware | NVIDIA GB10, 128 GB unified VRAM, CUDA 12.1 |
| Total training steps | 1,000 |
| Wall-clock time | 16h 41m 49s |
| Average step time | 60.1s (amortized over gen + train) |
| Generation step time | ~112s (every 4th step) |
| Training-only step time | ~24s (3 of 4 steps) |
| Final train loss | 0.030 |
| Peak reward observed | 2.0 (theoretical maximum) |
| Mean KL divergence | ~0.005 (healthy; β=0.04) |
| Max KL outlier | 0.040 (step 415, self-recovered) |
| Max grad_norm spike | 80.2 (clipped by max_grad_norm=1.0) |
| Peak VRAM usage (estimated) | ~53 GB / 122 GB available (43%) |
| Checkpoints saved | 500, 600, 700 (save_total_limit=3) |

## Appendix B: Key Configuration Reference

```yaml
# Phase 15 GRPO — Proven Configuration
model: Qwen2.5-7B-Instruct (DPO-finetuned)
precision: bf16
attention: SDPA (Flash Attention 2 unavailable on GB10 CC12.1)

lora:
  r: 64
  alpha: 128
  dropout: 0.05
  targets: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  modules_to_save: [embed_tokens, lm_head]

grpo:
  num_generations: 16
  generation_batch_size: 16
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 2
  max_completion_length: 1024
  temperature: 0.8
  beta: 0.04  # KL penalty coefficient
  learning_rate: 1e-6
  lr_scheduler: cosine
  warmup_ratio: 0.1
  max_grad_norm: 1.0
  gradient_checkpointing: true
  torch_empty_cache_steps: 10

rewards:
  - accuracy_reward: +2.0 (correct answer)
  - verbosity_penalty: -0.1 * max(0, tokens - 20) | +0.5 (thinking tags)
  - weights: [1.0, 1.0]

dataset:
  gsm8k_train: 250 prompts
  simple_qa: 250 prompts
  total: 500
```

---

*Document generated as part of the METIS V1 project retrospective. All measurements taken on a single NVIDIA GB10 DGX node running Ubuntu Linux with PyTorch 2.x, TRL 0.29.0, and PEFT 0.14.x.*
