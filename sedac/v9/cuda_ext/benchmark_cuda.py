"""SEDAC CUDA Kernels Benchmark"""
import torch
import time
import sedac_cuda

print("=" * 70)
print("SEDAC V9.0 CUDA Kernels Benchmark")
print("=" * 70)
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA Module: {sedac_cuda}")
print("=" * 70)

device = torch.device("cuda")

# Config
N = 4096
vocab_size = 32000
hidden_size = 4096

print(f"\nConfig: N={N}, vocab={vocab_size}, hidden={hidden_size}")

# Data
logits = torch.randn(N, vocab_size, device=device)
hidden = torch.randn(N, hidden_size, device=device)
prev_hidden = torch.randn(N, hidden_size, device=device)

# Warmup
print("\nWarmup (10 iterations)...")
for _ in range(10):
    result = sedac_cuda.fused_entropy_decision(
        logits, hidden, prev_hidden,
        3.0, 1.0, 0.5, 0.7
    )
torch.cuda.synchronize()

# Benchmark Fused Entropy Decision
iterations = 100
print(f"\nBenchmark ({iterations} iterations)...")

torch.cuda.synchronize()
start = time.perf_counter()

for _ in range(iterations):
    entropy, confidence, decision, load = sedac_cuda.fused_entropy_decision(
        logits, hidden, prev_hidden,
        3.0, 1.0, 0.5, 0.7
    )

torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) * 1000 / iterations

print(f"\n[Fused Entropy Decision - CUDA Kernel]")
print(f"  Latency: {elapsed:.3f} ms")
print(f"  Throughput: {N / elapsed * 1000:.0f} tokens/sec")
print(f"  Exit ratio: {decision.float().mean().item()*100:.1f}%")

# Benchmark Token Router Split
exit_mask = torch.rand(N, device=device) > 0.6

for _ in range(10):
    sedac_cuda.token_router_split(hidden, exit_mask)
torch.cuda.synchronize()

start = time.perf_counter()
for _ in range(iterations):
    result = sedac_cuda.token_router_split(hidden, exit_mask)
torch.cuda.synchronize()

elapsed = (time.perf_counter() - start) * 1000 / iterations

print(f"\n[Token Router Split - CUDA Kernel]")
print(f"  Latency: {elapsed:.3f} ms")
print(f"  Active: {result[0].shape[0]}, Exit: {result[2].shape[0]}")

# Compare with PyTorch baseline
print("\n" + "-" * 70)
print("PyTorch Baseline Comparison")
print("-" * 70)

import torch.nn.functional as F
import math

def pytorch_entropy_decision(logits, hidden, prev_hidden, mean_ent, std_ent, progress, thresh):
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -torch.sum(probs * log_probs, dim=-1) / math.log(2)
    
    diff = hidden - prev_hidden
    diff_norm = torch.norm(diff, p=2, dim=-1)
    hidden_norm = torch.norm(hidden, p=2, dim=-1)
    stability = 1.0 / (1.0 + diff_norm / (hidden_norm + 1e-6))
    
    z_score = (mean_ent - entropy) / (std_ent + 1e-6)
    confidence = torch.sigmoid(z_score * 2.0)
    
    load = (1.0 - confidence) * 0.5 + (1.0 - stability) * 0.3 + (1.0 - progress) * 0.2
    current_thresh = thresh - progress * 0.2
    decision = (confidence * stability * progress) > current_thresh
    
    return entropy, confidence, decision, load

# Warmup
for _ in range(10):
    pytorch_entropy_decision(logits, hidden, prev_hidden, 3.0, 1.0, 0.5, 0.7)
torch.cuda.synchronize()

start = time.perf_counter()
for _ in range(iterations):
    pytorch_entropy_decision(logits, hidden, prev_hidden, 3.0, 1.0, 0.5, 0.7)
torch.cuda.synchronize()

pytorch_elapsed = (time.perf_counter() - start) * 1000 / iterations

print(f"\n[Fused Entropy Decision - PyTorch]")
print(f"  Latency: {pytorch_elapsed:.3f} ms")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"CUDA Kernel:  {elapsed:.3f} ms")
print(f"PyTorch:      {pytorch_elapsed:.3f} ms")
print(f"Speedup:      {pytorch_elapsed / elapsed:.1f}x")
print("=" * 70)
