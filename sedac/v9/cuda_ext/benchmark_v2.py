"""SEDAC V2 CUDA Kernels Benchmark - Target: <1ms"""
import torch
import time

print("=" * 70)
print("SEDAC V9.0 CUDA Kernels V2 Benchmark")
print("=" * 70)

# Load both versions
import sedac_cuda
import sedac_cuda_v2

print(f"V1: {sedac_cuda}")
print(f"V2: {sedac_cuda_v2}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 70)

device = torch.device("cuda")

def benchmark_kernel(fn, args, name, iterations=100):
    """Benchmark a kernel function"""
    # Warmup
    for _ in range(20):
        fn(*args)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        fn(*args)
    torch.cuda.synchronize()
    
    elapsed_ms = (time.perf_counter() - start) * 1000 / iterations
    return elapsed_ms

# Test configurations
configs = [
    {"name": "Small (32 tokens)", "N": 32, "vocab": 32000, "hidden": 4096},
    {"name": "Medium (256 tokens)", "N": 256, "vocab": 32000, "hidden": 4096},
    {"name": "Large (1024 tokens)", "N": 1024, "vocab": 32000, "hidden": 4096},
    {"name": "XLarge (4096 tokens)", "N": 4096, "vocab": 32000, "hidden": 4096},
    {"name": "Qwen-like (30 tokens)", "N": 30, "vocab": 151936, "hidden": 2048},
    {"name": "Qwen-batch (512 tokens)", "N": 512, "vocab": 151936, "hidden": 2048},
]

print("\n" + "=" * 70)
print("Fused Entropy Decision Benchmark")
print("=" * 70)
print(f"{'Config':<25} {'V1 (ms)':<12} {'V2 (ms)':<12} {'Speedup':<10} {'Target'}")
print("-" * 70)

for cfg in configs:
    N, vocab, hidden = cfg["N"], cfg["vocab"], cfg["hidden"]
    
    logits = torch.randn(N, vocab, device=device, dtype=torch.float32)
    h = torch.randn(N, hidden, device=device, dtype=torch.float32)
    ph = torch.randn(N, hidden, device=device, dtype=torch.float32)
    
    # V1
    v1_time = benchmark_kernel(
        sedac_cuda.fused_entropy_decision,
        (logits, h, ph, 3.0, 1.0, 0.5, 0.7),
        "V1"
    )
    
    # V2
    v2_time = benchmark_kernel(
        sedac_cuda_v2.fused_entropy_decision_v2,
        (logits, h, ph, 3.0, 1.0, 0.5, 0.7),
        "V2"
    )
    
    speedup = v1_time / v2_time
    target = "OK" if v2_time < 1.0 else "MISS"
    
    print(f"{cfg['name']:<25} {v1_time:<12.3f} {v2_time:<12.3f} {speedup:<10.2f}x {target}")

# FP16 Test
print("\n" + "=" * 70)
print("FP16 Mode Benchmark")
print("=" * 70)

for cfg in configs[:4]:
    N, vocab, hidden = cfg["N"], cfg["vocab"], cfg["hidden"]
    
    logits = torch.randn(N, vocab, device=device, dtype=torch.float16)
    h = torch.randn(N, hidden, device=device, dtype=torch.float16)
    ph = torch.randn(N, hidden, device=device, dtype=torch.float16)
    
    v2_time = benchmark_kernel(
        sedac_cuda_v2.fused_entropy_decision_v2,
        (logits, h, ph, 3.0, 1.0, 0.5, 0.7),
        "V2-FP16"
    )
    
    target = "OK" if v2_time < 1.0 else "MISS"
    print(f"{cfg['name']:<25} FP16: {v2_time:.3f}ms {target}")

# Token Router Benchmark
print("\n" + "=" * 70)
print("Token Router Split Benchmark")
print("=" * 70)

for cfg in configs[:4]:
    N, vocab, hidden = cfg["N"], cfg["vocab"], cfg["hidden"]
    
    h = torch.randn(N, hidden, device=device)
    mask = torch.rand(N, device=device) > 0.6
    
    v1_time = benchmark_kernel(
        sedac_cuda.token_router_split,
        (h, mask),
        "V1"
    )
    
    v2_time = benchmark_kernel(
        sedac_cuda_v2.token_router_split_v2,
        (h, mask),
        "V2"
    )
    
    speedup = v1_time / v2_time
    print(f"{cfg['name']:<25} V1: {v1_time:.3f}ms, V2: {v2_time:.3f}ms, Speedup: {speedup:.2f}x")

# Correctness Check
print("\n" + "=" * 70)
print("Correctness Verification")
print("=" * 70)

N, vocab, hidden = 128, 32000, 4096
logits = torch.randn(N, vocab, device=device)
h = torch.randn(N, hidden, device=device)
ph = torch.randn(N, hidden, device=device)

e1, c1, d1, l1 = sedac_cuda.fused_entropy_decision(logits, h, ph, 3.0, 1.0, 0.5, 0.7)
e2, c2, d2, l2 = sedac_cuda_v2.fused_entropy_decision_v2(logits, h, ph, 3.0, 1.0, 0.5, 0.7)

entropy_diff = (e1 - e2).abs().max().item()
conf_diff = (c1 - c2).abs().max().item()
decision_match = (d1 == d2).all().item()

print(f"Entropy max diff: {entropy_diff:.6f}")
print(f"Confidence max diff: {conf_diff:.6f}")
print(f"Decision match: {decision_match}")
print(f"Correctness: {'PASS' if entropy_diff < 0.01 and decision_match else 'FAIL'}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
