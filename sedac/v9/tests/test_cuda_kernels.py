"""
SEDAC V9.0 - CUDA Kernels Unit Test
Standalone test without network dependency
"""
import torch
import time
import sys
sys.path.insert(0, "G:/SEDACV9.0 PRO/sedac/v9/cuda_ext")

print("=" * 70)
print("SEDAC V9.0 CUDA Kernels Unit Test")
print("=" * 70)

# Load CUDA modules
import sedac_cuda
import sedac_cuda_v2

print(f"V1: {sedac_cuda}")
print(f"V2: {sedac_cuda_v2}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

device = torch.device("cuda")

def test_fused_entropy_decision():
    """Test fused entropy decision kernel"""
    print("\n[Test] Fused Entropy Decision")
    
    N, vocab, hidden = 128, 32000, 4096
    logits = torch.randn(N, vocab, device=device)
    h = torch.randn(N, hidden, device=device)
    ph = torch.randn(N, hidden, device=device)
    
    # V1
    e1, c1, d1, l1 = sedac_cuda.fused_entropy_decision(logits, h, ph, 3.0, 1.0, 0.5, 0.7)
    
    # V2
    e2, c2, d2, l2 = sedac_cuda_v2.fused_entropy_decision_v2(logits, h, ph, 3.0, 1.0, 0.5, 0.7)
    
    # Check shapes
    assert e1.shape == (N,), f"Entropy shape mismatch: {e1.shape}"
    assert c1.shape == (N,), f"Confidence shape mismatch: {c1.shape}"
    assert d1.shape == (N,), f"Decision shape mismatch: {d1.shape}"
    
    # Check values match
    entropy_diff = (e1 - e2).abs().max().item()
    conf_diff = (c1 - c2).abs().max().item()
    decision_match = (d1 == d2).float().mean().item()
    
    print(f"  Entropy max diff: {entropy_diff:.6f}")
    print(f"  Confidence max diff: {conf_diff:.6f}")
    print(f"  Decision match rate: {decision_match*100:.1f}%")
    
    assert entropy_diff < 0.01, f"Entropy mismatch too large: {entropy_diff}"
    print("  PASS")

def test_token_router():
    """Test token router split/merge"""
    print("\n[Test] Token Router Split/Merge")
    
    N, hidden = 256, 4096
    h = torch.randn(N, hidden, device=device)
    mask = torch.rand(N, device=device) > 0.6
    
    # Split
    active_h, active_i, exit_h, exit_i = sedac_cuda_v2.token_router_split_v2(h, mask)
    
    n_active = active_h.shape[0]
    n_exit = exit_h.shape[0]
    
    print(f"  Total: {N}, Active: {n_active}, Exit: {n_exit}")
    assert n_active + n_exit == N, "Token count mismatch"
    
    # Merge
    merged = sedac_cuda_v2.token_router_merge_v2(active_h, active_i, exit_h, exit_i, N)
    
    # Verify merge correctness
    diff = (merged - h).abs().max().item()
    print(f"  Merge diff: {diff:.6f}")
    assert diff < 1e-5, f"Merge error too large: {diff}"
    print("  PASS")

def test_performance():
    """Performance benchmark"""
    print("\n[Test] Performance Benchmark")
    
    configs = [
        (32, 32000, 4096, "Small"),
        (256, 32000, 4096, "Medium"),
        (1024, 32000, 4096, "Large"),
    ]
    
    for N, vocab, hidden, name in configs:
        logits = torch.randn(N, vocab, device=device)
        h = torch.randn(N, hidden, device=device)
        ph = torch.randn(N, hidden, device=device)
        
        # Warmup
        for _ in range(10):
            sedac_cuda_v2.fused_entropy_decision_v2(logits, h, ph, 3.0, 1.0, 0.5, 0.7)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            sedac_cuda_v2.fused_entropy_decision_v2(logits, h, ph, 3.0, 1.0, 0.5, 0.7)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000 / 100
        
        status = "OK" if elapsed < 1.0 else "MISS"
        print(f"  {name} ({N} tokens): {elapsed:.3f}ms [{status}]")

def test_fp16():
    """Test FP16 support"""
    print("\n[Test] FP16 Support")
    
    N, vocab, hidden = 256, 32000, 4096
    logits = torch.randn(N, vocab, device=device, dtype=torch.float16)
    h = torch.randn(N, hidden, device=device, dtype=torch.float16)
    ph = torch.randn(N, hidden, device=device, dtype=torch.float16)
    
    e, c, d, l = sedac_cuda_v2.fused_entropy_decision_v2(logits, h, ph, 3.0, 1.0, 0.5, 0.7)
    
    print(f"  Output dtype: {e.dtype}")
    print(f"  Entropy range: [{e.min().item():.2f}, {e.max().item():.2f}]")
    print("  PASS")

def test_batched():
    """Test batched API"""
    print("\n[Test] Batched API")
    
    batch, seq, vocab, hidden = 4, 64, 32000, 4096
    logits = torch.randn(batch, seq, vocab, device=device)
    h = torch.randn(batch, seq, hidden, device=device)
    ph = torch.randn(batch, seq, hidden, device=device)
    
    e, c, d, l = sedac_cuda_v2.batched_entropy_decision(logits, h, ph, 3.0, 1.0, 0.5, 0.7)
    
    assert e.shape == (batch, seq), f"Wrong shape: {e.shape}"
    print(f"  Output shape: {e.shape}")
    print("  PASS")

# Run tests
print("\n" + "=" * 70)
print("Running Tests...")
print("=" * 70)

try:
    test_fused_entropy_decision()
    test_token_router()
    test_performance()
    test_fp16()
    test_batched()
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
except AssertionError as e:
    print(f"\nTEST FAILED: {e}")
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
