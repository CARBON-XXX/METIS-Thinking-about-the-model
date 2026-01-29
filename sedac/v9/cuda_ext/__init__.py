"""
SEDAC V9.0 - CUDA Extension Python Interface

自动检测并加载编译好的CUDA算子，提供Fallback实现。

Usage:
    from sedac.v9.cuda_ext import (
        fused_entropy_decision,
        token_router_split,
        token_router_merge,
        kv_cache_update,
        CUDA_KERNELS_AVAILABLE,
    )
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional
import logging
import math

logger = logging.getLogger(__name__)

# 尝试加载编译好的CUDA模块
CUDA_KERNELS_AVAILABLE = False
_sedac_cuda = None

try:
    import sedac_cuda as _sedac_cuda
    CUDA_KERNELS_AVAILABLE = True
    logger.info("✅ SEDAC CUDA Kernels Loaded Successfully!")
except ImportError as e:
    logger.warning(f"❌ CUDA Kernels not found ({e}), using PyTorch fallback.")
    logger.warning("To compile: cd sedac/v9/cuda_ext && pip install .")


# ============================================================================
# Fallback Implementations (Pure PyTorch)
# ============================================================================

def _fused_entropy_decision_fallback(
    logits: torch.Tensor,
    hidden: torch.Tensor,
    prev_hidden: torch.Tensor,
    mean_entropy: float,
    std_entropy: float,
    layer_progress: float,
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch Fallback - 优化版"""
    
    # 1. Entropy (数值稳定)
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -torch.sum(probs * log_probs, dim=-1) / math.log(2)
    
    # 2. Stability
    diff = hidden - prev_hidden
    diff_norm = torch.norm(diff, p=2, dim=-1)
    hidden_norm = torch.norm(hidden, p=2, dim=-1)
    stability = 1.0 / (1.0 + diff_norm / (hidden_norm + 1e-6))
    
    # 3. Confidence
    z_score = (mean_entropy - entropy) / (std_entropy + 1e-6)
    confidence = torch.sigmoid(z_score * 2.0)
    
    # 4. Cognitive Load
    cognitive_load = (1.0 - confidence) * 0.5 + (1.0 - stability) * 0.3 + (1.0 - layer_progress) * 0.2
    
    # 5. Decision
    dynamic_threshold = threshold - layer_progress * 0.2
    exit_score = confidence * stability * layer_progress
    decision = exit_score > dynamic_threshold
    
    return entropy, confidence, decision, cognitive_load


def _token_router_split_fallback(
    hidden: torch.Tensor,
    decision_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch Fallback"""
    
    active_indices = (~decision_mask).nonzero(as_tuple=False).squeeze(-1)
    exit_indices = decision_mask.nonzero(as_tuple=False).squeeze(-1)
    
    if active_indices.numel() > 0:
        active_hidden = hidden[active_indices]
    else:
        active_hidden = hidden.new_empty(0, hidden.shape[1])
    
    if exit_indices.numel() > 0:
        exit_hidden = hidden[exit_indices]
    else:
        exit_hidden = hidden.new_empty(0, hidden.shape[1])
    
    return active_hidden, active_indices, exit_hidden, exit_indices


def _token_router_merge_fallback(
    active_hidden: torch.Tensor,
    active_indices: torch.Tensor,
    exit_hidden: torch.Tensor,
    exit_indices: torch.Tensor,
    total_size: int,
) -> torch.Tensor:
    """PyTorch Fallback"""
    
    H = active_hidden.shape[1] if active_hidden.numel() > 0 else exit_hidden.shape[1]
    device = active_hidden.device if active_hidden.numel() > 0 else exit_hidden.device
    dtype = active_hidden.dtype if active_hidden.numel() > 0 else exit_hidden.dtype
    
    output = torch.empty(total_size, H, device=device, dtype=dtype)
    
    if active_hidden.numel() > 0:
        output.index_copy_(0, active_indices, active_hidden)
    
    if exit_hidden.numel() > 0:
        output.index_copy_(0, exit_indices, exit_hidden)
    
    return output


def _kv_cache_update_fallback(
    hidden: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """PyTorch Fallback"""
    
    k_proj = torch.mm(hidden, wk)
    v_proj = torch.mm(hidden, wv)
    
    k_cache.index_copy_(0, positions, k_proj)
    v_cache.index_copy_(0, positions, v_proj)


# ============================================================================
# Public API
# ============================================================================

def fused_entropy_decision(
    logits: torch.Tensor,
    hidden: torch.Tensor,
    prev_hidden: torch.Tensor,
    mean_entropy: float,
    std_entropy: float,
    layer_progress: float,
    threshold: float = 0.7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    融合熵决策内核
    
    Args:
        logits: [N, vocab_size] 模型输出logits
        hidden: [N, hidden_size] 当前隐藏状态
        prev_hidden: [N, hidden_size] 上一层隐藏状态
        mean_entropy: 历史熵均值
        std_entropy: 历史熵标准差
        layer_progress: 层进度 (0~1)
        threshold: 退出阈值
    
    Returns:
        entropy: [N] 信息熵
        confidence: [N] 置信度
        decision: [N] bool, True=退出
        cognitive_load: [N] 认知负载
    """
    if CUDA_KERNELS_AVAILABLE and logits.is_cuda:
        return _sedac_cuda.fused_entropy_decision(
            logits.contiguous(),
            hidden.contiguous(),
            prev_hidden.contiguous(),
            mean_entropy, std_entropy,
            layer_progress, threshold
        )
    else:
        return _fused_entropy_decision_fallback(
            logits, hidden, prev_hidden,
            mean_entropy, std_entropy,
            layer_progress, threshold
        )


def token_router_split(
    hidden: torch.Tensor,
    decision_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Token路由分割
    
    Args:
        hidden: [N, hidden_size]
        decision_mask: [N] bool, True=退出
    
    Returns:
        active_hidden: [n_active, hidden_size]
        active_indices: [n_active]
        exit_hidden: [n_exit, hidden_size]
        exit_indices: [n_exit]
    """
    if CUDA_KERNELS_AVAILABLE and hidden.is_cuda:
        return _sedac_cuda.token_router_split(
            hidden.contiguous(),
            decision_mask.contiguous()
        )
    else:
        return _token_router_split_fallback(hidden, decision_mask)


def token_router_merge(
    active_hidden: torch.Tensor,
    active_indices: torch.Tensor,
    exit_hidden: torch.Tensor,
    exit_indices: torch.Tensor,
    total_size: int,
) -> torch.Tensor:
    """
    Token路由合并
    
    Args:
        active_hidden: [n_active, hidden_size]
        active_indices: [n_active]
        exit_hidden: [n_exit, hidden_size]
        exit_indices: [n_exit]
        total_size: 原始token数量
    
    Returns:
        merged: [total_size, hidden_size]
    """
    if CUDA_KERNELS_AVAILABLE and active_hidden.is_cuda:
        return _sedac_cuda.token_router_merge(
            active_hidden.contiguous(),
            active_indices.contiguous().to(torch.int32),
            exit_hidden.contiguous(),
            exit_indices.contiguous().to(torch.int32),
            total_size
        )
    else:
        return _token_router_merge_fallback(
            active_hidden, active_indices,
            exit_hidden, exit_indices, total_size
        )


def kv_cache_update(
    hidden: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """
    KV缓存更新 (Paged Attention风格)
    
    Args:
        hidden: [N, hidden_size]
        wk: [hidden_size, kv_dim]
        wv: [hidden_size, kv_dim]
        k_cache: [max_seq, kv_dim] (inplace修改)
        v_cache: [max_seq, kv_dim] (inplace修改)
        positions: [N] 写入位置
    """
    if CUDA_KERNELS_AVAILABLE and hidden.is_cuda:
        _sedac_cuda.kv_cache_update(
            hidden.contiguous(),
            wk.contiguous(),
            wv.contiguous(),
            k_cache,
            v_cache,
            positions.contiguous().to(torch.int32)
        )
    else:
        _kv_cache_update_fallback(hidden, wk, wv, k_cache, v_cache, positions)


# ============================================================================
# Benchmark
# ============================================================================

def benchmark_kernels():
    """基准测试"""
    import time
    
    print("=" * 70)
    print("SEDAC V9.0 CUDA Kernels Benchmark")
    print("=" * 70)
    print(f"CUDA Kernels Available: {CUDA_KERNELS_AVAILABLE}")
    print(f"Backend: {'CUDA C++' if CUDA_KERNELS_AVAILABLE else 'PyTorch Fallback'}")
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping.")
        return
    
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
    for _ in range(10):
        fused_entropy_decision(logits, hidden, prev_hidden, 3.0, 1.0, 0.5)
    
    torch.cuda.synchronize()
    
    # Benchmark
    iterations = 100
    start = time.perf_counter()
    
    for _ in range(iterations):
        entropy, conf, decision, load = fused_entropy_decision(
            logits, hidden, prev_hidden, 3.0, 1.0, 0.5
        )
    
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000 / iterations
    
    print(f"\n[Fused Entropy Decision]")
    print(f"  Latency: {elapsed:.3f} ms")
    print(f"  Throughput: {N / elapsed * 1000:.0f} tokens/sec")
    print(f"  Exit ratio: {decision.float().mean().item()*100:.1f}%")
    
    # Token Router
    exit_mask = torch.rand(N, device=device) > 0.6
    
    for _ in range(10):
        token_router_split(hidden, exit_mask)
    
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        active_h, active_i, exit_h, exit_i = token_router_split(hidden, exit_mask)
    torch.cuda.synchronize()
    
    elapsed = (time.perf_counter() - start) * 1000 / iterations
    
    print(f"\n[Token Router Split]")
    print(f"  Latency: {elapsed:.3f} ms")
    print(f"  Active: {active_h.shape[0]}, Exit: {exit_h.shape[0]}")
    
    print("\n" + "=" * 70)
    if CUDA_KERNELS_AVAILABLE:
        print("✅ 使用CUDA Kernels - 预期: <0.5ms")
    else:
        print("⚠️ 使用PyTorch Fallback - 性能较差")
        print("   编译CUDA Kernels: cd sedac/v9/cuda_ext && pip install .")
    print("=" * 70)


__all__ = [
    "fused_entropy_decision",
    "token_router_split",
    "token_router_merge",
    "kv_cache_update",
    "CUDA_KERNELS_AVAILABLE",
    "benchmark_kernels",
]


if __name__ == "__main__":
    benchmark_kernels()
