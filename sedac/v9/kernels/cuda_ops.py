"""
SEDAC V9.0 - CUDA操作封装

提供高性能CUDA内核的Python接口
支持:
1. Triton JIT编译
2. 自定义CUDA扩展 (通过torch.utils.cpp_extension)
3. PyTorch Fallback
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import logging
import math

logger = logging.getLogger(__name__)

# 检查CUDA可用性
CUDA_AVAILABLE = torch.cuda.is_available()

# 检查Triton可用性
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    logger.info("Triton not available. Using PyTorch fallback.")


# ============================================================================
# 1. Fused Entropy Decision Kernel
# ============================================================================

if TRITON_AVAILABLE:
    
    @triton.jit
    def _fused_entropy_decision_kernel(
        # 输入指针
        logits_ptr,          # [N, vocab_size]
        hidden_ptr,          # [N, hidden_size]
        prev_hidden_ptr,     # [N, hidden_size]
        # 输出指针
        entropy_ptr,         # [N]
        confidence_ptr,      # [N]
        stability_ptr,       # [N]
        exit_mask_ptr,       # [N]
        # 统计量
        entropy_mean,
        entropy_std,
        # 参数
        layer_progress,
        exit_threshold,
        # 维度
        N,
        vocab_size,
        hidden_size,
        # Block配置
        VOCAB_BLOCK: tl.constexpr,
        HIDDEN_BLOCK: tl.constexpr,
    ):
        """
        融合的熵计算+决策Kernel
        
        一次调用完成:
        1. 数值稳定的Softmax
        2. 信息熵计算
        3. Hidden state稳定性
        4. 置信度计算
        5. 退出决策
        
        全程GPU，零CPU同步
        """
        pid = tl.program_id(0)
        
        if pid >= N:
            return
        
        # ========== 1. 熵计算 (数值稳定) ==========
        logits_offset = pid * vocab_size
        
        # 找max
        max_logit = tl.float32(-1e9)
        for i in range(0, vocab_size, VOCAB_BLOCK):
            offs = i + tl.arange(0, VOCAB_BLOCK)
            mask = offs < vocab_size
            vals = tl.load(logits_ptr + logits_offset + offs, mask=mask, other=-1e9)
            max_logit = tl.maximum(max_logit, tl.max(vals, axis=0))
        
        # sum(exp(x - max))
        sum_exp = tl.float32(0.0)
        for i in range(0, vocab_size, VOCAB_BLOCK):
            offs = i + tl.arange(0, VOCAB_BLOCK)
            mask = offs < vocab_size
            vals = tl.load(logits_ptr + logits_offset + offs, mask=mask, other=-1e9)
            sum_exp += tl.sum(tl.exp(vals - max_logit), axis=0)
        
        log_sum_exp = tl.log(sum_exp)
        
        # H = -sum(p * log(p)) = log(Z) - E[x]/Z
        weighted_sum = tl.float32(0.0)
        for i in range(0, vocab_size, VOCAB_BLOCK):
            offs = i + tl.arange(0, VOCAB_BLOCK)
            mask = offs < vocab_size
            vals = tl.load(logits_ptr + logits_offset + offs, mask=mask, other=0.0)
            probs = tl.exp(vals - max_logit) / sum_exp
            # p * log(p), 避免log(0)
            log_probs = vals - max_logit - log_sum_exp
            weighted_sum += tl.sum(probs * log_probs, axis=0)
        
        entropy = -weighted_sum / tl.log(tl.float32(2.0))  # 转换为bits
        
        # ========== 2. 稳定性计算 ==========
        hidden_offset = pid * hidden_size
        
        diff_sq = tl.float32(0.0)
        norm_sq = tl.float32(0.0)
        
        for i in range(0, hidden_size, HIDDEN_BLOCK):
            offs = i + tl.arange(0, HIDDEN_BLOCK)
            mask = offs < hidden_size
            
            curr = tl.load(hidden_ptr + hidden_offset + offs, mask=mask, other=0.0)
            prev = tl.load(prev_hidden_ptr + hidden_offset + offs, mask=mask, other=0.0)
            
            diff = curr - prev
            diff_sq += tl.sum(diff * diff, axis=0)
            norm_sq += tl.sum(curr * curr, axis=0)
        
        # stability = 1 / (1 + ||diff|| / ||curr||)
        stability = 1.0 / (1.0 + tl.sqrt(diff_sq) / (tl.sqrt(norm_sq) + 1e-6))
        
        # ========== 3. 置信度计算 ==========
        # z-score归一化
        z_score = (entropy_mean - entropy) / (entropy_std + 1e-6)
        confidence = 1.0 / (1.0 + tl.exp(-z_score * 2.0))  # sigmoid
        
        # ========== 4. 退出决策 ==========
        # 动态阈值：层越深，阈值越低
        dynamic_threshold = exit_threshold - layer_progress * 0.2
        exit_score = confidence * stability * (0.5 + layer_progress * 0.5)
        should_exit = exit_score > dynamic_threshold
        
        # ========== 存储结果 ==========
        tl.store(entropy_ptr + pid, entropy)
        tl.store(confidence_ptr + pid, confidence)
        tl.store(stability_ptr + pid, stability)
        tl.store(exit_mask_ptr + pid, should_exit)


def fused_entropy_decision_triton(
    logits: torch.Tensor,         # [batch, seq_len, vocab_size] or [N, vocab_size]
    hidden_states: torch.Tensor,  # [batch, seq_len, hidden] or [N, hidden]
    prev_hidden: torch.Tensor,    # same shape
    entropy_mean: float,
    entropy_std: float,
    layer_progress: float,
    exit_threshold: float = 0.7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Triton实现的融合熵决策
    """
    # 展平
    original_shape = hidden_states.shape[:-1]
    N = logits.numel() // logits.shape[-1]
    vocab_size = logits.shape[-1]
    hidden_size = hidden_states.shape[-1]
    
    logits_flat = logits.view(N, vocab_size).contiguous()
    hidden_flat = hidden_states.view(N, hidden_size).contiguous()
    prev_flat = prev_hidden.view(N, hidden_size).contiguous()
    
    device = logits.device
    
    # 分配输出
    entropy = torch.empty(N, device=device, dtype=torch.float32)
    confidence = torch.empty(N, device=device, dtype=torch.float32)
    stability = torch.empty(N, device=device, dtype=torch.float32)
    exit_mask = torch.empty(N, device=device, dtype=torch.bool)
    
    # 启动Kernel
    VOCAB_BLOCK = min(1024, triton.next_power_of_2(vocab_size))
    HIDDEN_BLOCK = min(256, triton.next_power_of_2(hidden_size))
    grid = (N,)
    
    _fused_entropy_decision_kernel[grid](
        logits_flat, hidden_flat, prev_flat,
        entropy, confidence, stability, exit_mask,
        entropy_mean, entropy_std,
        layer_progress, exit_threshold,
        N, vocab_size, hidden_size,
        VOCAB_BLOCK, HIDDEN_BLOCK,
    )
    
    return entropy, confidence, stability, exit_mask


def fused_entropy_decision_pytorch(
    logits: torch.Tensor,
    hidden_states: torch.Tensor,
    prev_hidden: torch.Tensor,
    entropy_mean: float,
    entropy_std: float,
    layer_progress: float,
    exit_threshold: float = 0.7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PyTorch Fallback实现
    """
    original_shape = hidden_states.shape[:-1]
    N = logits.numel() // logits.shape[-1]
    
    logits_flat = logits.view(N, -1)
    hidden_flat = hidden_states.view(N, -1)
    prev_flat = prev_hidden.view(N, -1)
    
    # 1. 熵计算
    probs = F.softmax(logits_flat, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1) / math.log(2)
    
    # 2. 稳定性
    diff = hidden_flat - prev_flat
    stability = 1.0 / (1.0 + torch.norm(diff, dim=-1) / (torch.norm(hidden_flat, dim=-1) + 1e-6))
    
    # 3. 置信度
    z_score = (entropy_mean - entropy) / (entropy_std + 1e-6)
    confidence = torch.sigmoid(z_score * 2.0)
    
    # 4. 退出决策
    dynamic_threshold = exit_threshold - layer_progress * 0.2
    exit_score = confidence * stability * (0.5 + layer_progress * 0.5)
    exit_mask = exit_score > dynamic_threshold
    
    return entropy, confidence, stability, exit_mask


def fused_entropy_decision(
    logits: torch.Tensor,
    hidden_states: torch.Tensor,
    prev_hidden: torch.Tensor,
    entropy_mean: float,
    entropy_std: float,
    layer_progress: float,
    exit_threshold: float = 0.7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    融合熵决策（自动选择后端）
    """
    if TRITON_AVAILABLE and logits.is_cuda:
        return fused_entropy_decision_triton(
            logits, hidden_states, prev_hidden,
            entropy_mean, entropy_std, layer_progress, exit_threshold
        )
    return fused_entropy_decision_pytorch(
        logits, hidden_states, prev_hidden,
        entropy_mean, entropy_std, layer_progress, exit_threshold
    )


# ============================================================================
# 2. KV Cache Update Kernel
# ============================================================================

if TRITON_AVAILABLE:
    
    @triton.jit
    def _kv_cache_update_kernel(
        # 输入
        hidden_ptr,           # [N, hidden_size]
        wk_ptr,               # [hidden_size, kv_dim]
        wv_ptr,               # [hidden_size, kv_dim]
        # 输出 (Paged Attention风格)
        key_cache_ptr,        # [max_pages, page_size, num_heads, head_dim]
        value_cache_ptr,      # same
        # 索引
        page_indices_ptr,     # [N] - 每个token的page索引
        slot_indices_ptr,     # [N] - 每个token在page内的slot
        # 维度
        N,
        hidden_size,
        num_heads,
        head_dim,
        page_size,
        # Block配置
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        KV Cache更新Kernel
        
        直接将KV投影结果写入非连续的Cache地址
        支持Paged Attention风格的内存布局
        """
        pid = tl.program_id(0)
        
        if pid >= N:
            return
        
        # 读取索引
        page_idx = tl.load(page_indices_ptr + pid)
        slot_idx = tl.load(slot_indices_ptr + pid)
        
        kv_dim = num_heads * head_dim
        
        # 计算KV投影
        for head in range(num_heads):
            for d in range(0, head_dim, BLOCK_SIZE):
                offs = d + tl.arange(0, BLOCK_SIZE)
                mask = offs < head_dim
                
                kv_col = head * head_dim + offs
                
                # K投影: hidden @ W_k
                k_acc = tl.float32(0.0)
                v_acc = tl.float32(0.0)
                
                for h in range(hidden_size):
                    hidden_val = tl.load(hidden_ptr + pid * hidden_size + h)
                    wk_val = tl.load(wk_ptr + h * kv_dim + kv_col, mask=mask, other=0.0)
                    wv_val = tl.load(wv_ptr + h * kv_dim + kv_col, mask=mask, other=0.0)
                    k_acc += hidden_val * wk_val
                    v_acc += hidden_val * wv_val
                
                # 写入Cache
                cache_offset = (page_idx * page_size + slot_idx) * num_heads * head_dim + head * head_dim + offs
                tl.store(key_cache_ptr + cache_offset, k_acc, mask=mask)
                tl.store(value_cache_ptr + cache_offset, v_acc, mask=mask)


def kv_cache_update_triton(
    hidden_states: torch.Tensor,  # [N, hidden_size]
    wk: torch.Tensor,             # [hidden_size, kv_dim]
    wv: torch.Tensor,             # [hidden_size, kv_dim]
    key_cache: torch.Tensor,      # [max_pages, page_size, num_heads, head_dim]
    value_cache: torch.Tensor,
    page_indices: torch.Tensor,   # [N]
    slot_indices: torch.Tensor,   # [N]
):
    """
    Triton KV Cache更新
    """
    N = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    num_heads = key_cache.shape[2]
    head_dim = key_cache.shape[3]
    page_size = key_cache.shape[1]
    
    BLOCK_SIZE = min(32, head_dim)
    grid = (N,)
    
    _kv_cache_update_kernel[grid](
        hidden_states, wk, wv,
        key_cache, value_cache,
        page_indices, slot_indices,
        N, hidden_size, num_heads, head_dim, page_size,
        BLOCK_SIZE,
    )


def kv_cache_update_pytorch(
    hidden_states: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    page_indices: torch.Tensor,
    slot_indices: torch.Tensor,
):
    """
    PyTorch KV Cache更新
    """
    N = hidden_states.shape[0]
    num_heads = key_cache.shape[2]
    head_dim = key_cache.shape[3]
    
    # 计算KV
    key = torch.matmul(hidden_states, wk)  # [N, kv_dim]
    value = torch.matmul(hidden_states, wv)
    
    # 重塑
    key = key.view(N, num_heads, head_dim)
    value = value.view(N, num_heads, head_dim)
    
    # 写入Cache (scatter)
    for i in range(N):
        page_idx = page_indices[i].item()
        slot_idx = slot_indices[i].item()
        key_cache[page_idx, slot_idx] = key[i]
        value_cache[page_idx, slot_idx] = value[i]


def kv_cache_update(
    hidden_states: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    page_indices: torch.Tensor,
    slot_indices: torch.Tensor,
):
    """
    KV Cache更新（自动选择后端）
    """
    if TRITON_AVAILABLE and hidden_states.is_cuda:
        return kv_cache_update_triton(
            hidden_states, wk, wv, key_cache, value_cache, page_indices, slot_indices
        )
    return kv_cache_update_pytorch(
        hidden_states, wk, wv, key_cache, value_cache, page_indices, slot_indices
    )


# ============================================================================
# 3. Token Router Kernel
# ============================================================================

if TRITON_AVAILABLE:
    
    @triton.jit
    def _token_router_split_kernel(
        # 输入
        hidden_ptr,           # [N, hidden_size]
        exit_mask_ptr,        # [N] bool
        # 输出
        active_hidden_ptr,    # [N_active, hidden_size] (预分配最大)
        active_indices_ptr,   # [N_active]
        exit_hidden_ptr,      # [N_exit, hidden_size]
        exit_indices_ptr,     # [N_exit]
        # 计数器 (atomic)
        active_count_ptr,
        exit_count_ptr,
        # 维度
        N,
        hidden_size,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Token Router Split Kernel
        
        将Batch分为Active组和Exit组
        """
        pid = tl.program_id(0)
        
        if pid >= N:
            return
        
        should_exit = tl.load(exit_mask_ptr + pid)
        
        if should_exit:
            # 原子递增exit计数
            idx = tl.atomic_add(exit_count_ptr, 1)
            tl.store(exit_indices_ptr + idx, pid)
            
            # 复制hidden
            for i in range(0, hidden_size, BLOCK_SIZE):
                offs = i + tl.arange(0, BLOCK_SIZE)
                mask = offs < hidden_size
                vals = tl.load(hidden_ptr + pid * hidden_size + offs, mask=mask)
                tl.store(exit_hidden_ptr + idx * hidden_size + offs, vals, mask=mask)
        else:
            # 原子递增active计数
            idx = tl.atomic_add(active_count_ptr, 1)
            tl.store(active_indices_ptr + idx, pid)
            
            # 复制hidden
            for i in range(0, hidden_size, BLOCK_SIZE):
                offs = i + tl.arange(0, BLOCK_SIZE)
                mask = offs < hidden_size
                vals = tl.load(hidden_ptr + pid * hidden_size + offs, mask=mask)
                tl.store(active_hidden_ptr + idx * hidden_size + offs, vals, mask=mask)
    
    
    @triton.jit
    def _token_router_merge_kernel(
        # 输入
        active_hidden_ptr,    # [N_active, hidden_size]
        active_indices_ptr,   # [N_active]
        exit_hidden_ptr,      # [N_exit, hidden_size]
        exit_indices_ptr,     # [N_exit]
        # 输出
        merged_ptr,           # [N, hidden_size]
        # 维度
        N_active,
        N_exit,
        hidden_size,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Token Router Merge Kernel
        
        将Active和Exit组合并回原始顺序
        """
        pid = tl.program_id(0)
        
        if pid < N_active:
            # Active组
            src_ptr = active_hidden_ptr
            idx_ptr = active_indices_ptr
            local_idx = pid
        elif pid < N_active + N_exit:
            # Exit组
            src_ptr = exit_hidden_ptr
            idx_ptr = exit_indices_ptr
            local_idx = pid - N_active
        else:
            return
        
        original_idx = tl.load(idx_ptr + local_idx)
        
        for i in range(0, hidden_size, BLOCK_SIZE):
            offs = i + tl.arange(0, BLOCK_SIZE)
            mask = offs < hidden_size
            vals = tl.load(src_ptr + local_idx * hidden_size + offs, mask=mask)
            tl.store(merged_ptr + original_idx * hidden_size + offs, vals, mask=mask)


def token_router_split_triton(
    hidden_states: torch.Tensor,  # [N, hidden_size]
    exit_mask: torch.Tensor,      # [N] bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Triton Token Router Split
    """
    N, hidden_size = hidden_states.shape
    device = hidden_states.device
    
    # 预分配输出
    active_hidden = torch.empty(N, hidden_size, device=device, dtype=hidden_states.dtype)
    active_indices = torch.empty(N, device=device, dtype=torch.long)
    exit_hidden = torch.empty(N, hidden_size, device=device, dtype=hidden_states.dtype)
    exit_indices = torch.empty(N, device=device, dtype=torch.long)
    
    active_count = torch.zeros(1, device=device, dtype=torch.int32)
    exit_count = torch.zeros(1, device=device, dtype=torch.int32)
    
    BLOCK_SIZE = min(128, hidden_size)
    grid = (N,)
    
    _token_router_split_kernel[grid](
        hidden_states, exit_mask,
        active_hidden, active_indices,
        exit_hidden, exit_indices,
        active_count, exit_count,
        N, hidden_size, BLOCK_SIZE,
    )
    
    # 裁剪到实际大小
    n_active = active_count.item()
    n_exit = exit_count.item()
    
    return (
        active_hidden[:n_active],
        active_indices[:n_active],
        exit_hidden[:n_exit],
        exit_indices[:n_exit],
    )


def token_router_split_pytorch(
    hidden_states: torch.Tensor,
    exit_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PyTorch Token Router Split
    """
    active_mask = ~exit_mask
    
    active_indices = active_mask.nonzero(as_tuple=True)[0]
    exit_indices = exit_mask.nonzero(as_tuple=True)[0]
    
    active_hidden = hidden_states[active_indices]
    exit_hidden = hidden_states[exit_indices]
    
    return active_hidden, active_indices, exit_hidden, exit_indices


def token_router_split(
    hidden_states: torch.Tensor,
    exit_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Token Router Split（自动选择后端）
    """
    if TRITON_AVAILABLE and hidden_states.is_cuda:
        return token_router_split_triton(hidden_states, exit_mask)
    return token_router_split_pytorch(hidden_states, exit_mask)


def token_router_merge_triton(
    active_hidden: torch.Tensor,
    active_indices: torch.Tensor,
    exit_hidden: torch.Tensor,
    exit_indices: torch.Tensor,
    total_size: int,
) -> torch.Tensor:
    """
    Triton Token Router Merge
    """
    hidden_size = active_hidden.shape[1] if active_hidden.numel() > 0 else exit_hidden.shape[1]
    device = active_hidden.device if active_hidden.numel() > 0 else exit_hidden.device
    dtype = active_hidden.dtype if active_hidden.numel() > 0 else exit_hidden.dtype
    
    merged = torch.empty(total_size, hidden_size, device=device, dtype=dtype)
    
    N_active = active_hidden.shape[0]
    N_exit = exit_hidden.shape[0]
    
    if N_active + N_exit == 0:
        return merged
    
    BLOCK_SIZE = min(128, hidden_size)
    grid = (N_active + N_exit,)
    
    _token_router_merge_kernel[grid](
        active_hidden, active_indices,
        exit_hidden, exit_indices,
        merged,
        N_active, N_exit, hidden_size, BLOCK_SIZE,
    )
    
    return merged


def token_router_merge_pytorch(
    active_hidden: torch.Tensor,
    active_indices: torch.Tensor,
    exit_hidden: torch.Tensor,
    exit_indices: torch.Tensor,
    total_size: int,
) -> torch.Tensor:
    """
    PyTorch Token Router Merge
    """
    hidden_size = active_hidden.shape[1] if active_hidden.numel() > 0 else exit_hidden.shape[1]
    device = active_hidden.device if active_hidden.numel() > 0 else exit_hidden.device
    dtype = active_hidden.dtype if active_hidden.numel() > 0 else exit_hidden.dtype
    
    merged = torch.empty(total_size, hidden_size, device=device, dtype=dtype)
    
    if active_hidden.numel() > 0:
        merged[active_indices] = active_hidden
    if exit_hidden.numel() > 0:
        merged[exit_indices] = exit_hidden
    
    return merged


def token_router_merge(
    active_hidden: torch.Tensor,
    active_indices: torch.Tensor,
    exit_hidden: torch.Tensor,
    exit_indices: torch.Tensor,
    total_size: int,
) -> torch.Tensor:
    """
    Token Router Merge（自动选择后端）
    """
    if TRITON_AVAILABLE and (active_hidden.is_cuda if active_hidden.numel() > 0 else exit_hidden.is_cuda):
        return token_router_merge_triton(
            active_hidden, active_indices, exit_hidden, exit_indices, total_size
        )
    return token_router_merge_pytorch(
        active_hidden, active_indices, exit_hidden, exit_indices, total_size
    )


# ============================================================================
# Benchmark
# ============================================================================

def benchmark_kernels():
    """基准测试"""
    print("=" * 70)
    print("SEDAC V9.0 CUDA Kernels Benchmark")
    print("=" * 70)
    print(f"CUDA Available: {CUDA_AVAILABLE}")
    print(f"Triton Available: {TRITON_AVAILABLE}")
    
    if not CUDA_AVAILABLE:
        print("CUDA not available. Skipping benchmark.")
        return
    
    device = torch.device("cuda")
    
    # 测试配置
    N = 4096
    vocab_size = 32000
    hidden_size = 4096
    
    # 准备数据
    logits = torch.randn(N, vocab_size, device=device)
    hidden = torch.randn(N, hidden_size, device=device)
    prev_hidden = torch.randn(N, hidden_size, device=device)
    
    # Warmup
    for _ in range(3):
        fused_entropy_decision(logits, hidden, prev_hidden, 3.0, 1.0, 0.5)
    
    torch.cuda.synchronize()
    
    # Benchmark
    import time
    
    iterations = 100
    start = time.perf_counter()
    
    for _ in range(iterations):
        entropy, conf, stab, mask = fused_entropy_decision(
            logits, hidden, prev_hidden, 3.0, 1.0, 0.5
        )
    
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000 / iterations
    
    print(f"\nFused Entropy Decision:")
    print(f"  Input: {N} tokens, vocab={vocab_size}, hidden={hidden_size}")
    print(f"  Latency: {elapsed:.3f} ms")
    print(f"  Throughput: {N / elapsed * 1000:.0f} tokens/sec")
    print(f"  Exit ratio: {mask.float().mean().item()*100:.1f}%")
    
    # Token Router
    exit_mask = torch.rand(N, device=device) > 0.6
    
    for _ in range(3):
        token_router_split(hidden, exit_mask)
    
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        active_h, active_i, exit_h, exit_i = token_router_split(hidden, exit_mask)
    
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000 / iterations
    
    print(f"\nToken Router Split:")
    print(f"  Input: {N} tokens")
    print(f"  Latency: {elapsed:.3f} ms")
    print(f"  Active: {active_h.shape[0]}, Exit: {exit_h.shape[0]}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    benchmark_kernels()
