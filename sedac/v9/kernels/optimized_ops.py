"""
SEDAC V9.0 - 优化算子 (Optimized Ops)

解决 26.6ms → <0.5ms 的性能问题

方案:
1. torch.compile (PyTorch 2.0+) - 无需额外编译
2. CUDA Extension (torch.utils.cpp_extension) - 极致性能
3. Fused Operations - 减少Kernel Launch

关键优化:
- 消除CPU-GPU同步
- 单次Kernel完成: Softmax → Entropy → Decision
- 向量化内存访问
- 减少HBM读写
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging
import math
import os

logger = logging.getLogger(__name__)

# ============================================================================
# 检测环境
# ============================================================================

CUDA_AVAILABLE = torch.cuda.is_available()
TORCH_COMPILE_AVAILABLE = hasattr(torch, 'compile')

# 检测CUDA扩展编译能力
try:
    from torch.utils.cpp_extension import load_inline
    CUDA_EXTENSION_AVAILABLE = CUDA_AVAILABLE
except ImportError:
    CUDA_EXTENSION_AVAILABLE = False

logger.info(f"CUDA: {CUDA_AVAILABLE}, torch.compile: {TORCH_COMPILE_AVAILABLE}, CUDA Extension: {CUDA_EXTENSION_AVAILABLE}")


# ============================================================================
# 1. torch.compile 优化版本 (最快部署)
# ============================================================================

class FusedEntropyDecisionCompiled(nn.Module):
    """
    使用torch.compile优化的融合熵决策
    
    预期性能: 26ms → 1-2ms (10-20x加速)
    """
    
    def __init__(self):
        super().__init__()
        self._compiled = False
    
    def _fused_forward(
        self,
        logits: torch.Tensor,          # [N, vocab_size]
        hidden_states: torch.Tensor,   # [N, hidden_size]
        prev_hidden: torch.Tensor,     # [N, hidden_size]
        entropy_mean: float,
        entropy_std: float,
        layer_progress: float,
        exit_threshold: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        融合的前向传播 - 单次Kernel完成所有计算
        """
        # 1. 数值稳定的Softmax + Entropy (融合)
        # 使用log_softmax避免数值溢出
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        
        # H = -sum(p * log(p)) 用bits表示
        entropy = -torch.sum(probs * log_probs, dim=-1) / math.log(2)
        
        # 2. 稳定性计算 (向量化)
        diff = hidden_states - prev_hidden
        diff_norm = torch.norm(diff, dim=-1)
        hidden_norm = torch.norm(hidden_states, dim=-1)
        stability = 1.0 / (1.0 + diff_norm / (hidden_norm + 1e-6))
        
        # 3. 置信度 (z-score归一化 + sigmoid)
        z_score = (entropy_mean - entropy) / (entropy_std + 1e-6)
        confidence = torch.sigmoid(z_score * 2.0)
        
        # 4. 退出决策 (全GPU计算，无CPU同步)
        dynamic_threshold = exit_threshold - layer_progress * 0.2
        exit_score = confidence * stability * (0.5 + layer_progress * 0.5)
        exit_mask = exit_score > dynamic_threshold
        
        return entropy, confidence, stability, exit_mask
    
    def forward(
        self,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        prev_hidden: torch.Tensor,
        entropy_mean: float,
        entropy_std: float,
        layer_progress: float,
        exit_threshold: float = 0.7,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 首次调用时编译
        if not self._compiled and TORCH_COMPILE_AVAILABLE:
            try:
                self._fused_forward = torch.compile(
                    self._fused_forward,
                    mode="reduce-overhead",  # 最小化开销
                    fullgraph=True,          # 完整图优化
                )
                self._compiled = True
                logger.info("FusedEntropyDecision compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        
        return self._fused_forward(
            logits, hidden_states, prev_hidden,
            entropy_mean, entropy_std, layer_progress, exit_threshold
        )


# ============================================================================
# 2. CUDA C++ Extension (极致性能)
# ============================================================================

CUDA_KERNEL_SOURCE = '''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Fused Entropy Decision Kernel
// 单个Kernel完成: Softmax → Entropy → Stability → Decision
__global__ void fused_entropy_decision_kernel(
    const float* __restrict__ logits,      // [N, vocab_size]
    const float* __restrict__ hidden,      // [N, hidden_size]
    const float* __restrict__ prev_hidden, // [N, hidden_size]
    float* __restrict__ entropy,           // [N]
    float* __restrict__ confidence,        // [N]
    float* __restrict__ stability,         // [N]
    bool* __restrict__ exit_mask,          // [N]
    int N,
    int vocab_size,
    int hidden_size,
    float entropy_mean,
    float entropy_std,
    float layer_progress,
    float exit_threshold
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    
    // ========== 1. 熵计算 (数值稳定) ==========
    const float* logits_row = logits + tid * vocab_size;
    
    // 找max (用于数值稳定)
    float max_logit = -1e9f;
    for (int i = 0; i < vocab_size; i++) {
        max_logit = fmaxf(max_logit, logits_row[i]);
    }
    
    // sum(exp(x - max))
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        sum_exp += expf(logits_row[i] - max_logit);
    }
    
    float log_sum_exp = logf(sum_exp);
    
    // H = -sum(p * log(p))
    float H = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        float log_p = logits_row[i] - max_logit - log_sum_exp;
        float p = expf(log_p);
        H -= p * log_p;
    }
    
    // 转换为bits
    float ent = H / 0.693147f;  // ln(2)
    entropy[tid] = ent;
    
    // ========== 2. 稳定性计算 ==========
    const float* h_row = hidden + tid * hidden_size;
    const float* p_row = prev_hidden + tid * hidden_size;
    
    float diff_sq = 0.0f;
    float norm_sq = 0.0f;
    
    for (int i = 0; i < hidden_size; i++) {
        float diff = h_row[i] - p_row[i];
        diff_sq += diff * diff;
        norm_sq += h_row[i] * h_row[i];
    }
    
    float stab = 1.0f / (1.0f + sqrtf(diff_sq) / (sqrtf(norm_sq) + 1e-6f));
    stability[tid] = stab;
    
    // ========== 3. 置信度 ==========
    float z_score = (entropy_mean - ent) / (entropy_std + 1e-6f);
    float conf = 1.0f / (1.0f + expf(-z_score * 2.0f));  // sigmoid
    confidence[tid] = conf;
    
    // ========== 4. 退出决策 ==========
    float dynamic_threshold = exit_threshold - layer_progress * 0.2f;
    float exit_score = conf * stab * (0.5f + layer_progress * 0.5f);
    exit_mask[tid] = (exit_score > dynamic_threshold);
}

// C++ Wrapper
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_entropy_decision_cuda(
    torch::Tensor logits,
    torch::Tensor hidden,
    torch::Tensor prev_hidden,
    float entropy_mean,
    float entropy_std,
    float layer_progress,
    float exit_threshold
) {
    int N = logits.size(0);
    int vocab_size = logits.size(1);
    int hidden_size = hidden.size(1);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(logits.device());
    auto bool_options = torch::TensorOptions().dtype(torch::kBool).device(logits.device());
    
    auto entropy = torch::empty({N}, options);
    auto confidence = torch::empty({N}, options);
    auto stability = torch::empty({N}, options);
    auto exit_mask = torch::empty({N}, bool_options);
    
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    
    fused_entropy_decision_kernel<<<blocks, threads>>>(
        logits.data_ptr<float>(),
        hidden.data_ptr<float>(),
        prev_hidden.data_ptr<float>(),
        entropy.data_ptr<float>(),
        confidence.data_ptr<float>(),
        stability.data_ptr<float>(),
        exit_mask.data_ptr<bool>(),
        N, vocab_size, hidden_size,
        entropy_mean, entropy_std, layer_progress, exit_threshold
    );
    
    return std::make_tuple(entropy, confidence, stability, exit_mask);
}

// Token Router Split Kernel (零拷贝)
__global__ void token_router_split_kernel(
    const float* __restrict__ hidden,    // [N, hidden_size]
    const bool* __restrict__ exit_mask,  // [N]
    float* __restrict__ active_out,      // [N, hidden_size] (预分配最大)
    float* __restrict__ exit_out,        // [N, hidden_size]
    int* __restrict__ active_indices,    // [N]
    int* __restrict__ exit_indices,      // [N]
    int* __restrict__ active_count,      // [1]
    int* __restrict__ exit_count,        // [1]
    int N,
    int hidden_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    
    bool should_exit = exit_mask[tid];
    const float* src = hidden + tid * hidden_size;
    
    if (should_exit) {
        int idx = atomicAdd(exit_count, 1);
        exit_indices[idx] = tid;
        float* dst = exit_out + idx * hidden_size;
        for (int i = 0; i < hidden_size; i++) {
            dst[i] = src[i];
        }
    } else {
        int idx = atomicAdd(active_count, 1);
        active_indices[idx] = tid;
        float* dst = active_out + idx * hidden_size;
        for (int i = 0; i < hidden_size; i++) {
            dst[i] = src[i];
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int>
token_router_split_cuda(
    torch::Tensor hidden,
    torch::Tensor exit_mask
) {
    int N = hidden.size(0);
    int hidden_size = hidden.size(1);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(hidden.device());
    auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(hidden.device());
    
    auto active_out = torch::empty({N, hidden_size}, options);
    auto exit_out = torch::empty({N, hidden_size}, options);
    auto active_indices = torch::empty({N}, int_options);
    auto exit_indices = torch::empty({N}, int_options);
    auto active_count = torch::zeros({1}, int_options);
    auto exit_count = torch::zeros({1}, int_options);
    
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    
    token_router_split_kernel<<<blocks, threads>>>(
        hidden.data_ptr<float>(),
        exit_mask.data_ptr<bool>(),
        active_out.data_ptr<float>(),
        exit_out.data_ptr<float>(),
        active_indices.data_ptr<int>(),
        exit_indices.data_ptr<int>(),
        active_count.data_ptr<int>(),
        exit_count.data_ptr<int>(),
        N, hidden_size
    );
    
    int n_active = active_count.item<int>();
    int n_exit = exit_count.item<int>();
    
    return std::make_tuple(
        active_out.slice(0, 0, n_active),
        active_indices.slice(0, 0, n_active),
        exit_out.slice(0, 0, n_exit),
        exit_indices.slice(0, 0, n_exit),
        n_active, n_exit
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_entropy_decision", &fused_entropy_decision_cuda, "Fused Entropy Decision (CUDA)");
    m.def("token_router_split", &token_router_split_cuda, "Token Router Split (CUDA)");
}
'''

# JIT编译CUDA扩展
_cuda_module = None

def get_cuda_module():
    """获取CUDA扩展模块（JIT编译）"""
    global _cuda_module
    
    if _cuda_module is not None:
        return _cuda_module
    
    if not CUDA_EXTENSION_AVAILABLE:
        logger.warning("CUDA extension not available")
        return None
    
    try:
        from torch.utils.cpp_extension import load_inline
        
        _cuda_module = load_inline(
            name="sedac_cuda_ops",
            cpp_sources="",
            cuda_sources=CUDA_KERNEL_SOURCE,
            functions=["fused_entropy_decision", "token_router_split"],
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
        logger.info("CUDA extension compiled successfully")
        return _cuda_module
    except Exception as e:
        logger.error(f"Failed to compile CUDA extension: {e}")
        return None


# ============================================================================
# 3. 统一接口
# ============================================================================

class OptimizedSEDACOps:
    """
    优化的SEDAC算子
    
    自动选择最优后端:
    1. CUDA Extension (如果可用)
    2. torch.compile (如果可用)
    3. PyTorch Fallback
    """
    
    def __init__(self, force_backend: str = "auto"):
        self.backend = force_backend
        self._compiled_module = FusedEntropyDecisionCompiled()
        self._cuda_module = None
        
        if force_backend == "cuda" or (force_backend == "auto" and CUDA_EXTENSION_AVAILABLE):
            self._cuda_module = get_cuda_module()
        
        if self._cuda_module:
            self.backend = "cuda"
        elif TORCH_COMPILE_AVAILABLE:
            self.backend = "torch_compile"
        else:
            self.backend = "pytorch"
        
        logger.info(f"OptimizedSEDACOps using backend: {self.backend}")
    
    def fused_entropy_decision(
        self,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        prev_hidden: torch.Tensor,
        entropy_mean: float,
        entropy_std: float,
        layer_progress: float,
        exit_threshold: float = 0.7,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """融合熵决策"""
        
        # 展平
        original_shape = hidden_states.shape[:-1]
        N = logits.numel() // logits.shape[-1]
        vocab_size = logits.shape[-1]
        hidden_size = hidden_states.shape[-1]
        
        logits_flat = logits.view(N, vocab_size).contiguous()
        hidden_flat = hidden_states.view(N, hidden_size).contiguous()
        prev_flat = prev_hidden.view(N, hidden_size).contiguous()
        
        if self.backend == "cuda" and self._cuda_module:
            return self._cuda_module.fused_entropy_decision(
                logits_flat, hidden_flat, prev_flat,
                entropy_mean, entropy_std, layer_progress, exit_threshold
            )
        else:
            return self._compiled_module(
                logits_flat, hidden_flat, prev_flat,
                entropy_mean, entropy_std, layer_progress, exit_threshold
            )
    
    def token_router_split(
        self,
        hidden_states: torch.Tensor,
        exit_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Token路由分割"""
        
        if self.backend == "cuda" and self._cuda_module:
            result = self._cuda_module.token_router_split(
                hidden_states.contiguous(),
                exit_mask.contiguous()
            )
            return result[0], result[1].long(), result[2], result[3].long()
        else:
            # PyTorch fallback
            active_mask = ~exit_mask
            active_indices = active_mask.nonzero(as_tuple=True)[0]
            exit_indices = exit_mask.nonzero(as_tuple=True)[0]
            
            return (
                hidden_states[active_indices],
                active_indices,
                hidden_states[exit_indices],
                exit_indices,
            )


# ============================================================================
# 基准测试
# ============================================================================

def benchmark_optimized_ops():
    """基准测试优化算子"""
    
    print("=" * 70)
    print("SEDAC V9.0 Optimized Ops Benchmark")
    print("=" * 70)
    print(f"CUDA Available: {CUDA_AVAILABLE}")
    print(f"torch.compile Available: {TORCH_COMPILE_AVAILABLE}")
    print(f"CUDA Extension Available: {CUDA_EXTENSION_AVAILABLE}")
    
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
    
    ops = OptimizedSEDACOps(force_backend="auto")
    
    print(f"\nBackend: {ops.backend}")
    print(f"Input: {N} tokens, vocab={vocab_size}, hidden={hidden_size}")
    
    # 预热
    print("\nWarmup...")
    for _ in range(10):
        ops.fused_entropy_decision(logits, hidden, prev_hidden, 3.0, 1.0, 0.5)
    
    torch.cuda.synchronize()
    
    # 测速
    import time
    iterations = 100
    
    start = time.perf_counter()
    for _ in range(iterations):
        entropy, conf, stab, mask = ops.fused_entropy_decision(
            logits, hidden, prev_hidden, 3.0, 1.0, 0.5
        )
    torch.cuda.synchronize()
    
    elapsed = (time.perf_counter() - start) * 1000 / iterations
    
    print(f"\n[Fused Entropy Decision]")
    print(f"  Latency: {elapsed:.3f} ms")
    print(f"  Throughput: {N / elapsed * 1000:.0f} tokens/sec")
    print(f"  Exit ratio: {mask.float().mean().item()*100:.1f}%")
    
    # Token Router
    exit_mask = torch.rand(N, device=device) > 0.6
    
    for _ in range(10):
        ops.token_router_split(hidden, exit_mask)
    
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        active_h, active_i, exit_h, exit_i = ops.token_router_split(hidden, exit_mask)
    torch.cuda.synchronize()
    
    elapsed = (time.perf_counter() - start) * 1000 / iterations
    
    print(f"\n[Token Router Split]")
    print(f"  Latency: {elapsed:.3f} ms")
    print(f"  Active: {active_h.shape[0]}, Exit: {exit_h.shape[0]}")
    
    print("\n" + "=" * 70)


# 创建全局实例
_global_ops = None

def get_optimized_ops() -> OptimizedSEDACOps:
    """获取全局优化算子实例"""
    global _global_ops
    if _global_ops is None:
        _global_ops = OptimizedSEDACOps()
    return _global_ops


if __name__ == "__main__":
    benchmark_optimized_ops()
