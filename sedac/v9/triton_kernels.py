"""
SEDAC V9.0 - Triton 高性能算子

解决"Python Overhead陷阱"：将决策逻辑编译为GPU Kernel

目标：将决策开销从毫秒级压缩到微秒级

包含：
1. 熵计算Kernel
2. 置信度阈值Kernel
3. 融合的决策Kernel
"""

from __future__ import annotations
import torch
import logging
from typing import Tuple, Optional
import math

logger = logging.getLogger(__name__)

# 尝试导入Triton
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    logger.warning("Triton not available. Install with: pip install triton")


if TRITON_AVAILABLE:
    
    @triton.jit
    def entropy_kernel(
        logits_ptr,      # [batch, vocab_size]
        entropy_ptr,     # [batch]
        batch_size,
        vocab_size,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        高效熵计算Kernel
        
        H = -sum(p * log(p))
        """
        pid = tl.program_id(0)
        
        if pid >= batch_size:
            return
        
        # 该batch的logits起始位置
        logits_offset = pid * vocab_size
        
        # 1. 计算max用于数值稳定
        max_val = -float('inf')
        for i in range(0, vocab_size, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < vocab_size
            logits = tl.load(logits_ptr + logits_offset + offsets, mask=mask, other=-float('inf'))
            max_val = tl.maximum(max_val, tl.max(logits, axis=0))
        
        # 2. 计算sum(exp(x - max))
        sum_exp = 0.0
        for i in range(0, vocab_size, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < vocab_size
            logits = tl.load(logits_ptr + logits_offset + offsets, mask=mask, other=-float('inf'))
            sum_exp += tl.sum(tl.exp(logits - max_val), axis=0)
        
        log_sum_exp = tl.log(sum_exp)
        
        # 3. 计算熵 H = log(sum_exp) - sum(p * (x - max)) / sum_exp
        #           = log(sum_exp) - (sum(x * exp(x-max)) / sum_exp - max)
        weighted_sum = 0.0
        for i in range(0, vocab_size, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < vocab_size
            logits = tl.load(logits_ptr + logits_offset + offsets, mask=mask, other=0.0)
            exp_logits = tl.exp(logits - max_val)
            weighted_sum += tl.sum(logits * exp_logits, axis=0)
        
        mean_logit = weighted_sum / sum_exp
        entropy = log_sum_exp + max_val - mean_logit
        
        # 转换为nats → bits并归一化
        entropy = entropy / tl.log(2.0)  # nats to bits
        
        tl.store(entropy_ptr + pid, entropy)
    
    
    @triton.jit
    def confidence_threshold_kernel(
        entropy_ptr,         # [batch]
        confidence_ptr,      # [batch] 输出
        decision_ptr,        # [batch] 输出：1=exit, 0=continue
        mean_entropy,        # 标量：历史熵均值
        std_entropy,         # 标量：历史熵标准差
        layer_progress,      # 标量：当前层进度 0-1
        batch_size,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        置信度计算和阈值决策Kernel
        
        confidence = sigmoid((mean - entropy) / std * scale)
        decision = confidence > threshold(layer_progress)
        """
        pid = tl.program_id(0)
        offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < batch_size
        
        # 加载熵
        entropy = tl.load(entropy_ptr + offset, mask=mask, other=0.0)
        
        # 计算z-score
        z_score = (mean_entropy - entropy) / (std_entropy + 1e-6)
        
        # Sigmoid转换为置信度
        confidence = 1.0 / (1.0 + tl.exp(-z_score * 2.0))
        
        # 动态阈值：层越深，阈值越低
        base_threshold = 0.7
        threshold = base_threshold - layer_progress * 0.3
        
        # 决策
        decision = tl.where(confidence > threshold, 1.0, 0.0)
        
        # 存储
        tl.store(confidence_ptr + offset, confidence, mask=mask)
        tl.store(decision_ptr + offset, decision, mask=mask)
    
    
    @triton.jit
    def fused_sedac_decision_kernel(
        hidden_ptr,          # [batch, hidden_size]
        prev_hidden_ptr,     # [batch, hidden_size] 上一层hidden
        entropy_ptr,         # [batch] 当前熵
        confidence_ptr,      # [batch] 输出
        cognitive_load_ptr,  # [batch] 输出
        decision_ptr,        # [batch] 输出
        mean_entropy,
        std_entropy,
        layer_progress,
        batch_size,
        hidden_size,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        融合的SEDAC决策Kernel
        
        一次性计算：
        1. Hidden state变化（稳定性指标）
        2. 置信度
        3. 认知负荷
        4. 退出决策
        """
        pid = tl.program_id(0)
        
        if pid >= batch_size:
            return
        
        hidden_offset = pid * hidden_size
        
        # 1. 计算hidden state变化（L2范数差异）
        diff_sq_sum = 0.0
        norm_sq_sum = 0.0
        
        for i in range(0, hidden_size, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < hidden_size
            
            curr = tl.load(hidden_ptr + hidden_offset + offsets, mask=mask, other=0.0)
            prev = tl.load(prev_hidden_ptr + hidden_offset + offsets, mask=mask, other=0.0)
            
            diff = curr - prev
            diff_sq_sum += tl.sum(diff * diff, axis=0)
            norm_sq_sum += tl.sum(curr * curr, axis=0)
        
        stability = 1.0 / (1.0 + tl.sqrt(diff_sq_sum) / (tl.sqrt(norm_sq_sum) + 1e-6))
        
        # 2. 加载熵并计算置信度
        entropy = tl.load(entropy_ptr + pid)
        z_score = (mean_entropy - entropy) / (std_entropy + 1e-6)
        confidence = 1.0 / (1.0 + tl.exp(-z_score * 2.0))
        
        # 3. 计算认知负荷
        # cognitive_load = (1 - confidence) * (1 - stability) * (1 - layer_progress)
        cognitive_load = (1.0 - confidence) * 0.5 + (1.0 - stability) * 0.3 + (1.0 - layer_progress) * 0.2
        
        # 4. 决策
        # 高置信 + 低认知负荷 + 足够层数 → 退出
        exit_score = confidence * (1.0 - cognitive_load) * layer_progress
        decision = tl.where(exit_score > 0.5, 1.0, 0.0)
        
        # 存储
        tl.store(confidence_ptr + pid, confidence)
        tl.store(cognitive_load_ptr + pid, cognitive_load)
        tl.store(decision_ptr + pid, decision)


class TritonSEDACOps:
    """
    Triton加速的SEDAC操作
    """
    
    def __init__(self, vocab_size: int = 32000, hidden_size: int = 4096):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # 在线统计量
        self.entropy_mean = 3.0
        self.entropy_std = 1.0
        self.n_samples = 0
    
    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        计算batch熵
        
        Args:
            logits: [batch, vocab_size]
            
        Returns:
            entropy: [batch]
        """
        if not TRITON_AVAILABLE:
            # Fallback到PyTorch
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            return entropy / math.log(2)  # nats to bits
        
        batch_size = logits.shape[0]
        entropy = torch.empty(batch_size, device=logits.device, dtype=logits.dtype)
        
        BLOCK_SIZE = 1024
        grid = (batch_size,)
        
        entropy_kernel[grid](
            logits, entropy,
            batch_size, self.vocab_size,
            BLOCK_SIZE,
        )
        
        return entropy
    
    def compute_decision(
        self,
        entropy: torch.Tensor,
        layer_progress: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算置信度和决策
        
        Args:
            entropy: [batch]
            layer_progress: 0-1
            
        Returns:
            confidence: [batch]
            decision: [batch] (1=exit, 0=continue)
        """
        if not TRITON_AVAILABLE:
            # Fallback
            z_score = (self.entropy_mean - entropy) / (self.entropy_std + 1e-6)
            confidence = torch.sigmoid(z_score * 2.0)
            threshold = 0.7 - layer_progress * 0.3
            decision = (confidence > threshold).float()
            return confidence, decision
        
        batch_size = entropy.shape[0]
        confidence = torch.empty_like(entropy)
        decision = torch.empty_like(entropy)
        
        BLOCK_SIZE = 256
        grid = ((batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        
        confidence_threshold_kernel[grid](
            entropy, confidence, decision,
            self.entropy_mean, self.entropy_std,
            layer_progress,
            batch_size,
            BLOCK_SIZE,
        )
        
        return confidence, decision
    
    def fused_decision(
        self,
        hidden: torch.Tensor,
        prev_hidden: torch.Tensor,
        entropy: torch.Tensor,
        layer_progress: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        融合决策（一次Kernel调用）
        
        Returns:
            confidence, cognitive_load, decision
        """
        if not TRITON_AVAILABLE:
            # Fallback
            # 稳定性
            diff = hidden - prev_hidden
            stability = 1.0 / (1.0 + torch.norm(diff, dim=-1) / (torch.norm(hidden, dim=-1) + 1e-6))
            
            # 置信度
            z_score = (self.entropy_mean - entropy) / (self.entropy_std + 1e-6)
            confidence = torch.sigmoid(z_score * 2.0)
            
            # 认知负荷
            cognitive_load = (1 - confidence) * 0.5 + (1 - stability) * 0.3 + (1 - layer_progress) * 0.2
            
            # 决策
            exit_score = confidence * (1 - cognitive_load) * layer_progress
            decision = (exit_score > 0.5).float()
            
            return confidence, cognitive_load, decision
        
        batch_size = hidden.shape[0]
        confidence = torch.empty(batch_size, device=hidden.device, dtype=hidden.dtype)
        cognitive_load = torch.empty_like(confidence)
        decision = torch.empty_like(confidence)
        
        BLOCK_SIZE = 256
        grid = (batch_size,)
        
        fused_sedac_decision_kernel[grid](
            hidden, prev_hidden, entropy,
            confidence, cognitive_load, decision,
            self.entropy_mean, self.entropy_std,
            layer_progress,
            batch_size, self.hidden_size,
            BLOCK_SIZE,
        )
        
        return confidence, cognitive_load, decision
    
    def update_statistics(self, entropy: torch.Tensor):
        """更新在线统计量"""
        batch_mean = entropy.mean().item()
        batch_std = entropy.std().item()
        
        # Welford在线更新
        self.n_samples += entropy.shape[0]
        delta = batch_mean - self.entropy_mean
        self.entropy_mean += delta / self.n_samples
        self.entropy_std = 0.9 * self.entropy_std + 0.1 * batch_std


# PyTorch Fallback实现（无Triton时使用）
class PyTorchSEDACOps:
    """
    纯PyTorch实现的SEDAC操作（Fallback）
    """
    
    def __init__(self):
        self.entropy_mean = 3.0
        self.entropy_std = 1.0
        self.n_samples = 0
    
    @torch.no_grad()
    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """计算熵"""
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy / math.log(2)
    
    @torch.no_grad()
    def compute_decision(
        self,
        entropy: torch.Tensor,
        layer_progress: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算决策"""
        z_score = (self.entropy_mean - entropy) / (self.entropy_std + 1e-6)
        confidence = torch.sigmoid(z_score * 2.0)
        threshold = 0.7 - layer_progress * 0.3
        decision = (confidence > threshold).float()
        return confidence, decision
    
    def update_statistics(self, entropy: torch.Tensor):
        """更新统计量"""
        batch_mean = entropy.mean().item()
        batch_std = entropy.std().item()
        self.n_samples += entropy.shape[0]
        delta = batch_mean - self.entropy_mean
        self.entropy_mean += delta / self.n_samples
        self.entropy_std = 0.9 * self.entropy_std + 0.1 * batch_std


def create_sedac_ops(use_triton: bool = True) -> TritonSEDACOps:
    """创建SEDAC操作实例"""
    if use_triton and TRITON_AVAILABLE:
        logger.info("Using Triton-accelerated SEDAC ops")
        return TritonSEDACOps()
    else:
        logger.info("Using PyTorch fallback SEDAC ops")
        return PyTorchSEDACOps()


def benchmark_sedac_ops(batch_size: int = 32, vocab_size: int = 32000, num_runs: int = 100):
    """
    性能基准测试
    """
    import time
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 生成测试数据
    logits = torch.randn(batch_size, vocab_size, device=device)
    hidden = torch.randn(batch_size, 4096, device=device)
    prev_hidden = torch.randn(batch_size, 4096, device=device)
    
    # PyTorch版本
    pytorch_ops = PyTorchSEDACOps()
    
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()
    for _ in range(num_runs):
        entropy = pytorch_ops.compute_entropy(logits)
        confidence, decision = pytorch_ops.compute_decision(entropy, 0.5)
    torch.cuda.synchronize() if device.type == "cuda" else None
    pytorch_time = (time.perf_counter() - start) / num_runs * 1000
    
    print(f"PyTorch: {pytorch_time:.3f} ms/iter")
    
    # Triton版本
    if TRITON_AVAILABLE:
        triton_ops = TritonSEDACOps(vocab_size=vocab_size)
        
        # Warmup
        for _ in range(10):
            entropy = triton_ops.compute_entropy(logits)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_runs):
            entropy = triton_ops.compute_entropy(logits)
            confidence, decision = triton_ops.compute_decision(entropy, 0.5)
        torch.cuda.synchronize()
        triton_time = (time.perf_counter() - start) / num_runs * 1000
        
        print(f"Triton: {triton_time:.3f} ms/iter")
        print(f"Speedup: {pytorch_time / triton_time:.2f}x")
    else:
        print("Triton not available for benchmark")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    benchmark_sedac_ops()
