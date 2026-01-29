"""
SEDAC V9.0 - 端到端评测与性能基准

评测内容:
1. 加速比 (Speedup)
2. 精度保持 (Accuracy Retention)
3. 显存占用 (Memory Usage)
4. 吞吐量 (Throughput)
5. 各策略对比 (Strategy Comparison)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List
import logging
import time
import json
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# 导入SEDAC组件
from sedac.v9.industrial_integrator import create_industrial_integrator, IntegrationStrategy
from sedac.v9.production_layer import ProductionSEDACLayer, LayerConfig, SkipMode
from sedac.v9.model_integration import inject_sedac, SEDACModel
from sedac.v9.kernels.cuda_ops import (
    fused_entropy_decision, token_router_split, token_router_merge,
    CUDA_AVAILABLE, TRITON_AVAILABLE,
)


@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    # 模型参数
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    vocab_size: int = 32000
    
    # 测试参数
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16])
    seq_lengths: List[int] = field(default_factory=lambda: [128, 512, 1024, 2048])
    warmup_iterations: int = 10
    benchmark_iterations: int = 100
    
    # 策略
    strategies: List[str] = field(default_factory=lambda: ["baseline", "safe", "fast", "ultimate"])
    
    # 输出
    output_dir: str = "./benchmark_results"


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    strategy: str
    batch_size: int
    seq_length: int
    
    # 延迟
    latency_ms: float
    latency_std: float
    
    # 吞吐量
    throughput_tokens_per_sec: float
    
    # 显存
    memory_mb: float
    peak_memory_mb: float
    
    # SEDAC指标
    skip_ratio: float = 0.0
    computed_layers: int = 0
    
    # 精度（如果有参考）
    accuracy_retention: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
            "latency_ms": round(self.latency_ms, 3),
            "latency_std": round(self.latency_std, 3),
            "throughput_tokens_per_sec": round(self.throughput_tokens_per_sec, 1),
            "memory_mb": round(self.memory_mb, 1),
            "peak_memory_mb": round(self.peak_memory_mb, 1),
            "skip_ratio": round(self.skip_ratio, 3),
            "computed_layers": self.computed_layers,
            "accuracy_retention": round(self.accuracy_retention, 4),
        }


class MockTransformerModel(nn.Module):
    """
    模拟Transformer模型（用于基准测试）
    """
    
    def __init__(
        self,
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        intermediate_size: int = 11008,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 构建层
        self.layers = nn.ModuleList([
            self._make_layer(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_size)
        
        # 配置（用于SEDAC注入）
        self.config = type('Config', (), {
            'num_hidden_layers': num_layers,
            'hidden_size': hidden_size,
            'num_attention_heads': num_heads,
        })()
    
    def _make_layer(self, hidden_size, num_heads, intermediate_size):
        """创建单层"""
        layer = nn.Module()
        layer.input_layernorm = nn.LayerNorm(hidden_size)
        layer.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        layer.post_attention_layernorm = nn.LayerNorm(hidden_size)
        layer.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.SiLU(),
            nn.Linear(intermediate_size, hidden_size),
        )
        
        # 添加k_proj和v_proj
        layer.self_attn.k_proj = nn.Linear(hidden_size, hidden_size)
        layer.self_attn.v_proj = nn.Linear(hidden_size, hidden_size)
        
        def forward(hidden_states, **kwargs):
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            attn_output, _ = layer.self_attn(hidden_states, hidden_states, hidden_states)
            hidden_states = residual + attn_output
            
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
            
            return (hidden_states,)
        
        layer.forward = forward
        return layer
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)[0]
        return self.norm(hidden_states)


class SEDACBenchmark:
    """
    SEDAC基准测试器
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
        self.results: List[BenchmarkResult] = []
        
        os.makedirs(config.output_dir, exist_ok=True)
    
    @contextmanager
    def _measure_memory(self):
        """测量显存"""
        if CUDA_AVAILABLE:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        yield
        
        if CUDA_AVAILABLE:
            torch.cuda.synchronize()
    
    def _get_memory_mb(self) -> Tuple[float, float]:
        """获取显存使用（当前，峰值）"""
        if CUDA_AVAILABLE:
            current = torch.cuda.memory_allocated() / 1024 / 1024
            peak = torch.cuda.max_memory_allocated() / 1024 / 1024
            return current, peak
        return 0.0, 0.0
    
    def benchmark_baseline(
        self,
        model: nn.Module,
        batch_size: int,
        seq_length: int,
    ) -> BenchmarkResult:
        """基线测试（无SEDAC）"""
        cfg = self.config
        
        # 准备输入
        hidden = torch.randn(batch_size, seq_length, cfg.hidden_size, device=self.device)
        
        # 预热
        for _ in range(cfg.warmup_iterations):
            _ = model(hidden)
        
        if CUDA_AVAILABLE:
            torch.cuda.synchronize()
        
        # 测量
        latencies = []
        
        with self._measure_memory():
            for _ in range(cfg.benchmark_iterations):
                start = time.perf_counter()
                _ = model(hidden)
                if CUDA_AVAILABLE:
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000)
        
        current_mem, peak_mem = self._get_memory_mb()
        
        avg_latency = sum(latencies) / len(latencies)
        std_latency = (sum((l - avg_latency) ** 2 for l in latencies) / len(latencies)) ** 0.5
        throughput = (batch_size * seq_length) / (avg_latency / 1000)
        
        return BenchmarkResult(
            strategy="baseline",
            batch_size=batch_size,
            seq_length=seq_length,
            latency_ms=avg_latency,
            latency_std=std_latency,
            throughput_tokens_per_sec=throughput,
            memory_mb=current_mem,
            peak_memory_mb=peak_mem,
            computed_layers=cfg.num_layers,
        )
    
    def benchmark_sedac(
        self,
        model: SEDACModel,
        batch_size: int,
        seq_length: int,
        strategy: str,
    ) -> BenchmarkResult:
        """SEDAC测试"""
        cfg = self.config
        
        # 准备输入
        hidden = torch.randn(batch_size, seq_length, cfg.hidden_size, device=self.device)
        
        # 重置SEDAC统计
        model.reset_sedac()
        
        # 预热
        for _ in range(cfg.warmup_iterations):
            model.reset_sedac()
            _ = model(hidden)
        
        if CUDA_AVAILABLE:
            torch.cuda.synchronize()
        
        # 测量
        latencies = []
        
        with self._measure_memory():
            for _ in range(cfg.benchmark_iterations):
                model.reset_sedac()
                start = time.perf_counter()
                _ = model(hidden)
                if CUDA_AVAILABLE:
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000)
        
        current_mem, peak_mem = self._get_memory_mb()
        stats = model.get_sedac_stats()
        
        avg_latency = sum(latencies) / len(latencies)
        std_latency = (sum((l - avg_latency) ** 2 for l in latencies) / len(latencies)) ** 0.5
        throughput = (batch_size * seq_length) / (avg_latency / 1000)
        
        return BenchmarkResult(
            strategy=strategy,
            batch_size=batch_size,
            seq_length=seq_length,
            latency_ms=avg_latency,
            latency_std=std_latency,
            throughput_tokens_per_sec=throughput,
            memory_mb=current_mem,
            peak_memory_mb=peak_mem,
            skip_ratio=stats.get("skipped_ratio", 0.0),
            computed_layers=int(stats.get("computed_ratio", 1.0) * cfg.num_layers),
        )
    
    def benchmark_kernels(self) -> Dict[str, float]:
        """内核基准测试"""
        cfg = self.config
        results = {}
        
        if not CUDA_AVAILABLE:
            logger.warning("CUDA not available. Skipping kernel benchmarks.")
            return results
        
        N = 4096
        
        # Fused Entropy Decision
        logits = torch.randn(N, cfg.vocab_size, device=self.device)
        hidden = torch.randn(N, cfg.hidden_size, device=self.device)
        prev_hidden = torch.randn(N, cfg.hidden_size, device=self.device)
        
        # 预热
        for _ in range(10):
            fused_entropy_decision(logits, hidden, prev_hidden, 3.0, 1.0, 0.5)
        
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(100):
            fused_entropy_decision(logits, hidden, prev_hidden, 3.0, 1.0, 0.5)
        torch.cuda.synchronize()
        
        results["fused_entropy_decision_ms"] = (time.perf_counter() - start) * 10
        
        # Token Router
        exit_mask = torch.rand(N, device=self.device) > 0.6
        
        for _ in range(10):
            token_router_split(hidden, exit_mask)
        
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(100):
            token_router_split(hidden, exit_mask)
        torch.cuda.synchronize()
        
        results["token_router_split_ms"] = (time.perf_counter() - start) * 10
        
        return results
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """运行完整基准测试"""
        cfg = self.config
        
        print("=" * 70)
        print("SEDAC V9.0 Full Benchmark")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"CUDA Available: {CUDA_AVAILABLE}")
        print(f"Triton Available: {TRITON_AVAILABLE}")
        print("=" * 70)
        
        # 创建基础模型
        base_model = MockTransformerModel(
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
        ).to(self.device)
        
        # 创建SEDAC模型
        sedac_models = {}
        for strategy in ["safe", "fast", "ultimate"]:
            sedac_models[strategy] = inject_sedac(
                MockTransformerModel(
                    hidden_size=cfg.hidden_size,
                    num_layers=cfg.num_layers,
                    num_heads=cfg.num_heads,
                ).to(self.device),
                strategy=strategy,
            )
        
        # 内核基准
        print("\n[1/3] Kernel Benchmarks")
        kernel_results = self.benchmark_kernels()
        for name, latency in kernel_results.items():
            print(f"  {name}: {latency:.3f} ms")
        
        # 模型基准
        print("\n[2/3] Model Benchmarks")
        
        for batch_size in cfg.batch_sizes[:2]:  # 限制测试数量
            for seq_length in cfg.seq_lengths[:2]:
                print(f"\n  Batch={batch_size}, Seq={seq_length}")
                
                # 基线
                result = self.benchmark_baseline(base_model, batch_size, seq_length)
                self.results.append(result)
                print(f"    Baseline: {result.latency_ms:.2f}ms, {result.throughput_tokens_per_sec:.0f} tok/s")
                
                baseline_latency = result.latency_ms
                
                # SEDAC策略
                for strategy in ["safe", "fast"]:
                    result = self.benchmark_sedac(
                        sedac_models[strategy], batch_size, seq_length, strategy
                    )
                    self.results.append(result)
                    
                    speedup = baseline_latency / result.latency_ms
                    print(f"    {strategy.capitalize()}: {result.latency_ms:.2f}ms, "
                          f"speedup={speedup:.2f}x, skip={result.skip_ratio*100:.1f}%")
        
        # 精度测试
        print("\n[3/3] Accuracy Retention Test")
        accuracy = self._test_accuracy(base_model, sedac_models["safe"])
        print(f"  Output Similarity: {accuracy:.4f}")
        
        # 汇总
        summary = {
            "device": str(self.device),
            "cuda_available": CUDA_AVAILABLE,
            "triton_available": TRITON_AVAILABLE,
            "config": {
                "hidden_size": cfg.hidden_size,
                "num_layers": cfg.num_layers,
                "num_heads": cfg.num_heads,
            },
            "kernel_results": kernel_results,
            "model_results": [r.to_dict() for r in self.results],
            "accuracy_retention": accuracy,
        }
        
        # 保存结果
        output_path = os.path.join(cfg.output_dir, "benchmark_results.json")
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {output_path}")
        
        return summary
    
    def _test_accuracy(
        self,
        base_model: nn.Module,
        sedac_model: SEDACModel,
    ) -> float:
        """测试精度保持"""
        cfg = self.config
        
        # 随机输入
        hidden = torch.randn(2, 64, cfg.hidden_size, device=self.device)
        
        with torch.no_grad():
            base_output = base_model(hidden)
            sedac_model.reset_sedac()
            sedac_output = sedac_model(hidden)
        
        # 余弦相似度
        similarity = F.cosine_similarity(
            base_output.flatten(), sedac_output.flatten(), dim=0
        ).item()
        
        return similarity


def run_benchmark(
    hidden_size: int = 512,
    num_layers: int = 12,
    output_dir: str = "./benchmark_results",
):
    """运行基准测试"""
    config = BenchmarkConfig(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=8,
        head_dim=hidden_size // 8,
        vocab_size=32000,
        batch_sizes=[1, 4],
        seq_lengths=[64, 256],
        warmup_iterations=5,
        benchmark_iterations=50,
        output_dir=output_dir,
    )
    
    benchmark = SEDACBenchmark(config)
    return benchmark.run_full_benchmark()


if __name__ == "__main__":
    run_benchmark()
