"""
SEDAC V9.0 Production Benchmark

生产级性能基准测试
符合 MLPerf Inference 标准
"""
from __future__ import annotations
import torch
import time
import json
import statistics
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging
import gc

from .config import ProductionConfig, ModelConfig
from .inference import SEDACInferencePipeline, GenerationConfig
from .metrics import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16])
    input_lengths: List[int] = field(default_factory=lambda: [128, 256, 512])
    output_lengths: List[int] = field(default_factory=lambda: [64, 128, 256])
    warmup_iterations: int = 5
    benchmark_iterations: int = 20
    
    include_baseline: bool = True
    include_sedac: bool = True
    
    output_file: Optional[str] = None


@dataclass
class BenchmarkResult:
    """单次基准测试结果"""
    name: str
    batch_size: int
    input_length: int
    output_length: int
    
    latency_ms: float
    latency_std_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    
    throughput_tokens_per_second: float
    throughput_requests_per_second: float
    
    avg_exit_layer: float = 0.0
    skip_ratio: float = 0.0
    gpu_memory_gb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkReport:
    """完整基准测试报告"""
    model_name: str
    device: str
    precision: str
    timestamp: str
    
    results: List[BenchmarkResult] = field(default_factory=list)
    
    speedup_summary: Dict[str, float] = field(default_factory=dict)
    
    def add_result(self, result: BenchmarkResult) -> None:
        self.results.append(result)
    
    def compute_speedup(self) -> None:
        """计算 SEDAC 相对基线的加速比"""
        baseline_results = {
            (r.batch_size, r.input_length, r.output_length): r
            for r in self.results if r.name == "baseline"
        }
        
        for result in self.results:
            if result.name == "sedac":
                key = (result.batch_size, result.input_length, result.output_length)
                if key in baseline_results:
                    baseline = baseline_results[key]
                    speedup = baseline.latency_ms / result.latency_ms
                    tps_improvement = result.throughput_tokens_per_second / baseline.throughput_tokens_per_second
                    
                    self.speedup_summary[f"batch{key[0]}_in{key[1]}_out{key[2]}"] = {
                        "latency_speedup": speedup,
                        "tps_improvement": tps_improvement,
                        "skip_ratio": result.skip_ratio,
                    }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "precision": self.precision,
            "timestamp": self.timestamp,
            "results": [r.to_dict() for r in self.results],
            "speedup_summary": self.speedup_summary,
        }
    
    def save(self, path: str) -> None:
        """保存报告"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Benchmark report saved to {path}")
    
    def print_summary(self) -> None:
        """打印摘要"""
        print("\n" + "=" * 80)
        print("SEDAC V9.0 Benchmark Report")
        print("=" * 80)
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Precision: {self.precision}")
        print()
        
        print(f"{'Config':<30} {'Latency (ms)':<15} {'TPS':<15} {'Skip%':<10}")
        print("-" * 70)
        
        for r in self.results:
            config = f"{r.name} B{r.batch_size}/I{r.input_length}/O{r.output_length}"
            skip_pct = f"{r.skip_ratio*100:.1f}%" if r.skip_ratio > 0 else "-"
            print(f"{config:<30} {r.latency_ms:<15.2f} {r.throughput_tokens_per_second:<15.1f} {skip_pct:<10}")
        
        if self.speedup_summary:
            print("\n" + "-" * 70)
            print("Speedup Summary (SEDAC vs Baseline):")
            for key, metrics in self.speedup_summary.items():
                print(f"  {key}: {metrics['latency_speedup']:.2f}x latency, "
                      f"{metrics['tps_improvement']:.2f}x TPS, "
                      f"{metrics['skip_ratio']*100:.1f}% skip")
        
        print("=" * 80)


class ProductionBenchmark:
    """
    生产级基准测试器
    
    特性:
    - 标准化测试流程
    - 多维度性能评估
    - 自动生成报告
    - 支持基线对比
    """
    
    def __init__(
        self,
        model_name: str,
        config: Optional[ProductionConfig] = None,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.prod_config = config or ProductionConfig()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.pipeline: Optional[SEDACInferencePipeline] = None
        self.baseline_model = None
        self.tokenizer = None
    
    def setup(self) -> None:
        """初始化"""
        logger.info(f"Setting up benchmark for {self.model_name}")
        
        self.pipeline = SEDACInferencePipeline(
            self.model_name,
            config=self.prod_config,
            device=str(self.device),
        )
        self.pipeline.load()
        
        self.tokenizer = self.pipeline.tokenizer
        self.baseline_model = self.pipeline.model
        
        logger.info("Benchmark setup complete")
    
    def _generate_test_prompts(self, input_length: int, batch_size: int) -> List[str]:
        """生成测试 prompts"""
        base_prompt = "Please explain the following concept in detail: "
        
        filler = "artificial intelligence and machine learning " * 50
        
        prompts = []
        for i in range(batch_size):
            prompt = base_prompt + filler
            tokens = self.tokenizer.encode(prompt)[:input_length]
            prompt = self.tokenizer.decode(tokens)
            prompts.append(prompt)
        
        return prompts
    
    def _clear_cache(self) -> None:
        """清理缓存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _get_gpu_memory(self) -> float:
        """获取 GPU 内存使用"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0
    
    @torch.no_grad()
    def benchmark_baseline(
        self,
        batch_size: int,
        input_length: int,
        output_length: int,
        warmup_iters: int,
        benchmark_iters: int,
    ) -> BenchmarkResult:
        """基线基准测试 (无 SEDAC)"""
        prompts = self._generate_test_prompts(input_length, batch_size)
        
        self._clear_cache()
        
        for _ in range(warmup_iters):
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                self.baseline_model.generate(
                    **inputs,
                    max_new_tokens=output_length,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        
        torch.cuda.synchronize()
        
        latencies = []
        total_tokens = 0
        
        for _ in range(benchmark_iters):
            start = time.perf_counter()
            
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                outputs = self.baseline_model.generate(
                    **inputs,
                    max_new_tokens=output_length,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                total_tokens += outputs.shape[1] - inputs.input_ids.shape[1]
            
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
        
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        total_time_s = sum(latencies) / 1000
        
        return BenchmarkResult(
            name="baseline",
            batch_size=batch_size,
            input_length=input_length,
            output_length=output_length,
            latency_ms=statistics.mean(latencies),
            latency_std_ms=statistics.stdev(latencies) if n > 1 else 0.0,
            latency_p50_ms=sorted_latencies[int(n * 0.5)],
            latency_p95_ms=sorted_latencies[int(n * 0.95)],
            latency_p99_ms=sorted_latencies[min(int(n * 0.99), n - 1)],
            throughput_tokens_per_second=total_tokens / total_time_s,
            throughput_requests_per_second=benchmark_iters * batch_size / total_time_s,
            gpu_memory_gb=self._get_gpu_memory(),
        )
    
    @torch.no_grad()
    def benchmark_sedac(
        self,
        batch_size: int,
        input_length: int,
        output_length: int,
        warmup_iters: int,
        benchmark_iters: int,
    ) -> BenchmarkResult:
        """SEDAC 基准测试"""
        prompts = self._generate_test_prompts(input_length, batch_size)
        
        gen_config = GenerationConfig(
            max_new_tokens=output_length,
            do_sample=False,
        )
        
        self._clear_cache()
        self.pipeline.reset_metrics()
        
        for _ in range(warmup_iters):
            for prompt in prompts:
                self.pipeline(prompt, gen_config)
        
        torch.cuda.synchronize()
        self.pipeline.reset_metrics()
        
        latencies = []
        total_tokens = 0
        exit_layers = []
        
        for _ in range(benchmark_iters):
            start = time.perf_counter()
            
            for prompt in prompts:
                result = self.pipeline(prompt, gen_config)
                total_tokens += result.generated_tokens
                exit_layers.append(result.avg_exit_layer)
            
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
        
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        total_time_s = sum(latencies) / 1000
        avg_exit = statistics.mean(exit_layers) if exit_layers else self.prod_config.model.num_hidden_layers
        total_layers = self.prod_config.model.num_hidden_layers
        
        return BenchmarkResult(
            name="sedac",
            batch_size=batch_size,
            input_length=input_length,
            output_length=output_length,
            latency_ms=statistics.mean(latencies),
            latency_std_ms=statistics.stdev(latencies) if n > 1 else 0.0,
            latency_p50_ms=sorted_latencies[int(n * 0.5)],
            latency_p95_ms=sorted_latencies[int(n * 0.95)],
            latency_p99_ms=sorted_latencies[min(int(n * 0.99), n - 1)],
            throughput_tokens_per_second=total_tokens / total_time_s,
            throughput_requests_per_second=benchmark_iters * batch_size / total_time_s,
            avg_exit_layer=avg_exit,
            skip_ratio=1.0 - avg_exit / total_layers,
            gpu_memory_gb=self._get_gpu_memory(),
        )
    
    def run(self, config: BenchmarkConfig) -> BenchmarkReport:
        """
        运行完整基准测试
        
        Args:
            config: 基准测试配置
        
        Returns:
            基准测试报告
        """
        if self.pipeline is None:
            self.setup()
        
        from datetime import datetime
        
        report = BenchmarkReport(
            model_name=self.model_name,
            device=str(self.device),
            precision=self.prod_config.precision.value,
            timestamp=datetime.now().isoformat(),
        )
        
        total_configs = (
            len(config.batch_sizes) * 
            len(config.input_lengths) * 
            len(config.output_lengths)
        )
        
        current = 0
        
        for batch_size in config.batch_sizes:
            for input_len in config.input_lengths:
                for output_len in config.output_lengths:
                    current += 1
                    logger.info(f"[{current}/{total_configs}] Running B{batch_size}/I{input_len}/O{output_len}")
                    
                    if config.include_baseline:
                        try:
                            baseline_result = self.benchmark_baseline(
                                batch_size, input_len, output_len,
                                config.warmup_iterations, config.benchmark_iterations
                            )
                            report.add_result(baseline_result)
                            logger.info(f"  Baseline: {baseline_result.latency_ms:.2f}ms, {baseline_result.throughput_tokens_per_second:.1f} TPS")
                        except Exception as e:
                            logger.error(f"  Baseline failed: {e}")
                    
                    if config.include_sedac:
                        try:
                            sedac_result = self.benchmark_sedac(
                                batch_size, input_len, output_len,
                                config.warmup_iterations, config.benchmark_iterations
                            )
                            report.add_result(sedac_result)
                            logger.info(f"  SEDAC: {sedac_result.latency_ms:.2f}ms, {sedac_result.throughput_tokens_per_second:.1f} TPS, {sedac_result.skip_ratio*100:.1f}% skip")
                        except Exception as e:
                            logger.error(f"  SEDAC failed: {e}")
        
        report.compute_speedup()
        
        if config.output_file:
            report.save(config.output_file)
        
        return report


def run_production_benchmark(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    output_file: Optional[str] = None,
) -> BenchmarkReport:
    """
    运行生产级基准测试
    
    Args:
        model_name: 模型名称
        output_file: 输出文件路径
    
    Returns:
        基准测试报告
    """
    config = BenchmarkConfig(
        batch_sizes=[1, 4],
        input_lengths=[128, 256],
        output_lengths=[64, 128],
        warmup_iterations=3,
        benchmark_iterations=10,
        output_file=output_file,
    )
    
    benchmark = ProductionBenchmark(model_name)
    report = benchmark.run(config)
    report.print_summary()
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SEDAC V9.0 Benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", type=str, default="benchmark_report.json")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    run_production_benchmark(args.model, args.output)
