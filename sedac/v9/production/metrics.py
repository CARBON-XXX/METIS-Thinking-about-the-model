"""
SEDAC V9.0 Production Metrics

生产级性能监控与指标收集
符合 Prometheus/Grafana 标准
"""
from __future__ import annotations
import time
import threading
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


@dataclass
class LatencyStats:
    """延迟统计"""
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    
    _samples: deque = field(default_factory=lambda: deque(maxlen=10000))
    
    def record(self, latency_ms: float) -> None:
        self.count += 1
        self.total_ms += latency_ms
        self.min_ms = min(self.min_ms, latency_ms)
        self.max_ms = max(self.max_ms, latency_ms)
        self._samples.append(latency_ms)
        
        if len(self._samples) >= 100:
            sorted_samples = sorted(self._samples)
            n = len(sorted_samples)
            self.p50_ms = sorted_samples[int(n * 0.5)]
            self.p95_ms = sorted_samples[int(n * 0.95)]
            self.p99_ms = sorted_samples[min(int(n * 0.99), n - 1)]
    
    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "avg_ms": self.avg_ms,
            "min_ms": self.min_ms if self.min_ms != float("inf") else 0.0,
            "max_ms": self.max_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
        }


@dataclass
class ThroughputStats:
    """吞吐量统计"""
    total_tokens: int = 0
    total_requests: int = 0
    window_tokens: int = 0
    window_requests: int = 0
    window_start_time: float = field(default_factory=time.time)
    window_duration: float = 60.0
    
    _tokens_per_second: float = 0.0
    _requests_per_second: float = 0.0
    
    def record(self, tokens: int, requests: int = 1) -> None:
        self.total_tokens += tokens
        self.total_requests += requests
        self.window_tokens += tokens
        self.window_requests += requests
        
        elapsed = time.time() - self.window_start_time
        if elapsed >= self.window_duration:
            self._tokens_per_second = self.window_tokens / elapsed
            self._requests_per_second = self.window_requests / elapsed
            self.window_tokens = 0
            self.window_requests = 0
            self.window_start_time = time.time()
    
    @property
    def tokens_per_second(self) -> float:
        return self._tokens_per_second
    
    @property
    def requests_per_second(self) -> float:
        return self._requests_per_second
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "total_requests": self.total_requests,
            "tokens_per_second": self.tokens_per_second,
            "requests_per_second": self.requests_per_second,
        }


@dataclass
class SEDACStats:
    """SEDAC 特定统计"""
    total_layers_computed: int = 0
    total_layers_possible: int = 0
    early_exits: int = 0
    full_forwards: int = 0
    ghost_kv_hits: int = 0
    o1_activations: int = 0
    
    _exit_layer_histogram: Dict[int, int] = field(default_factory=dict)
    _entropy_samples: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def record_forward(
        self, 
        exit_layer: int, 
        total_layers: int,
        entropy: float,
        used_ghost_kv: bool = False,
        used_o1: bool = False
    ) -> None:
        self.total_layers_computed += exit_layer + 1
        self.total_layers_possible += total_layers
        
        if exit_layer < total_layers - 1:
            self.early_exits += 1
        else:
            self.full_forwards += 1
        
        if used_ghost_kv:
            self.ghost_kv_hits += 1
        
        if used_o1:
            self.o1_activations += 1
        
        self._exit_layer_histogram[exit_layer] = self._exit_layer_histogram.get(exit_layer, 0) + 1
        self._entropy_samples.append(entropy)
    
    @property
    def skip_ratio(self) -> float:
        if self.total_layers_possible == 0:
            return 0.0
        return 1.0 - self.total_layers_computed / self.total_layers_possible
    
    @property
    def early_exit_ratio(self) -> float:
        total = self.early_exits + self.full_forwards
        return self.early_exits / total if total > 0 else 0.0
    
    @property
    def avg_entropy(self) -> float:
        if not self._entropy_samples:
            return 0.0
        return sum(self._entropy_samples) / len(self._entropy_samples)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "skip_ratio": self.skip_ratio,
            "early_exit_ratio": self.early_exit_ratio,
            "avg_entropy": self.avg_entropy,
            "early_exits": self.early_exits,
            "full_forwards": self.full_forwards,
            "ghost_kv_hits": self.ghost_kv_hits,
            "o1_activations": self.o1_activations,
            "exit_layer_histogram": dict(self._exit_layer_histogram),
        }


class MetricsCollector:
    """指标收集器 - 线程安全"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self.latency = LatencyStats()
        self.throughput = ThroughputStats()
        self.sedac = SEDACStats()
        self._custom_metrics: Dict[str, float] = {}
        self._start_time = time.time()
    
    def record_latency(self, latency_ms: float) -> None:
        with self._lock:
            self.latency.record(latency_ms)
    
    def record_throughput(self, tokens: int, requests: int = 1) -> None:
        with self._lock:
            self.throughput.record(tokens, requests)
    
    def record_sedac(
        self,
        exit_layer: int,
        total_layers: int,
        entropy: float,
        used_ghost_kv: bool = False,
        used_o1: bool = False
    ) -> None:
        with self._lock:
            self.sedac.record_forward(exit_layer, total_layers, entropy, used_ghost_kv, used_o1)
    
    def set_metric(self, name: str, value: float) -> None:
        with self._lock:
            self._custom_metrics[name] = value
    
    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "uptime_seconds": time.time() - self._start_time,
                "latency": self.latency.to_dict(),
                "throughput": self.throughput.to_dict(),
                "sedac": self.sedac.to_dict(),
                "custom": dict(self._custom_metrics),
            }
    
    def reset(self) -> None:
        with self._lock:
            self.latency = LatencyStats()
            self.throughput = ThroughputStats()
            self.sedac = SEDACStats()
            self._custom_metrics.clear()
            self._start_time = time.time()
    
    def export_prometheus(self) -> str:
        """导出 Prometheus 格式"""
        metrics = self.get_metrics()
        lines = []
        
        lines.append(f'sedac_uptime_seconds {metrics["uptime_seconds"]}')
        
        lat = metrics["latency"]
        lines.append(f'sedac_latency_avg_ms {lat["avg_ms"]}')
        lines.append(f'sedac_latency_p50_ms {lat["p50_ms"]}')
        lines.append(f'sedac_latency_p95_ms {lat["p95_ms"]}')
        lines.append(f'sedac_latency_p99_ms {lat["p99_ms"]}')
        
        tp = metrics["throughput"]
        lines.append(f'sedac_tokens_per_second {tp["tokens_per_second"]}')
        lines.append(f'sedac_requests_per_second {tp["requests_per_second"]}')
        lines.append(f'sedac_total_tokens {tp["total_tokens"]}')
        
        sed = metrics["sedac"]
        lines.append(f'sedac_skip_ratio {sed["skip_ratio"]}')
        lines.append(f'sedac_early_exit_ratio {sed["early_exit_ratio"]}')
        lines.append(f'sedac_avg_entropy {sed["avg_entropy"]}')
        lines.append(f'sedac_ghost_kv_hits {sed["ghost_kv_hits"]}')
        lines.append(f'sedac_o1_activations {sed["o1_activations"]}')
        
        return "\n".join(lines)


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, collector: Optional[MetricsCollector] = None):
        self.collector = collector or MetricsCollector()
        self._cuda_available = False
        
        try:
            import torch
            self._cuda_available = torch.cuda.is_available()
        except:
            pass
    
    @contextmanager
    def measure_latency(self):
        """测量延迟上下文管理器"""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.collector.record_latency(elapsed_ms)
    
    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """获取 GPU 内存使用"""
        if not self._cuda_available:
            return {}
        
        import torch
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "utilization": allocated / total if total > 0 else 0.0,
        }
    
    def log_summary(self) -> None:
        """记录摘要日志"""
        metrics = self.collector.get_metrics()
        
        logger.info("=" * 60)
        logger.info("SEDAC Performance Summary")
        logger.info("=" * 60)
        logger.info(f"Uptime: {metrics['uptime_seconds']:.1f}s")
        logger.info(f"Latency: avg={metrics['latency']['avg_ms']:.2f}ms, p95={metrics['latency']['p95_ms']:.2f}ms")
        logger.info(f"Throughput: {metrics['throughput']['tokens_per_second']:.1f} tokens/s")
        logger.info(f"SEDAC Skip Ratio: {metrics['sedac']['skip_ratio']*100:.1f}%")
        logger.info(f"Early Exit Ratio: {metrics['sedac']['early_exit_ratio']*100:.1f}%")
        
        if self._cuda_available:
            gpu = self.get_gpu_memory_usage()
            logger.info(f"GPU Memory: {gpu['allocated_gb']:.2f}/{gpu['total_gb']:.2f} GB ({gpu['utilization']*100:.1f}%)")
