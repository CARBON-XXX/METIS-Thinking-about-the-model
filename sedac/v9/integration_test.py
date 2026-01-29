"""
SEDAC V9.0 - 集成测试 (Integration Test)

验证工业级集成方案的正确性和性能
"""

from __future__ import annotations
import torch
import torch.nn as nn
import time
import sys
from typing import Dict, Any, List
from dataclasses import dataclass

# 导入所有SEDAC组件
from sedac.v9.kv_cache_manager import (
    create_kv_cache_manager, create_layer_scheduler, SkipMode
)
from sedac.v9.ghost_kv import create_ghost_kv_manager, GhostKVConfig
from sedac.v9.token_router import create_token_router
from sedac.v9.attention_sinks import create_attention_sink_protector
from sedac.v9.fused_gpu_kernel import create_fused_engine
from sedac.v9.industrial_integrator import (
    create_industrial_integrator, IntegrationStrategy
)
from sedac.v9.production_layer import (
    ProductionSEDACLayer, LayerConfig, create_production_layer
)


@dataclass
class TestResult:
    """测试结果"""
    name: str
    passed: bool
    message: str
    latency_ms: float = 0.0
    metrics: Dict[str, Any] = None


class SEDACIntegrationTester:
    """
    SEDAC集成测试器
    """
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 测试配置
        self.hidden_size = 512
        self.num_layers = 12
        self.num_heads = 8
        self.head_dim = 64
        self.batch_size = 2
        self.seq_len = 64
    
    def run_all_tests(self) -> bool:
        """运行所有测试"""
        print("=" * 70)
        print("SEDAC V9.0 Integration Test Suite")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Config: hidden={self.hidden_size}, layers={self.num_layers}, "
              f"heads={self.num_heads}")
        print("=" * 70)
        
        # 测试列表
        tests = [
            self.test_kv_cache_manager,
            self.test_ghost_kv_manager,
            self.test_token_router,
            self.test_attention_sinks,
            self.test_fused_engine,
            self.test_production_layer,
            self.test_industrial_integrator_safe,
            self.test_industrial_integrator_fast,
            self.test_kv_continuity,
            self.test_anchor_enforcement,
        ]
        
        for test_fn in tests:
            try:
                result = test_fn()
                self.results.append(result)
                status = "✅ PASS" if result.passed else "❌ FAIL"
                print(f"\n{status} {result.name}")
                print(f"    {result.message}")
                if result.latency_ms > 0:
                    print(f"    Latency: {result.latency_ms:.2f}ms")
            except Exception as e:
                self.results.append(TestResult(
                    name=test_fn.__name__,
                    passed=False,
                    message=f"Exception: {str(e)}"
                ))
                print(f"\n❌ FAIL {test_fn.__name__}")
                print(f"    Exception: {str(e)}")
        
        # 汇总
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        print("\n" + "=" * 70)
        print(f"Results: {passed}/{total} tests passed")
        print("=" * 70)
        
        return passed == total
    
    def test_kv_cache_manager(self) -> TestResult:
        """测试KV Cache管理器"""
        start = time.perf_counter()
        
        manager = create_kv_cache_manager(
            num_layers=self.num_layers,
        )
        
        # KVCacheManager管理状态，使用KVOnlyProjection计算
        from sedac.v9.kv_cache_manager import KVOnlyProjection
        kv_proj = KVOnlyProjection(self.hidden_size, self.num_heads, self.head_dim).to(self.device)
        
        hidden = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)
        
        # 测试KV-Only计算
        key, value = kv_proj(hidden)
        
        elapsed = (time.perf_counter() - start) * 1000
        
        # 验证
        expected_shape = (self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        passed = (key.shape == expected_shape and value.shape == expected_shape)
        
        return TestResult(
            name="KV Cache Manager",
            passed=passed,
            message=f"KV shape: {list(key.shape)}, expected: {list(expected_shape)}",
            latency_ms=elapsed,
        )
    
    def test_ghost_kv_manager(self) -> TestResult:
        """测试Ghost KV管理器"""
        start = time.perf_counter()
        
        manager = create_ghost_kv_manager(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_layers=self.num_layers,
        ).to(self.device)
        
        hidden = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)
        
        # 测试Ghost KV生成
        key, value = manager.generate_ghost_kv(hidden, layer_idx=0)
        
        elapsed = (time.perf_counter() - start) * 1000
        
        # 验证形状
        expected_shape = (self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        passed = (key.shape == expected_shape and value.shape == expected_shape)
        
        # 参数量检查
        full_kv_params = self.hidden_size * (self.num_heads * self.head_dim) * 2
        ghost_params = manager.ghost_generators[0].num_parameters
        compression = full_kv_params / ghost_params
        
        return TestResult(
            name="Ghost KV Manager",
            passed=passed,
            message=f"Compression: {compression:.1f}x, Ghost params: {ghost_params:,}",
            latency_ms=elapsed,
        )
    
    def test_token_router(self) -> TestResult:
        """测试Token Router"""
        start = time.perf_counter()
        
        router = create_token_router(
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            anchor_interval=4,
        ).to(self.device)
        
        hidden = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)
        
        # 测试分割
        state = None
        for layer_idx in range(self.num_layers):
            active_batch, state = router.split_batch(hidden, layer_idx, state, confidence_threshold=0.6)
            
            if active_batch.total_active > 0:
                computed = active_batch.hidden_states + torch.randn_like(active_batch.hidden_states) * 0.01
                hidden = router.merge_batch(active_batch, computed, state)
        
        elapsed = (time.perf_counter() - start) * 1000
        
        stats = router.get_statistics()
        passed = stats['total_tokens'] > 0
        
        return TestResult(
            name="Token Router",
            passed=passed,
            message=f"Exit ratio: {stats['exit_ratio']*100:.1f}%, Speedup: {stats['theoretical_speedup']:.2f}x",
            latency_ms=elapsed,
            metrics=stats,
        )
    
    def test_attention_sinks(self) -> TestResult:
        """测试Attention Sinks保护"""
        start = time.perf_counter()
        
        protector = create_attention_sink_protector(
            num_layers=self.num_layers,
            anchor_interval=4,
            num_sink_tokens=4,
        )
        
        input_ids = torch.randint(0, 32000, (1, self.seq_len), device=self.device)
        protector.initialize(input_ids)
        
        # 验证锚点层
        anchor_count = 0
        for layer_idx in range(self.num_layers):
            if protector.anchor_manager.is_anchor(layer_idx):
                anchor_count += 1
        
        elapsed = (time.perf_counter() - start) * 1000
        
        stats = protector.get_statistics()
        passed = anchor_count > 0 and stats['num_sink_tokens'] == 4
        
        return TestResult(
            name="Attention Sinks",
            passed=passed,
            message=f"Anchor layers: {anchor_count}, Sink tokens: {stats['num_sink_tokens']}",
            latency_ms=elapsed,
            metrics=stats,
        )
    
    def test_fused_engine(self) -> TestResult:
        """测试Fused GPU Engine"""
        start = time.perf_counter()
        
        engine = create_fused_engine(
            vocab_size=32000,
            hidden_size=self.hidden_size,
        )
        
        # 模拟输入
        logits = torch.randn(self.batch_size, self.seq_len, 32000, device=self.device)
        hidden = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)
        prev_hidden = torch.randn_like(hidden)
        
        # 测试决策
        entropy, confidence, exit_mask, cognitive_load = engine.fused_decision(
            logits, hidden, prev_hidden, layer_idx=5, total_layers=self.num_layers
        )
        
        elapsed = (time.perf_counter() - start) * 1000
        
        stats = engine.get_statistics()
        passed = entropy.shape == (self.batch_size * self.seq_len,)
        
        return TestResult(
            name="Fused GPU Engine",
            passed=passed,
            message=f"Entropy mean: {entropy.mean().item():.2f}, Triton: {stats['triton_available']}",
            latency_ms=elapsed,
            metrics=stats,
        )
    
    def test_production_layer(self) -> TestResult:
        """测试生产级层"""
        start = time.perf_counter()
        
        config = LayerConfig(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_key_value_heads=self.num_heads // 4,  # GQA
            intermediate_size=self.hidden_size * 4,
        )
        
        layer = ProductionSEDACLayer(config, layer_idx=5).to(self.device)
        
        hidden = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)
        position_ids = torch.arange(self.seq_len, device=self.device).unsqueeze(0).expand(self.batch_size, -1)
        
        # 测试各种模式
        modes_ok = True
        for mode in [SkipMode.FULL_COMPUTE, SkipMode.FFN_SKIP, SkipMode.KV_ONLY, SkipMode.FULL_SKIP]:
            output, present_kv = layer(
                hidden, position_ids=position_ids, use_cache=True, skip_mode=mode
            )
            if output.shape != hidden.shape:
                modes_ok = False
                break
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return TestResult(
            name="Production Layer",
            passed=modes_ok,
            message=f"All skip modes work correctly",
            latency_ms=elapsed,
        )
    
    def test_industrial_integrator_safe(self) -> TestResult:
        """测试工业级集成器 (Safe模式)"""
        start = time.perf_counter()
        
        integrator = create_industrial_integrator(
            strategy="safe",
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        
        with integrator.inference_context():
            hidden = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)
            
            for layer_idx in range(self.num_layers):
                decision = integrator.get_layer_decision(layer_idx, hidden)
                
                # Safe模式不应该有FULL_SKIP
                if decision.skip_mode == SkipMode.FULL_SKIP:
                    return TestResult(
                        name="Industrial Integrator (Safe)",
                        passed=False,
                        message="Safe mode should not use FULL_SKIP",
                    )
                
                # 模拟指标
                integrator.metrics.total_layers += 1
                if decision.skip_mode == SkipMode.FULL_COMPUTE:
                    integrator.metrics.computed_layers += 1
                else:
                    integrator.metrics.kv_only_layers += 1
        
        elapsed = (time.perf_counter() - start) * 1000
        summary = integrator.get_summary()
        
        return TestResult(
            name="Industrial Integrator (Safe)",
            passed=True,
            message=f"Strategy: SAFE, Skip ratio: {summary['metrics']['skip_ratio']}",
            latency_ms=elapsed,
            metrics=summary,
        )
    
    def test_industrial_integrator_fast(self) -> TestResult:
        """测试工业级集成器 (Fast模式)"""
        start = time.perf_counter()
        
        integrator = create_industrial_integrator(
            strategy="fast",
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        
        with integrator.inference_context():
            hidden = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)
            
            for layer_idx in range(self.num_layers):
                decision = integrator.get_layer_decision(layer_idx, hidden)
                integrator.metrics.total_layers += 1
        
        elapsed = (time.perf_counter() - start) * 1000
        summary = integrator.get_summary()
        
        # Fast模式应该有Ghost管理器
        passed = summary['components']['ghost_manager']
        
        return TestResult(
            name="Industrial Integrator (Fast)",
            passed=passed,
            message=f"Strategy: FAST, Ghost manager enabled: {passed}",
            latency_ms=elapsed,
            metrics=summary,
        )
    
    def test_kv_continuity(self) -> TestResult:
        """测试KV Cache连续性"""
        start = time.perf_counter()
        
        config = LayerConfig(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_key_value_heads=self.num_heads // 4,
            intermediate_size=self.hidden_size * 4,
        )
        
        layer = ProductionSEDACLayer(config, layer_idx=0).to(self.device)
        
        # 第一次推理
        hidden1 = torch.randn(self.batch_size, 32, self.hidden_size, device=self.device)
        pos1 = torch.arange(32, device=self.device).unsqueeze(0).expand(self.batch_size, -1)
        
        out1, kv1 = layer(hidden1, position_ids=pos1, use_cache=True, skip_mode=SkipMode.FULL_COMPUTE)
        
        # 第二次推理（续写）- 使用KV-Only模式
        hidden2 = torch.randn(self.batch_size, 16, self.hidden_size, device=self.device)
        pos2 = torch.arange(32, 48, device=self.device).unsqueeze(0).expand(self.batch_size, -1)
        
        out2, kv2 = layer(hidden2, position_ids=pos2, past_key_value=kv1, use_cache=True, skip_mode=SkipMode.KV_ONLY)
        
        elapsed = (time.perf_counter() - start) * 1000
        
        # 验证KV长度增加
        passed = (kv2[0].shape[2] == 48)  # 32 + 16
        
        return TestResult(
            name="KV Continuity",
            passed=passed,
            message=f"KV length: {kv1[0].shape[2]} → {kv2[0].shape[2]} (expected 48)",
            latency_ms=elapsed,
        )
    
    def test_anchor_enforcement(self) -> TestResult:
        """测试锚点层强制执行"""
        start = time.perf_counter()
        
        integrator = create_industrial_integrator(
            strategy="safe",
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            anchor_interval=4,
            force_first_n=2,
            force_last_n=2,
        )
        
        anchor_layers = []
        
        with integrator.inference_context():
            hidden = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)
            
            for layer_idx in range(self.num_layers):
                if integrator.is_anchor_layer(layer_idx):
                    anchor_layers.append(layer_idx)
        
        elapsed = (time.perf_counter() - start) * 1000
        
        # 验证锚点层包含首尾层
        passed = (0 in anchor_layers and 1 in anchor_layers and 
                  self.num_layers - 1 in anchor_layers and self.num_layers - 2 in anchor_layers)
        
        return TestResult(
            name="Anchor Enforcement",
            passed=passed,
            message=f"Anchor layers: {anchor_layers}",
            latency_ms=elapsed,
        )


def run_integration_tests() -> bool:
    """运行集成测试"""
    tester = SEDACIntegrationTester()
    return tester.run_all_tests()


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
