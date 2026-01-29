"""
SEDAC V9.0 Production Tests

生产级测试套件
覆盖核心功能、边界条件、性能回归
"""
from __future__ import annotations
import torch
import torch.nn as nn
import unittest
import time
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class TestProductionConfig(unittest.TestCase):
    """配置测试"""
    
    def test_default_config(self):
        from .config import ProductionConfig, ModelConfig
        
        config = ProductionConfig()
        self.assertIsNotNone(config.model)
        self.assertIsNotNone(config.sedac)
        self.assertIsNotNone(config.performance)
    
    def test_config_validation(self):
        from .config import ProductionConfig
        
        config = ProductionConfig()
        errors = config.validate()
        self.assertEqual(len(errors), 0, f"Validation errors: {errors}")
    
    def test_config_serialization(self):
        from .config import ProductionConfig
        import tempfile
        import os
        
        config = ProductionConfig()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.save_json(f.name)
            path = f.name
        
        try:
            loaded = ProductionConfig.from_json(path)
            self.assertEqual(config.precision.value, loaded.precision.value)
        finally:
            os.unlink(path)
    
    def test_dtype_mapping(self):
        from .config import ProductionConfig, PrecisionMode
        
        config = ProductionConfig()
        
        config.precision = PrecisionMode.FP32
        self.assertEqual(config.get_dtype(), torch.float32)
        
        config.precision = PrecisionMode.FP16
        self.assertEqual(config.get_dtype(), torch.float16)


class TestEntropyComputer(unittest.TestCase):
    """熵计算测试"""
    
    def setUp(self):
        from .config import ProductionConfig
        from .engine import EntropyComputer
        
        self.config = ProductionConfig()
        self.config.device = "cpu"
        self.computer = EntropyComputer(self.config)
    
    def test_entropy_range(self):
        batch, vocab = 4, 1000
        logits = torch.randn(batch, vocab)
        
        entropy, confidence = self.computer.compute(logits)
        
        self.assertEqual(entropy.shape, (batch,))
        self.assertTrue((entropy >= 0).all())
        self.assertTrue((confidence >= 0).all())
        self.assertTrue((confidence <= 1).all())
    
    def test_low_entropy_confident(self):
        batch, vocab = 4, 1000
        logits = torch.zeros(batch, vocab)
        logits[:, 0] = 100.0
        
        entropy, confidence = self.computer.compute(logits)
        
        self.assertTrue((entropy < 1.0).all())
        self.assertTrue((confidence > 0.9).all())
    
    def test_high_entropy_uncertain(self):
        batch, vocab = 4, 1000
        logits = torch.zeros(batch, vocab)
        
        entropy, confidence = self.computer.compute(logits)
        
        expected_max_entropy = torch.log2(torch.tensor(vocab, dtype=torch.float32))
        self.assertTrue((entropy > expected_max_entropy * 0.9).all())


class TestAdaptiveThreshold(unittest.TestCase):
    """自适应阈值测试"""
    
    def setUp(self):
        from .config import ProductionConfig
        from .engine import AdaptiveThresholdController
        
        self.config = ProductionConfig()
        self.controller = AdaptiveThresholdController(self.config)
    
    def test_initial_threshold(self):
        threshold = self.controller.get_threshold(0.5)
        self.assertGreater(threshold, 0)
        self.assertLess(threshold, 1)
    
    def test_threshold_adaptation(self):
        low_entropy = torch.tensor([1.0, 1.5, 2.0])
        high_entropy = torch.tensor([5.0, 5.5, 6.0])
        
        for _ in range(200):
            self.controller.update(low_entropy)
        
        low_mean = self.controller._entropy_mean
        
        for _ in range(200):
            self.controller.update(high_entropy)
        
        high_mean = self.controller._entropy_mean
        
        self.assertLess(low_mean, high_mean)


class TestGhostKVGenerator(unittest.TestCase):
    """Ghost KV 生成器测试"""
    
    def setUp(self):
        from .config import ProductionConfig
        from .engine import GhostKVGenerator
        
        self.config = ProductionConfig()
        self.config.device = "cpu"
        self.ghost_kv = GhostKVGenerator(self.config)
    
    def test_output_shape(self):
        batch, seq = 2, 32
        hidden = torch.randn(batch, seq, self.config.model.hidden_size)
        
        kv_pairs = self.ghost_kv(hidden, num_skip_layers=4)
        
        self.assertEqual(len(kv_pairs), 4)
        
        for k, v in kv_pairs:
            self.assertEqual(k.shape[0], batch)
            self.assertEqual(k.shape[2], seq)
            self.assertEqual(k.shape[1], self.config.model.num_key_value_heads)
            self.assertEqual(k.shape[3], self.config.model.head_dim)
    
    def test_gradient_flow(self):
        batch, seq = 2, 32
        hidden = torch.randn(batch, seq, self.config.model.hidden_size, requires_grad=True)
        
        kv_pairs = self.ghost_kv(hidden, num_skip_layers=2)
        
        loss = sum(k.sum() + v.sum() for k, v in kv_pairs)
        loss.backward()
        
        self.assertIsNotNone(hidden.grad)
        self.assertTrue((hidden.grad != 0).any())


class TestO1Controller(unittest.TestCase):
    """O1 推理控制器测试"""
    
    def setUp(self):
        from .config import ProductionConfig
        from .engine import O1ReasoningController
        
        self.config = ProductionConfig()
        self.controller = O1ReasoningController(self.config)
    
    def test_activation_on_high_entropy(self):
        high_entropy = 5.0
        low_confidence = 0.2
        
        should_activate = self.controller.should_activate(high_entropy, low_confidence)
        self.assertTrue(should_activate)
    
    def test_no_activation_on_low_entropy(self):
        low_entropy = 2.0
        high_confidence = 0.9
        
        should_activate = self.controller.should_activate(low_entropy, high_confidence)
        self.assertFalse(should_activate)
    
    def test_thinking_termination(self):
        for step in range(20):
            should_continue = self.controller.should_continue(4.0, 0.5, step)
            self.controller.record_step(4.0)
            
            if step >= self.config.sedac.o1_max_thinking_steps:
                self.assertFalse(should_continue)


class TestMetricsCollector(unittest.TestCase):
    """指标收集器测试"""
    
    def setUp(self):
        from .metrics import MetricsCollector
        self.collector = MetricsCollector()
    
    def test_latency_recording(self):
        for i in range(100):
            self.collector.record_latency(10.0 + i * 0.1)
        
        metrics = self.collector.get_metrics()
        self.assertEqual(metrics["latency"]["count"], 100)
        self.assertGreater(metrics["latency"]["avg_ms"], 0)
    
    def test_throughput_recording(self):
        self.collector.record_throughput(1000, 10)
        
        metrics = self.collector.get_metrics()
        self.assertEqual(metrics["throughput"]["total_tokens"], 1000)
        self.assertEqual(metrics["throughput"]["total_requests"], 10)
    
    def test_sedac_stats(self):
        for i in range(10):
            self.collector.record_sedac(
                exit_layer=i % 5 + 10,
                total_layers=28,
                entropy=3.0 + i * 0.1,
                used_ghost_kv=(i % 2 == 0),
                used_o1=(i % 3 == 0),
            )
        
        metrics = self.collector.get_metrics()
        self.assertGreater(metrics["sedac"]["skip_ratio"], 0)
    
    def test_prometheus_export(self):
        self.collector.record_latency(10.0)
        self.collector.record_throughput(100)
        
        prometheus_text = self.collector.export_prometheus()
        
        self.assertIn("sedac_latency", prometheus_text)
        self.assertIn("sedac_tokens", prometheus_text)


class TestProductionEngine(unittest.TestCase):
    """生产引擎测试"""
    
    def setUp(self):
        from .config import ProductionConfig
        from .engine import ProductionSEDACEngine
        
        self.config = ProductionConfig()
        self.config.device = "cpu"
        self.config.sedac.enable_ghost_kv = False
        
        self.engine = ProductionSEDACEngine(self.config)
    
    def test_engine_initialization(self):
        self.assertIsNotNone(self.engine.entropy_computer)
        self.assertIsNotNone(self.engine.threshold_controller)
        self.assertIsNotNone(self.engine.o1_controller)
        self.assertIsNotNone(self.engine.metrics)
    
    def test_should_exit_early_layers(self):
        batch, seq = 2, 32
        hidden = torch.randn(batch, seq, self.config.model.hidden_size)
        logits = torch.randn(batch, seq, self.config.model.vocab_size)
        
        exit_mask, entropy, confidence = self.engine.should_exit(
            hidden, logits, layer_idx=1, total_layers=28
        )
        
        self.assertTrue((~exit_mask).all())
    
    def test_metrics_collection(self):
        metrics = self.engine.get_metrics()
        
        self.assertIn("latency", metrics)
        self.assertIn("throughput", metrics)
        self.assertIn("sedac", metrics)


class IntegrationTests(unittest.TestCase):
    """集成测试 (需要 GPU 和真实模型)"""
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_computation(self):
        """测试 CUDA 计算"""
        from .config import ProductionConfig
        from .engine import EntropyComputer
        
        config = ProductionConfig()
        config.device = "cuda"
        
        computer = EntropyComputer(config)
        
        logits = torch.randn(4, 32000, device="cuda")
        entropy, confidence = computer.compute(logits)
        
        self.assertEqual(entropy.device.type, "cuda")
        self.assertEqual(entropy.shape[0], 4)


def run_tests(verbosity: int = 2) -> unittest.TestResult:
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestProductionConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestEntropyComputer))
    suite.addTests(loader.loadTestsFromTestCase(TestAdaptiveThreshold))
    suite.addTests(loader.loadTestsFromTestCase(TestGhostKVGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestO1Controller))
    suite.addTests(loader.loadTestsFromTestCase(TestMetricsCollector))
    suite.addTests(loader.loadTestsFromTestCase(TestProductionEngine))
    suite.addTests(loader.loadTestsFromTestCase(IntegrationTests))
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run_tests()
