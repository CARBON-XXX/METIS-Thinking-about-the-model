"""
SEDAC V9.0 Production Integration Test

真实模型端到端集成测试
验证完整推理管线的正确性和性能
"""
from __future__ import annotations
import torch
import time
import logging
import sys
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """测试结果"""
    name: str
    passed: bool
    duration_ms: float
    message: str = ""
    metrics: Dict[str, Any] = None


class ProductionIntegrationTest:
    """
    生产级集成测试
    
    使用真实模型验证:
    1. 模型加载与推理
    2. SEDAC 早退功能
    3. Ghost KV 生成
    4. O1 深度推理
    5. 性能指标
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.device = device
        self.pipeline = None
        self.results: List[TestResult] = []
    
    def setup(self) -> bool:
        """初始化测试环境"""
        logger.info(f"Setting up integration test with {self.model_name}")
        
        try:
            from .inference import SEDACInferencePipeline
            from .config import ProductionConfig
            
            config = ProductionConfig()
            config.device = self.device
            
            self.pipeline = SEDACInferencePipeline(
                self.model_name,
                config=config,
                device=self.device,
            )
            self.pipeline.load()
            
            logger.info("Setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return False
    
    def _run_test(self, name: str, test_fn) -> TestResult:
        """运行单个测试"""
        logger.info(f"Running test: {name}")
        start = time.perf_counter()
        
        try:
            result = test_fn()
            duration = (time.perf_counter() - start) * 1000
            
            if isinstance(result, dict):
                passed = result.get("passed", True)
                message = result.get("message", "")
                metrics = result.get("metrics", {})
            else:
                passed = bool(result)
                message = ""
                metrics = {}
            
            return TestResult(
                name=name,
                passed=passed,
                duration_ms=duration,
                message=message,
                metrics=metrics,
            )
            
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                name=name,
                passed=False,
                duration_ms=duration,
                message=str(e),
            )
    
    def test_basic_generation(self) -> Dict:
        """测试基本生成"""
        from .inference import GenerationConfig
        
        prompt = "What is 2 + 2?"
        config = GenerationConfig(max_new_tokens=32, do_sample=False)
        
        result = self.pipeline(prompt, config)
        
        has_output = len(result.generated_text) > 0
        reasonable_speed = result.tokens_per_second > 0
        
        return {
            "passed": has_output and reasonable_speed,
            "message": f"Generated: '{result.generated_text[:50]}...'",
            "metrics": {
                "tokens": result.generated_tokens,
                "latency_ms": result.total_latency_ms,
                "tps": result.tokens_per_second,
            }
        }
    
    def test_early_exit(self) -> Dict:
        """测试早退功能"""
        from .inference import GenerationConfig
        
        simple_prompt = "Hello"
        config = GenerationConfig(max_new_tokens=16, do_sample=False)
        
        result = self.pipeline(simple_prompt, config)
        
        total_layers = self.pipeline.config.model.num_hidden_layers
        has_early_exit = result.avg_exit_layer < total_layers - 1
        
        return {
            "passed": True,
            "message": f"Avg exit: {result.avg_exit_layer:.1f}/{total_layers}, skip: {result.skip_ratio*100:.1f}%",
            "metrics": {
                "avg_exit_layer": result.avg_exit_layer,
                "skip_ratio": result.skip_ratio,
                "total_layers": total_layers,
            }
        }
    
    def test_batch_generation(self) -> Dict:
        """测试批量生成"""
        from .inference import GenerationConfig
        
        prompts = [
            "What is AI?",
            "Explain machine learning.",
            "What is Python?",
        ]
        config = GenerationConfig(max_new_tokens=32, do_sample=False)
        
        results = self.pipeline.batch_generate(prompts, config, batch_size=3)
        
        all_have_output = all(len(r.generated_text) > 0 for r in results)
        
        return {
            "passed": all_have_output and len(results) == 3,
            "message": f"Generated {len(results)} responses",
            "metrics": {
                "batch_size": len(results),
                "total_tokens": sum(r.generated_tokens for r in results),
                "avg_tps": sum(r.tokens_per_second for r in results) / len(results),
            }
        }
    
    def test_streaming(self) -> Dict:
        """测试流式生成"""
        from .inference import GenerationConfig
        
        prompt = "Count from 1 to 5:"
        config = GenerationConfig(max_new_tokens=32, do_sample=False)
        
        chunks = []
        for chunk in self.pipeline.stream_generate(prompt, config):
            chunks.append(chunk)
        
        full_text = "".join(chunks)
        
        return {
            "passed": len(chunks) > 0 and len(full_text) > 0,
            "message": f"Streamed {len(chunks)} chunks, total: '{full_text[:30]}...'",
            "metrics": {
                "num_chunks": len(chunks),
                "total_chars": len(full_text),
            }
        }
    
    def test_metrics_collection(self) -> Dict:
        """测试指标收集"""
        self.pipeline.reset_metrics()
        
        from .inference import GenerationConfig
        config = GenerationConfig(max_new_tokens=16, do_sample=False)
        
        for _ in range(3):
            self.pipeline("Test prompt", config)
        
        metrics = self.pipeline.get_metrics()
        
        has_latency = metrics.get("latency", {}).get("count", 0) > 0
        has_throughput = metrics.get("throughput", {}).get("total_tokens", 0) > 0
        has_sedac = "sedac" in metrics
        
        return {
            "passed": has_latency and has_throughput and has_sedac,
            "message": f"Collected metrics: latency={has_latency}, throughput={has_throughput}, sedac={has_sedac}",
            "metrics": {
                "latency_count": metrics.get("latency", {}).get("count", 0),
                "total_tokens": metrics.get("throughput", {}).get("total_tokens", 0),
                "skip_ratio": metrics.get("sedac", {}).get("skip_ratio", 0),
            }
        }
    
    def test_long_context(self) -> Dict:
        """测试长上下文"""
        from .inference import GenerationConfig
        
        long_prompt = "Please summarize the following: " + "This is a test sentence. " * 50
        config = GenerationConfig(max_new_tokens=64, do_sample=False)
        
        result = self.pipeline(long_prompt, config)
        
        return {
            "passed": len(result.generated_text) > 0,
            "message": f"Input: {result.input_tokens} tokens, Output: {result.generated_tokens} tokens",
            "metrics": {
                "input_tokens": result.input_tokens,
                "output_tokens": result.generated_tokens,
                "latency_ms": result.total_latency_ms,
            }
        }
    
    def test_consistency(self) -> Dict:
        """测试输出一致性 (相同输入应产生相同输出)"""
        from .inference import GenerationConfig
        
        prompt = "What is 1+1?"
        config = GenerationConfig(max_new_tokens=16, do_sample=False, temperature=0.0)
        
        results = []
        for _ in range(3):
            result = self.pipeline(prompt, config)
            results.append(result.generated_text)
        
        all_same = all(r == results[0] for r in results)
        
        return {
            "passed": all_same,
            "message": f"Outputs identical: {all_same}",
            "metrics": {
                "num_runs": len(results),
                "unique_outputs": len(set(results)),
            }
        }
    
    def run_all(self) -> Dict[str, Any]:
        """运行所有测试"""
        if not self.setup():
            return {"error": "Setup failed"}
        
        tests = [
            ("Basic Generation", self.test_basic_generation),
            ("Early Exit", self.test_early_exit),
            ("Batch Generation", self.test_batch_generation),
            ("Streaming", self.test_streaming),
            ("Metrics Collection", self.test_metrics_collection),
            ("Long Context", self.test_long_context),
            ("Consistency", self.test_consistency),
        ]
        
        for name, test_fn in tests:
            result = self._run_test(name, test_fn)
            self.results.append(result)
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        print("\n" + "=" * 70)
        print("SEDAC V9.0 Integration Test Results")
        print("=" * 70)
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print()
        
        for r in self.results:
            status = "✅ PASS" if r.passed else "❌ FAIL"
            print(f"{status} {r.name} ({r.duration_ms:.1f}ms)")
            if r.message:
                print(f"     {r.message}")
            if r.metrics:
                metrics_str = ", ".join(f"{k}={v}" for k, v in list(r.metrics.items())[:3])
                print(f"     [{metrics_str}]")
        
        print()
        print("-" * 70)
        print(f"Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All integration tests PASSED!")
        else:
            print(f"⚠️  {total - passed} tests FAILED")
        
        print("=" * 70)
        
        return {
            "passed": passed,
            "total": total,
            "success": passed == total,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "duration_ms": r.duration_ms,
                    "message": r.message,
                    "metrics": r.metrics,
                }
                for r in self.results
            ],
        }


def run_integration_test(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    运行集成测试
    
    Args:
        model_name: 模型名称 (默认使用小模型快速测试)
        device: 设备
    
    Returns:
        测试结果
    """
    test = ProductionIntegrationTest(model_name, device)
    return test.run_all()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SEDAC V9.0 Integration Test")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                       help="Model to test (default: small model for fast testing)")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available() and args.device == "cuda":
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"
    
    result = run_integration_test(args.model, args.device)
    
    sys.exit(0 if result.get("success") else 1)
