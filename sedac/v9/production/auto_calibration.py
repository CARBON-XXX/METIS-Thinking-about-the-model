"""
SEDAC V9.0 Auto-Calibration Module

自动校准 SEDAC 核心参数
阈值不再是人工设定，而是从数据中学习

核心理念:
    1. 熵阈值 - 根据模型输出分布自动确定
    2. 早退层数 - 根据任务复杂度自适应
    3. O1 激活条件 - 根据高熵样本分布学习
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from collections import deque
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CalibrationStats:
    """校准统计"""
    entropy_samples: List[float] = field(default_factory=list)
    confidence_samples: List[float] = field(default_factory=list)
    exit_layer_samples: List[int] = field(default_factory=list)
    output_quality_samples: List[float] = field(default_factory=list)
    
    def add(
        self,
        entropy: float,
        confidence: float,
        exit_layer: int,
        quality: float = 1.0,
    ) -> None:
        self.entropy_samples.append(entropy)
        self.confidence_samples.append(confidence)
        self.exit_layer_samples.append(exit_layer)
        self.output_quality_samples.append(quality)
    
    def get_percentile(self, data: List[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p)
        return sorted_data[min(idx, len(sorted_data) - 1)]


@dataclass
class CalibratedParameters:
    """校准后的参数"""
    entropy_threshold_base: float = 0.5
    entropy_threshold_min: float = 0.2
    entropy_threshold_max: float = 0.9
    
    min_exit_layer: int = 4
    max_skip_layers: int = 8
    exit_confidence_threshold: float = 0.85
    
    o1_high_entropy_threshold: float = 4.5
    o1_confidence_threshold: float = 0.3
    o1_max_thinking_steps: int = 8
    
    calibration_samples: int = 0
    calibration_quality: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entropy": {
                "base": self.entropy_threshold_base,
                "min": self.entropy_threshold_min,
                "max": self.entropy_threshold_max,
            },
            "exit": {
                "min_layer": self.min_exit_layer,
                "max_skip": self.max_skip_layers,
                "confidence": self.exit_confidence_threshold,
            },
            "o1": {
                "high_entropy": self.o1_high_entropy_threshold,
                "confidence": self.o1_confidence_threshold,
                "max_steps": self.o1_max_thinking_steps,
            },
            "meta": {
                "samples": self.calibration_samples,
                "quality": self.calibration_quality,
            },
        }
    
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "CalibratedParameters":
        with open(path) as f:
            data = json.load(f)
        
        params = cls()
        params.entropy_threshold_base = data["entropy"]["base"]
        params.entropy_threshold_min = data["entropy"]["min"]
        params.entropy_threshold_max = data["entropy"]["max"]
        params.min_exit_layer = data["exit"]["min_layer"]
        params.max_skip_layers = data["exit"]["max_skip"]
        params.exit_confidence_threshold = data["exit"]["confidence"]
        params.o1_high_entropy_threshold = data["o1"]["high_entropy"]
        params.o1_confidence_threshold = data["o1"]["confidence"]
        params.o1_max_thinking_steps = data["o1"]["max_steps"]
        params.calibration_samples = data["meta"]["samples"]
        params.calibration_quality = data["meta"]["quality"]
        return params


class AutoCalibrator:
    """
    SEDAC 自动校准器
    
    从真实推理数据中学习最优参数
    
    校准策略:
    1. 收集推理样本的熵/置信度分布
    2. 基于分布百分位确定阈值
    3. 持续在线更新
    """
    
    def __init__(
        self,
        model_layers: int = 28,
        calibration_samples: int = 1000,
        warmup_samples: int = 100,
    ):
        self.model_layers = model_layers
        self.target_samples = calibration_samples
        self.warmup_samples = warmup_samples
        
        self.stats = CalibrationStats()
        self.params = CalibratedParameters()
        
        self._is_calibrated = False
        self._online_buffer = deque(maxlen=1000)
    
    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated
    
    @property
    def sample_count(self) -> int:
        return len(self.stats.entropy_samples)
    
    def record_sample(
        self,
        entropy: float,
        confidence: float,
        exit_layer: int,
        quality: float = 1.0,
    ) -> None:
        """记录单个推理样本"""
        self.stats.add(entropy, confidence, exit_layer, quality)
        self._online_buffer.append((entropy, confidence, exit_layer, quality))
        
        if self.sample_count >= self.warmup_samples and not self._is_calibrated:
            self._calibrate()
        
        if self._is_calibrated and self.sample_count % 100 == 0:
            self._online_update()
    
    def _calibrate(self) -> None:
        """执行校准"""
        logger.info(f"Calibrating SEDAC with {self.sample_count} samples...")
        
        entropy_arr = np.array(self.stats.entropy_samples)
        conf_arr = np.array(self.stats.confidence_samples)
        exit_arr = np.array(self.stats.exit_layer_samples)
        quality_arr = np.array(self.stats.output_quality_samples)
        
        self.params.entropy_threshold_base = float(np.percentile(entropy_arr, 50))
        self.params.entropy_threshold_min = float(np.percentile(entropy_arr, 20))
        self.params.entropy_threshold_max = float(np.percentile(entropy_arr, 80))
        
        self.params.o1_high_entropy_threshold = float(np.percentile(entropy_arr, 90))
        self.params.o1_confidence_threshold = float(np.percentile(conf_arr, 20))
        
        self.params.exit_confidence_threshold = float(np.percentile(conf_arr, 70))
        
        low_entropy_exits = exit_arr[entropy_arr < np.percentile(entropy_arr, 30)]
        if len(low_entropy_exits) > 0:
            self.params.min_exit_layer = max(2, int(np.percentile(low_entropy_exits, 10)))
        
        avg_exit = np.mean(exit_arr)
        self.params.max_skip_layers = max(4, int(self.model_layers - avg_exit))
        
        high_entropy_mask = entropy_arr > self.params.o1_high_entropy_threshold
        if high_entropy_mask.sum() > 0:
            high_entropy_exits = exit_arr[high_entropy_mask]
            self.params.o1_max_thinking_steps = max(4, int(np.percentile(high_entropy_exits, 90) / 3))
        
        self.params.calibration_samples = self.sample_count
        self.params.calibration_quality = float(np.mean(quality_arr))
        
        self._is_calibrated = True
        logger.info(f"Calibration complete: {self.params.to_dict()}")
    
    def _online_update(self) -> None:
        """在线更新参数"""
        if len(self._online_buffer) < 100:
            return
        
        recent = list(self._online_buffer)
        entropy_arr = np.array([x[0] for x in recent])
        
        alpha = 0.1
        
        new_base = float(np.percentile(entropy_arr, 50))
        self.params.entropy_threshold_base = (
            (1 - alpha) * self.params.entropy_threshold_base + 
            alpha * new_base
        )
        
        new_high = float(np.percentile(entropy_arr, 90))
        self.params.o1_high_entropy_threshold = (
            (1 - alpha) * self.params.o1_high_entropy_threshold + 
            alpha * new_high
        )
    
    def get_calibrated_params(self) -> CalibratedParameters:
        """获取校准后的参数"""
        return self.params
    
    def apply_to_config(self, config: Any) -> None:
        """将校准参数应用到配置"""
        if hasattr(config, 'sedac'):
            sedac = config.sedac
            sedac.entropy_threshold_base = self.params.entropy_threshold_base
            sedac.entropy_threshold_min = self.params.entropy_threshold_min
            sedac.entropy_threshold_max = self.params.entropy_threshold_max
            sedac.min_exit_layer = self.params.min_exit_layer
            sedac.max_skip_layers = self.params.max_skip_layers
            sedac.exit_confidence_threshold = self.params.exit_confidence_threshold
            sedac.o1_high_entropy_threshold = self.params.o1_high_entropy_threshold
            sedac.o1_max_thinking_steps = self.params.o1_max_thinking_steps
            
            logger.info("Applied calibrated parameters to config")


class ModelAwareCalibrator(AutoCalibrator):
    """
    模型感知校准器
    
    根据模型特性自动确定初始参数范围
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        calibration_prompts: Optional[List[str]] = None,
        device: str = "cuda",
    ):
        try:
            num_layers = model.config.num_hidden_layers
        except:
            num_layers = 28
        
        super().__init__(model_layers=num_layers)
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        self.calibration_prompts = calibration_prompts or self._get_default_prompts()
    
    def _get_default_prompts(self) -> List[str]:
        """默认校准提示"""
        return [
            "Hello",
            "What is 2+2?",
            "Explain quantum computing in detail.",
            "Write a poem about AI.",
            "Solve: If x^2 + 2x - 3 = 0, find x.",
            "Translate 'Hello World' to Chinese.",
            "What is the capital of France?",
            "Explain the theory of relativity.",
            "Write a function to sort a list in Python.",
            "What are the ethical implications of artificial intelligence?",
        ]
    
    @torch.no_grad()
    def run_calibration(
        self,
        num_samples: int = 100,
        max_tokens_per_sample: int = 32,
    ) -> CalibratedParameters:
        """
        运行校准
        
        Args:
            num_samples: 校准样本数
            max_tokens_per_sample: 每样本最大 token 数
        
        Returns:
            校准后的参数
        """
        logger.info(f"Running calibration with {num_samples} samples...")
        
        from .engine import EntropyComputer
        from .config import ProductionConfig
        
        config = ProductionConfig()
        config.device = self.device
        entropy_computer = EntropyComputer(config)
        
        prompts = (self.calibration_prompts * (num_samples // len(self.calibration_prompts) + 1))[:num_samples]
        
        for i, prompt in enumerate(prompts):
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.device)
            
            for step in range(max_tokens_per_sample):
                outputs = self.model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )
                
                logits = outputs.logits[:, -1, :]
                
                entropy, confidence = entropy_computer.compute(logits)
                
                entropy_val = entropy.mean().item()
                conf_val = confidence.mean().item()
                
                layer_estimate = int(self.model_layers * (1 - conf_val * 0.5))
                
                self.record_sample(
                    entropy=entropy_val,
                    confidence=conf_val,
                    exit_layer=layer_estimate,
                    quality=1.0,
                )
                
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
            
            if (i + 1) % 10 == 0:
                logger.info(f"Calibration progress: {i+1}/{num_samples}")
        
        return self.get_calibrated_params()


def calibrate_for_model(
    model_name: str,
    output_path: Optional[str] = None,
    device: str = "cuda",
    num_samples: int = 100,
) -> CalibratedParameters:
    """
    为指定模型运行校准
    
    Args:
        model_name: HuggingFace 模型名称
        output_path: 保存路径
        device: 设备
        num_samples: 样本数
    
    Returns:
        校准后的参数
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    logger.info(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    
    calibrator = ModelAwareCalibrator(model, tokenizer, device=device)
    params = calibrator.run_calibration(num_samples=num_samples)
    
    if output_path:
        params.save(output_path)
        logger.info(f"Saved calibrated parameters to {output_path}")
    
    return params


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SEDAC Auto Calibration")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", type=str, default="calibrated_params.json")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if not torch.cuda.is_available() and args.device == "cuda":
        args.device = "cpu"
    
    params = calibrate_for_model(
        args.model,
        args.output,
        args.device,
        args.samples,
    )
    
    print("\nCalibrated Parameters:")
    print(json.dumps(params.to_dict(), indent=2))
