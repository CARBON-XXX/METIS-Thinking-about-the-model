"""
SEDAC V9.0 Production Configuration

生产环境配置管理，支持多种部署场景
"""
from __future__ import annotations
import os
import json
import torch
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Literal
from pathlib import Path
from enum import Enum


class DeploymentMode(Enum):
    """部署模式"""
    SINGLE_GPU = "single_gpu"
    MULTI_GPU = "multi_gpu"
    TENSOR_PARALLEL = "tensor_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"


class PrecisionMode(Enum):
    """精度模式"""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


class KernelBackend(Enum):
    """内核后端"""
    CUDA_CPP = "cuda_cpp"
    TRITON = "triton"
    PYTORCH = "pytorch"
    TENSORRT = "tensorrt"


@dataclass
class ModelConfig:
    """模型架构配置"""
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    hidden_size: int = 3584
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    num_hidden_layers: int = 28
    intermediate_size: int = 18944
    vocab_size: int = 152064
    max_position_embeddings: int = 32768
    head_dim: int = 128
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "ModelConfig":
        """从预训练模型加载配置"""
        try:
            from transformers import AutoConfig
            hf_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            return cls(
                model_name=model_name_or_path,
                hidden_size=getattr(hf_config, "hidden_size", 3584),
                num_attention_heads=getattr(hf_config, "num_attention_heads", 28),
                num_key_value_heads=getattr(hf_config, "num_key_value_heads", 4),
                num_hidden_layers=getattr(hf_config, "num_hidden_layers", 28),
                intermediate_size=getattr(hf_config, "intermediate_size", 18944),
                vocab_size=getattr(hf_config, "vocab_size", 152064),
                max_position_embeddings=getattr(hf_config, "max_position_embeddings", 32768),
                head_dim=getattr(hf_config, "head_dim", 128),
                rope_theta=getattr(hf_config, "rope_theta", 1000000.0),
                rms_norm_eps=getattr(hf_config, "rms_norm_eps", 1e-6),
            )
        except Exception as e:
            raise ValueError(f"Failed to load model config: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SEDACConfig:
    """
    SEDAC 核心配置
    
    注意: 以下参数均为初始默认值
    实际运行时会通过 AutoCalibrator 自动校准:
    - entropy_threshold_* : 根据模型输出熵分布自动确定
    - min_exit_layer/max_skip_layers: 根据任务复杂度自适应
    - o1_high_entropy_threshold: 根据高熵样本分布学习
    
    启用 auto_calibrate=True 后，这些值会被自动覆盖
    """
    # === 自适应校准开关 ===
    auto_calibrate: bool = True  # 启用自动校准
    calibration_samples: int = 100  # 校准样本数
    calibration_file: Optional[str] = None  # 加载已校准参数
    
    # === 熵阈值 (自动校准目标) ===
    # 这些值将根据模型输出分布自动调整
    entropy_threshold_base: float = 0.5  # 初始值，会被校准覆盖
    entropy_threshold_min: float = 0.2
    entropy_threshold_max: float = 0.9
    
    # === 早退配置 (自动校准目标) ===
    min_exit_layer: int = 4  # 最小退出层，根据模型深度调整
    max_skip_layers: int = 8  # 最大跳过层数
    exit_confidence_threshold: float = 0.85  # 退出置信度阈值
    
    # === 自适应阈值 (在线学习) ===
    adaptive_threshold: bool = True
    threshold_ema_decay: float = 0.99
    threshold_warmup_steps: int = 100
    
    # === Ghost KV ===
    enable_ghost_kv: bool = True
    ghost_kv_similarity_target: float = 0.98
    ghost_kv_mlp_hidden_mult: float = 0.25
    
    # === O1 推理 (自动校准目标) ===
    enable_o1_reasoning: bool = True
    o1_high_entropy_threshold: float = 4.5  # 高熵阈值，根据分布 P90 自动确定
    o1_max_thinking_steps: int = 8
    o1_confidence_target: float = 0.9
    
    # === Token 路由 ===
    enable_token_routing: bool = True
    routing_batch_threshold: int = 4
    
    def apply_calibration(self, params: Any) -> None:
        """应用校准参数"""
        if hasattr(params, 'entropy_threshold_base'):
            self.entropy_threshold_base = params.entropy_threshold_base
            self.entropy_threshold_min = params.entropy_threshold_min
            self.entropy_threshold_max = params.entropy_threshold_max
            self.min_exit_layer = params.min_exit_layer
            self.max_skip_layers = params.max_skip_layers
            self.exit_confidence_threshold = params.exit_confidence_threshold
            self.o1_high_entropy_threshold = params.o1_high_entropy_threshold
            self.o1_max_thinking_steps = params.o1_max_thinking_steps


@dataclass
class PerformanceConfig:
    """性能优化配置"""
    # 内核选择
    kernel_backend: KernelBackend = KernelBackend.CUDA_CPP
    
    # CUDA 优化
    use_cuda_graphs: bool = True
    cuda_graph_warmup_iters: int = 3
    enable_flash_attention: bool = True
    
    # 内存优化
    enable_kv_cache_quantization: bool = False
    kv_cache_dtype: str = "float16"
    max_kv_cache_length: int = 32768
    
    # 批处理
    max_batch_size: int = 64
    dynamic_batching: bool = True
    batch_timeout_ms: float = 5.0
    
    # 并行
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1


@dataclass
class ProductionConfig:
    """生产环境完整配置"""
    # 子配置
    model: ModelConfig = field(default_factory=ModelConfig)
    sedac: SEDACConfig = field(default_factory=SEDACConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # 部署配置
    deployment_mode: DeploymentMode = DeploymentMode.SINGLE_GPU
    precision: PrecisionMode = PrecisionMode.FP16
    device: str = "cuda:0"
    
    # 日志与监控
    enable_metrics: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"
    enable_profiling: bool = False
    
    # 检查点
    checkpoint_dir: str = "./checkpoints"
    save_interval_steps: int = 1000
    
    @classmethod
    def from_yaml(cls, path: str) -> "ProductionConfig":
        """从 YAML 文件加载配置"""
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)
    
    @classmethod
    def from_json(cls, path: str) -> "ProductionConfig":
        """从 JSON 文件加载配置"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "ProductionConfig":
        """从字典构建配置"""
        model_data = data.get("model", {})
        sedac_data = data.get("sedac", {})
        perf_data = data.get("performance", {})
        
        return cls(
            model=ModelConfig(**model_data) if model_data else ModelConfig(),
            sedac=SEDACConfig(**sedac_data) if sedac_data else SEDACConfig(),
            performance=PerformanceConfig(**perf_data) if perf_data else PerformanceConfig(),
            deployment_mode=DeploymentMode(data.get("deployment_mode", "single_gpu")),
            precision=PrecisionMode(data.get("precision", "fp16")),
            device=data.get("device", "cuda:0"),
            enable_metrics=data.get("enable_metrics", True),
            metrics_port=data.get("metrics_port", 9090),
            log_level=data.get("log_level", "INFO"),
            enable_profiling=data.get("enable_profiling", False),
            checkpoint_dir=data.get("checkpoint_dir", "./checkpoints"),
            save_interval_steps=data.get("save_interval_steps", 1000),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return {
            "model": self.model.to_dict(),
            "sedac": asdict(self.sedac),
            "performance": asdict(self.performance),
            "deployment_mode": self.deployment_mode.value,
            "precision": self.precision.value,
            "device": self.device,
            "enable_metrics": self.enable_metrics,
            "metrics_port": self.metrics_port,
            "log_level": self.log_level,
            "enable_profiling": self.enable_profiling,
            "checkpoint_dir": self.checkpoint_dir,
            "save_interval_steps": self.save_interval_steps,
        }
    
    def save_json(self, path: str) -> None:
        """保存为 JSON"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    def validate(self) -> List[str]:
        """验证配置有效性"""
        errors = []
        
        if self.sedac.min_exit_layer >= self.model.num_hidden_layers:
            errors.append(f"min_exit_layer ({self.sedac.min_exit_layer}) >= num_layers ({self.model.num_hidden_layers})")
        
        if self.sedac.max_skip_layers > self.model.num_hidden_layers - self.sedac.min_exit_layer:
            errors.append("max_skip_layers too large for model depth")
        
        if self.performance.tensor_parallel_size > 1 and self.deployment_mode == DeploymentMode.SINGLE_GPU:
            errors.append("tensor_parallel_size > 1 requires MULTI_GPU or TENSOR_PARALLEL mode")
        
        if self.precision == PrecisionMode.BF16 and not torch.cuda.is_bf16_supported():
            errors.append("BF16 not supported on this GPU")
        
        return errors
    
    def get_dtype(self) -> torch.dtype:
        """获取 PyTorch 数据类型"""
        dtype_map = {
            PrecisionMode.FP32: torch.float32,
            PrecisionMode.FP16: torch.float16,
            PrecisionMode.BF16: torch.bfloat16,
            PrecisionMode.INT8: torch.int8,
            PrecisionMode.INT4: torch.int8,
        }
        return dtype_map.get(self.precision, torch.float16)


def get_default_config(model_name: str = "Qwen/Qwen2.5-7B-Instruct") -> ProductionConfig:
    """获取默认生产配置"""
    config = ProductionConfig()
    try:
        config.model = ModelConfig.from_pretrained(model_name)
    except:
        pass
    return config
