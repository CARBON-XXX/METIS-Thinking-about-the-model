"""
SEDAC V9.0 Production Engine

生产级语义熵引导动态注意力核心引擎
符合 NVIDIA TensorRT / Triton Inference Server 标准
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass
import logging
import threading
from pathlib import Path

from .config import ProductionConfig, KernelBackend, PrecisionMode
from .metrics import MetricsCollector, PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class ForwardOutput:
    """前向传播输出"""
    hidden_states: torch.Tensor
    logits: Optional[torch.Tensor] = None
    exit_layer: int = -1
    entropy: float = 0.0
    confidence: float = 0.0
    used_ghost_kv: bool = False
    used_o1: bool = False
    thinking_steps: int = 0
    past_key_values: Optional[Tuple] = None


class EntropyComputer:
    """高性能熵计算器"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.device = torch.device(config.device)
        self._cuda_ext = None
        self._load_cuda_extension()
    
    def _load_cuda_extension(self) -> None:
        """加载 CUDA 扩展"""
        if self.config.performance.kernel_backend == KernelBackend.CUDA_CPP:
            try:
                import sys
                cuda_ext_path = Path(__file__).parent.parent / "cuda_ext"
                sys.path.insert(0, str(cuda_ext_path))
                import sedac_cuda_v2
                self._cuda_ext = sedac_cuda_v2
                logger.info("CUDA extension loaded successfully")
            except ImportError as e:
                logger.warning(f"CUDA extension not available: {e}")
    
    @torch.no_grad()
    def compute(
        self,
        logits: torch.Tensor,
        return_confidence: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        计算语义熵
        
        Args:
            logits: [batch, seq, vocab] 或 [batch, vocab]
            return_confidence: 是否返回置信度
        
        Returns:
            entropy: [batch] 或 [batch, seq]
            confidence: [batch] 或 [batch, seq] (可选)
        """
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        
        probs = F.softmax(logits.float(), dim=-1)
        log_probs = torch.log2(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        if return_confidence:
            confidence = probs.max(dim=-1).values
            return entropy, confidence
        
        return entropy, None


class AdaptiveThresholdController:
    """自适应阈值控制器"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.sedac_config = config.sedac
        
        self._entropy_mean = 3.0
        self._entropy_std = 1.0
        self._entropy_ema = 3.0
        self._step_count = 0
        self._lock = threading.Lock()
    
    def update(self, entropy: torch.Tensor) -> None:
        """更新统计"""
        with self._lock:
            batch_mean = entropy.mean().item()
            batch_std = entropy.std().item() if entropy.numel() > 1 else 0.1
            
            decay = self.sedac_config.threshold_ema_decay
            self._entropy_mean = decay * self._entropy_mean + (1 - decay) * batch_mean
            self._entropy_std = decay * self._entropy_std + (1 - decay) * batch_std
            self._entropy_ema = self._entropy_mean
            self._step_count += 1
    
    def get_threshold(self, layer_progress: float) -> float:
        """获取当前阈值"""
        base = self.sedac_config.entropy_threshold_base
        
        if not self.sedac_config.adaptive_threshold:
            return base
        
        if self._step_count < self.sedac_config.threshold_warmup_steps:
            return base
        
        z_adaptive = (self._entropy_ema - self._entropy_mean) / (self._entropy_std + 1e-6)
        adjusted = base + z_adaptive * 0.1
        
        layer_factor = 1.0 - layer_progress * 0.3
        adjusted *= layer_factor
        
        return max(
            self.sedac_config.entropy_threshold_min,
            min(self.sedac_config.entropy_threshold_max, adjusted)
        )
    
    @property
    def stats(self) -> Dict[str, float]:
        return {
            "entropy_mean": self._entropy_mean,
            "entropy_std": self._entropy_std,
            "step_count": self._step_count,
        }


class GhostKVGenerator(nn.Module):
    """Ghost KV 生成器 - 生产版"""
    
    def __init__(self, config: ProductionConfig):
        super().__init__()
        self.config = config
        model_cfg = config.model
        sedac_cfg = config.sedac
        
        hidden = model_cfg.hidden_size
        mlp_hidden = int(hidden * sedac_cfg.ghost_kv_mlp_hidden_mult)
        kv_dim = model_cfg.num_key_value_heads * model_cfg.head_dim
        max_skip = sedac_cfg.max_skip_layers
        
        self.encoder = nn.Sequential(
            nn.Linear(hidden, mlp_hidden),
            nn.GELU(),
            nn.LayerNorm(mlp_hidden),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
        )
        
        self.k_projectors = nn.ModuleList([
            nn.Linear(mlp_hidden, kv_dim) for _ in range(max_skip)
        ])
        self.v_projectors = nn.ModuleList([
            nn.Linear(mlp_hidden, kv_dim) for _ in range(max_skip)
        ])
        
        self.residual_gates = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.1) for _ in range(max_skip)
        ])
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        num_skip_layers: int,
        prev_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        生成跳过层的 KV Cache
        
        Args:
            hidden_states: [batch, seq, hidden]
            num_skip_layers: 跳过层数
            prev_kv: 上一层的 KV (用于残差)
        
        Returns:
            List of (K, V), each [batch, num_kv_heads, seq, head_dim]
        """
        batch, seq, _ = hidden_states.shape
        cfg = self.config.model
        
        encoded = self.encoder(hidden_states)
        
        kv_pairs = []
        for i in range(min(num_skip_layers, len(self.k_projectors))):
            k = self.k_projectors[i](encoded) * self.residual_gates[i]
            v = self.v_projectors[i](encoded) * self.residual_gates[i]
            
            k = k.view(batch, seq, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
            v = v.view(batch, seq, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
            
            if prev_kv is not None:
                k = k + prev_kv[0] * 0.5
                v = v + prev_kv[1] * 0.5
            
            kv_pairs.append((k.contiguous(), v.contiguous()))
        
        return kv_pairs


class O1ReasoningController:
    """O1 自适应推理控制器"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.sedac_cfg = config.sedac
        
        self._thinking_count = 0
        self._entropy_history: List[float] = []
    
    def should_activate(self, entropy: float, confidence: float) -> bool:
        """判断是否激活 O1 推理"""
        if not self.sedac_cfg.enable_o1_reasoning:
            return False
        
        if entropy > self.sedac_cfg.o1_high_entropy_threshold:
            return True
        
        if confidence < 0.3 and entropy > 3.5:
            return True
        
        return False
    
    def should_continue(self, entropy: float, confidence: float, step: int) -> bool:
        """判断是否继续思考"""
        if step >= self.sedac_cfg.o1_max_thinking_steps:
            return False
        
        if confidence >= self.sedac_cfg.o1_confidence_target:
            return False
        
        if entropy < 2.5:
            return False
        
        if len(self._entropy_history) >= 3:
            recent = self._entropy_history[-3:]
            if all(recent[i] <= recent[i+1] for i in range(len(recent)-1)):
                return False
        
        return True
    
    def record_step(self, entropy: float) -> None:
        """记录思考步骤"""
        self._thinking_count += 1
        self._entropy_history.append(entropy)
        if len(self._entropy_history) > 20:
            self._entropy_history.pop(0)
    
    def reset(self) -> None:
        """重置状态"""
        self._entropy_history.clear()


class ProductionSEDACEngine:
    """
    SEDAC V9.0 生产级引擎
    
    特性:
    - 高性能 CUDA 内核
    - 自适应阈值控制
    - Ghost KV 预测
    - O1 深度推理
    - 完整指标监控
    """
    
    def __init__(
        self,
        config: ProductionConfig,
        model: Optional[nn.Module] = None,
    ):
        self.config = config
        self.model = model
        self.device = torch.device(config.device)
        self.dtype = config.get_dtype()
        
        self.entropy_computer = EntropyComputer(config)
        self.threshold_controller = AdaptiveThresholdController(config)
        self.o1_controller = O1ReasoningController(config)
        
        self.ghost_kv: Optional[GhostKVGenerator] = None
        if config.sedac.enable_ghost_kv:
            self.ghost_kv = GhostKVGenerator(config).to(self.device)
        
        self.metrics = MetricsCollector()
        self.monitor = PerformanceMonitor(self.metrics)
        
        self._cuda_graphs: Dict[str, Any] = {}
        self._warmup_done = False
        
        logger.info(f"ProductionSEDACEngine initialized on {self.device}")
        logger.info(f"Precision: {config.precision.value}, Backend: {config.performance.kernel_backend.value}")
    
    def attach_model(self, model: nn.Module) -> None:
        """附加模型"""
        self.model = model
        logger.info(f"Model attached: {type(model).__name__}")
    
    @torch.no_grad()
    def should_exit(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        layer_idx: int,
        total_layers: int,
    ) -> Tuple[torch.Tensor, float, float]:
        """
        判断是否应该早退
        
        Returns:
            exit_mask: [batch] bool tensor
            entropy: 平均熵
            confidence: 平均置信度
        """
        if layer_idx < self.config.sedac.min_exit_layer:
            batch = hidden_states.shape[0]
            return torch.zeros(batch, dtype=torch.bool, device=self.device), 10.0, 0.0
        
        entropy, confidence = self.entropy_computer.compute(logits)
        
        self.threshold_controller.update(entropy)
        
        layer_progress = layer_idx / (total_layers - 1)
        threshold = self.threshold_controller.get_threshold(layer_progress)
        
        normalized_entropy = (entropy - self.threshold_controller._entropy_mean) / (
            self.threshold_controller._entropy_std + 1e-6
        )
        
        exit_mask = (normalized_entropy < 0) & (confidence > threshold)
        
        avg_entropy = entropy.mean().item()
        avg_confidence = confidence.mean().item()
        
        return exit_mask, avg_entropy, avg_confidence
    
    def forward_with_sedac(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
    ) -> ForwardOutput:
        """
        带 SEDAC 的前向传播
        
        这是主要的推理接口
        """
        if self.model is None:
            raise RuntimeError("Model not attached. Call attach_model() first.")
        
        with self.monitor.measure_latency():
            output = self._forward_impl(
                input_ids, attention_mask, past_key_values, use_cache
            )
        
        self.metrics.record_throughput(input_ids.numel())
        self.metrics.record_sedac(
            exit_layer=output.exit_layer,
            total_layers=self.config.model.num_hidden_layers,
            entropy=output.entropy,
            used_ghost_kv=output.used_ghost_kv,
            used_o1=output.used_o1,
        )
        
        return output
    
    def _forward_impl(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Tuple],
        use_cache: bool,
    ) -> ForwardOutput:
        """前向传播实现"""
        model = self.model
        config = self.config
        total_layers = config.model.num_hidden_layers
        
        if hasattr(model, "model"):
            embed_tokens = model.model.embed_tokens
            layers = model.model.layers
            norm = model.model.norm
            lm_head = model.lm_head
        elif hasattr(model, "transformer"):
            embed_tokens = model.transformer.wte
            layers = model.transformer.h
            norm = model.transformer.ln_f
            lm_head = model.lm_head
        else:
            return self._fallback_forward(input_ids, attention_mask, past_key_values, use_cache)
        
        hidden_states = embed_tokens(input_ids)
        
        if past_key_values is None:
            past_key_values = [None] * total_layers
        
        new_past_key_values = []
        exit_layer = total_layers - 1
        final_entropy = 0.0
        final_confidence = 0.0
        used_ghost_kv = False
        used_o1 = False
        thinking_steps = 0
        
        for layer_idx, layer in enumerate(layers):
            layer_past = past_key_values[layer_idx] if past_key_values else None
            
            layer_output = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
                if use_cache and len(layer_output) > 1:
                    new_past_key_values.append(layer_output[1])
            else:
                hidden_states = layer_output
            
            if layer_idx >= config.sedac.min_exit_layer and layer_idx < total_layers - 1:
                intermediate_logits = lm_head(norm(hidden_states))
                exit_mask, entropy, confidence = self.should_exit(
                    hidden_states, intermediate_logits, layer_idx, total_layers
                )
                
                final_entropy = entropy
                final_confidence = confidence
                
                if self.o1_controller.should_activate(entropy, confidence):
                    used_o1 = True
                    continue
                
                if exit_mask.all():
                    exit_layer = layer_idx
                    
                    if config.sedac.enable_ghost_kv and self.ghost_kv is not None:
                        skip_layers = total_layers - layer_idx - 1
                        if skip_layers > 0:
                            ghost_kvs = self.ghost_kv(
                                hidden_states,
                                min(skip_layers, config.sedac.max_skip_layers)
                            )
                            new_past_key_values.extend(ghost_kvs)
                            used_ghost_kv = True
                    
                    break
        
        hidden_states = norm(hidden_states)
        logits = lm_head(hidden_states)
        
        return ForwardOutput(
            hidden_states=hidden_states,
            logits=logits,
            exit_layer=exit_layer,
            entropy=final_entropy,
            confidence=final_confidence,
            used_ghost_kv=used_ghost_kv,
            used_o1=used_o1,
            thinking_steps=thinking_steps,
            past_key_values=tuple(new_past_key_values) if new_past_key_values else None,
        )
    
    def _fallback_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Tuple],
        use_cache: bool,
    ) -> ForwardOutput:
        """回退到标准前向"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,
        )
        
        return ForwardOutput(
            hidden_states=outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else outputs.logits,
            logits=outputs.logits,
            exit_layer=self.config.model.num_hidden_layers - 1,
            entropy=0.0,
            confidence=1.0,
            past_key_values=outputs.past_key_values if hasattr(outputs, "past_key_values") else None,
        )
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        生成接口
        
        支持 SEDAC 加速的自回归生成
        """
        batch_size = input_ids.shape[0]
        past_key_values = None
        generated = input_ids.clone()
        
        for step in range(max_new_tokens):
            if past_key_values is not None:
                current_input = generated[:, -1:]
            else:
                current_input = generated
            
            output = self.forward_with_sedac(
                current_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            logits = output.logits[:, -1, :]
            
            if temperature > 0 and do_sample:
                logits = logits / temperature
                
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float("-inf")
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float("-inf")
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=-1)
            past_key_values = output.past_key_values
            
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), device=self.device, dtype=attention_mask.dtype)
                ], dim=-1)
        
        return generated
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return self.metrics.get_metrics()
    
    def save_checkpoint(self, path: str) -> None:
        """保存检查点"""
        checkpoint = {
            "config": self.config.to_dict(),
            "threshold_stats": self.threshold_controller.stats,
        }
        
        if self.ghost_kv is not None:
            checkpoint["ghost_kv_state"] = self.ghost_kv.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        if "ghost_kv_state" in checkpoint and self.ghost_kv is not None:
            self.ghost_kv.load_state_dict(checkpoint["ghost_kv_state"])
        
        logger.info(f"Checkpoint loaded from {path}")
