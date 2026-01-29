"""
SEDAC V9.0 - 工业级集成器 (Industrial Integrator)

基于 NVIDIA/OpenAI 生产级标准的完整集成方案

支持三种策略：
- 方案A (Safe): 锚点层 + KV-Only计算 - 稳定性优先
- 方案B (Fast): Ghost KV预测 - 性能优先  
- 方案C (Ultimate): Per-Token混合策略 - 极致优化

架构：
┌─────────────────────────────────────────────────────────────┐
│                    SEDAC V9.0 Industrial                     │
├─────────────────────────────────────────────────────────────┤
│  Input → TokenRouter → LayerDecision → Execution → Output   │
│              ↓              ↓              ↓                │
│         Split Batch    SEDAC Engine   KV Strategy           │
│              ↓              ↓              ↓                │
│         Active/Exit    Confidence    Full/KV-Only/Ghost     │
│                              ↓                              │
│                      AttentionSinks                         │
│                      (Safety Net)                           │
└─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List, Callable, Union
from enum import Enum, auto
import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# 导入 SEDAC V9.0 组件
from sedac.v9.kv_cache_manager import (
    KVCacheManager, KVOnlyProjection, AdaptiveLayerScheduler,
    SkipMode, LayerDecision, create_kv_cache_manager, create_layer_scheduler,
)
from sedac.v9.ghost_kv import (
    GhostKVGenerator, GhostKVConfig, GhostKVManager,
    create_ghost_kv_manager,
)
from sedac.v9.token_router import (
    TokenRouter, TokenState, RaggedBatch, RouterState,
    create_token_router,
)
from sedac.v9.attention_sinks import (
    AttentionSinkProtector, AnchorLayerManager, ProtectionLevel,
    create_attention_sink_protector,
)
from sedac.v9.fused_gpu_kernel import (
    FusedSEDACEngine, create_fused_engine,
)


class IntegrationStrategy(Enum):
    """集成策略"""
    SAFE = auto()       # 方案A: 锚点层 + KV-Only (稳定性优先)
    FAST = auto()       # 方案B: Ghost KV预测 (性能优先)
    ULTIMATE = auto()   # 方案C: Per-Token混合 (极致优化)
    ADAPTIVE = auto()   # 自适应：根据负载自动选择


@dataclass
class IndustrialConfig:
    """工业级配置"""
    # 模型参数
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    vocab_size: int = 32000
    max_seq_len: int = 4096
    
    # 策略配置
    strategy: IntegrationStrategy = IntegrationStrategy.SAFE
    anchor_interval: int = 4          # 锚点层间隔
    num_sink_tokens: int = 4          # Attention Sink数量
    
    # 决策阈值
    exit_threshold: float = 0.7       # 退出阈值
    kv_only_threshold: float = 0.5    # KV-Only阈值
    ghost_threshold: float = 0.3      # Ghost KV阈值
    
    # 性能配置
    use_cuda_graphs: bool = True      # CUDA Graph加速
    use_triton: bool = True           # Triton算子
    profile_enabled: bool = False     # 性能分析
    
    # 安全配置
    max_skip_ratio: float = 0.6       # 最大跳层比例
    min_compute_layers: int = 8       # 最少计算层数
    force_first_n: int = 2            # 强制计算前N层
    force_last_n: int = 2             # 强制计算后N层


@dataclass 
class LayerOutput:
    """层输出"""
    hidden_states: torch.Tensor
    key: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None
    skip_mode: SkipMode = SkipMode.FULL_COMPUTE
    confidence: float = 0.0
    latency_ms: float = 0.0


@dataclass
class InferenceMetrics:
    """推理指标"""
    total_layers: int = 0
    computed_layers: int = 0
    kv_only_layers: int = 0
    ghost_layers: int = 0
    skipped_layers: int = 0
    total_latency_ms: float = 0.0
    layer_latencies: List[float] = field(default_factory=list)
    
    @property
    def skip_ratio(self) -> float:
        if self.total_layers == 0:
            return 0.0
        return (self.kv_only_layers + self.ghost_layers + self.skipped_layers) / self.total_layers
    
    @property
    def theoretical_speedup(self) -> float:
        if self.computed_layers == 0:
            return 1.0
        return self.total_layers / self.computed_layers
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_layers": self.total_layers,
            "computed_layers": self.computed_layers,
            "kv_only_layers": self.kv_only_layers,
            "ghost_layers": self.ghost_layers,
            "skipped_layers": self.skipped_layers,
            "skip_ratio": f"{self.skip_ratio*100:.1f}%",
            "theoretical_speedup": f"{self.theoretical_speedup:.2f}x",
            "total_latency_ms": f"{self.total_latency_ms:.2f}ms",
            "avg_layer_latency_ms": f"{sum(self.layer_latencies)/max(len(self.layer_latencies),1):.3f}ms",
        }


class SEDACLayerWrapper(nn.Module):
    """
    SEDAC层包装器
    
    包装原始TransformerLayer，注入SEDAC决策逻辑
    """
    
    def __init__(
        self,
        original_layer: nn.Module,
        layer_idx: int,
        integrator: 'IndustrialIntegrator',
    ):
        super().__init__()
        self.original_layer = original_layer
        self.layer_idx = layer_idx
        self.integrator = integrator
        
        # 提取层组件（假设标准Transformer结构）
        self._extract_components()
    
    def _extract_components(self):
        """提取层组件"""
        # 尝试常见的属性名
        self.self_attn = getattr(self.original_layer, 'self_attn', None)
        self.mlp = getattr(self.original_layer, 'mlp', None)
        self.ffn = getattr(self.original_layer, 'ffn', self.mlp)
        self.input_layernorm = getattr(self.original_layer, 'input_layernorm', None)
        self.post_attention_layernorm = getattr(self.original_layer, 'post_attention_layernorm', None)
        
        # LLaMA风格
        if self.self_attn is None:
            self.self_attn = getattr(self.original_layer, 'attention', None)
        
        # GPT风格
        if self.self_attn is None:
            self.self_attn = getattr(self.original_layer, 'attn', None)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        带SEDAC决策的前向传播
        """
        start_time = time.perf_counter()
        
        # 1. 获取SEDAC决策
        decision = self.integrator.get_layer_decision(
            self.layer_idx, hidden_states
        )
        
        # 2. 检查锚点层
        if self.integrator.is_anchor_layer(self.layer_idx):
            decision.skip_mode = SkipMode.FULL_COMPUTE
            decision.reason = "Anchor layer (forced)"
        
        # 3. 执行分支
        if decision.skip_mode == SkipMode.FULL_COMPUTE:
            # 完整计算
            output, present_kv = self._full_compute(
                hidden_states, attention_mask, position_ids, past_key_value, use_cache, **kwargs
            )
            self.integrator.metrics.computed_layers += 1
            
        elif decision.skip_mode == SkipMode.KV_ONLY:
            # 只计算KV
            output, present_kv = self._kv_only_compute(
                hidden_states, attention_mask, position_ids, past_key_value, **kwargs
            )
            self.integrator.metrics.kv_only_layers += 1
            
        elif decision.skip_mode == SkipMode.FFN_SKIP:
            # 跳过FFN
            output, present_kv = self._ffn_skip_compute(
                hidden_states, attention_mask, position_ids, past_key_value, use_cache, **kwargs
            )
            self.integrator.metrics.kv_only_layers += 1
            
        else:  # FULL_SKIP
            # Ghost KV或完全跳过
            output, present_kv = self._ghost_or_skip(
                hidden_states, past_key_value, **kwargs
            )
            self.integrator.metrics.ghost_layers += 1
        
        self.integrator.metrics.total_layers += 1
        
        # 记录延迟
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.integrator.metrics.layer_latencies.append(latency_ms)
        self.integrator.metrics.total_latency_ms += latency_ms
        
        return output, present_kv
    
    def _full_compute(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        past_key_value: Optional[Tuple],
        use_cache: bool,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """完整计算"""
        # 调用原始层
        outputs = self.original_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            **kwargs,
        )
        
        if isinstance(outputs, tuple):
            return outputs[0], outputs[1] if len(outputs) > 1 else None
        return outputs, None
    
    def _kv_only_compute(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        past_key_value: Optional[Tuple],
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        只计算KV，跳过Attention Score和FFN
        
        这是方案A的核心：保持KV Cache连续性
        """
        # 使用KV-Only投影
        key, value = self.integrator.kv_projections[self.layer_idx](hidden_states)
        
        # 更新KV Cache
        if past_key_value is not None:
            key = torch.cat([past_key_value[0], key], dim=2)
            value = torch.cat([past_key_value[1], value], dim=2)
        
        present_kv = (key, value)
        
        # 直接返回残差（跳过这一层的计算）
        return hidden_states, present_kv
    
    def _ffn_skip_compute(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        past_key_value: Optional[Tuple],
        use_cache: bool,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        只跳过FFN，执行Self-Attention
        """
        residual = hidden_states
        
        # LayerNorm
        if self.input_layernorm is not None:
            hidden_states = self.input_layernorm(hidden_states)
        
        # Self-Attention
        if self.self_attn is not None:
            attn_output = self.self_attn(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
                **kwargs,
            )
            
            if isinstance(attn_output, tuple):
                attn_output, present_kv = attn_output[0], attn_output[1] if len(attn_output) > 1 else None
            else:
                present_kv = None
            
            # 残差连接
            hidden_states = residual + attn_output
        else:
            present_kv = None
        
        # 跳过FFN，直接返回
        return hidden_states, present_kv
    
    def _ghost_or_skip(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple],
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        使用Ghost KV或完全跳过
        
        这是方案B的核心：TinyMLP预测KV
        """
        strategy = self.integrator.config.strategy
        
        if strategy in [IntegrationStrategy.FAST, IntegrationStrategy.ULTIMATE]:
            # 使用Ghost KV
            prev_key = past_key_value[0] if past_key_value else None
            prev_value = past_key_value[1] if past_key_value else None
            
            ghost_key, ghost_value = self.integrator.ghost_manager.generate_ghost_kv(
                hidden_states, self.layer_idx, prev_key, prev_value
            )
            
            # 拼接历史KV
            if past_key_value is not None:
                ghost_key = torch.cat([past_key_value[0], ghost_key], dim=2)
                ghost_value = torch.cat([past_key_value[1], ghost_value], dim=2)
            
            present_kv = (ghost_key, ghost_value)
        else:
            # 完全跳过（复用上一层KV）
            present_kv = past_key_value
        
        # 返回残差
        return hidden_states, present_kv


class IndustrialIntegrator:
    """
    SEDAC V9.0 工业级集成器
    
    整合所有组件，提供生产级接口
    """
    
    def __init__(self, config: IndustrialConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化组件
        self._init_components()
        
        # 推理指标
        self.metrics = InferenceMetrics()
        
        # 状态
        self._prev_hidden: Optional[torch.Tensor] = None
        self._layer_decisions: Dict[int, LayerDecision] = {}
        
        logger.info(f"IndustrialIntegrator initialized with strategy: {config.strategy.name}")
    
    def _init_components(self):
        """初始化所有SEDAC组件"""
        cfg = self.config
        
        # 1. KV Cache Manager (状态管理)
        self.kv_manager = create_kv_cache_manager(
            num_layers=cfg.num_layers,
        )
        
        # 1.1 KV-Only Projections (每层一个)
        from sedac.v9.kv_cache_manager import KVOnlyProjection
        self.kv_projections = nn.ModuleList([
            KVOnlyProjection(cfg.hidden_size, cfg.num_heads, cfg.head_dim)
            for _ in range(cfg.num_layers)
        ])
        
        # 2. Layer Scheduler
        self.scheduler = create_layer_scheduler(
            num_layers=cfg.num_layers,
        )
        
        # 3. Ghost KV Manager (方案B/C)
        if cfg.strategy in [IntegrationStrategy.FAST, IntegrationStrategy.ULTIMATE, IntegrationStrategy.ADAPTIVE]:
            self.ghost_manager = create_ghost_kv_manager(
                hidden_size=cfg.hidden_size,
                num_heads=cfg.num_heads,
                head_dim=cfg.head_dim,
                num_layers=cfg.num_layers,
                strategy="ghost",
            )
        else:
            self.ghost_manager = None
        
        # 4. Token Router (方案C)
        if cfg.strategy in [IntegrationStrategy.ULTIMATE, IntegrationStrategy.ADAPTIVE]:
            self.token_router = create_token_router(
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                anchor_interval=cfg.anchor_interval,
            )
        else:
            self.token_router = None
        
        # 5. Attention Sink Protector
        self.sink_protector = create_attention_sink_protector(
            num_layers=cfg.num_layers,
            anchor_interval=cfg.anchor_interval,
            num_sink_tokens=cfg.num_sink_tokens,
        )
        
        # 6. Fused GPU Engine
        if cfg.use_triton:
            self.fused_engine = create_fused_engine(
                vocab_size=cfg.vocab_size,
                hidden_size=cfg.hidden_size,
            )
        else:
            self.fused_engine = None
    
    def is_anchor_layer(self, layer_idx: int) -> bool:
        """是否是锚点层"""
        cfg = self.config
        
        # 强制计算前N层
        if layer_idx < cfg.force_first_n:
            return True
        
        # 强制计算后N层
        if layer_idx >= cfg.num_layers - cfg.force_last_n:
            return True
        
        # 锚点层
        return self.sink_protector.anchor_manager.is_anchor(layer_idx)
    
    def get_layer_decision(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
    ) -> LayerDecision:
        """
        获取层决策
        
        基于SEDAC引擎的智能决策
        """
        cfg = self.config
        
        # 计算置信度和认知负荷
        if self.fused_engine is not None and logits is not None:
            # 使用Fused GPU Engine（零CPU同步）
            entropy, confidence, exit_mask, cognitive_load = self.fused_engine.fused_decision(
                logits, hidden_states, 
                self._prev_hidden if self._prev_hidden is not None else hidden_states,
                layer_idx, cfg.num_layers, cfg.exit_threshold,
            )
            avg_confidence = confidence.mean().item()
            avg_cognitive = cognitive_load.mean().item()
        else:
            # 简化计算
            avg_confidence = self._estimate_confidence(hidden_states)
            avg_cognitive = 1.0 - avg_confidence
        
        # 层进度
        layer_progress = layer_idx / (cfg.num_layers - 1)
        
        # 决策逻辑
        decision = self.scheduler.make_decision(layer_idx, avg_confidence, avg_cognitive)
        
        # 应用策略约束
        decision = self._apply_strategy_constraints(decision, layer_idx, avg_confidence)
        
        # 缓存
        self._layer_decisions[layer_idx] = decision
        self._prev_hidden = hidden_states.detach()
        
        return decision
    
    def _estimate_confidence(self, hidden_states: torch.Tensor) -> float:
        """简化的置信度估计"""
        # 基于hidden states的方差
        var = hidden_states.var().item()
        # 低方差 = 高置信度
        confidence = 1.0 / (1.0 + var)
        return min(max(confidence, 0.0), 1.0)
    
    def _apply_strategy_constraints(
        self,
        decision: LayerDecision,
        layer_idx: int,
        confidence: float,
    ) -> LayerDecision:
        """应用策略约束"""
        cfg = self.config
        
        # 检查最大跳层比例
        current_skip_ratio = self.metrics.skip_ratio
        if current_skip_ratio >= cfg.max_skip_ratio:
            decision.skip_mode = SkipMode.FULL_COMPUTE
            decision.reason = "Max skip ratio reached"
            return decision
        
        # 检查最少计算层数
        if self.metrics.computed_layers < cfg.min_compute_layers:
            remaining_layers = cfg.num_layers - layer_idx
            needed_computes = cfg.min_compute_layers - self.metrics.computed_layers
            if remaining_layers <= needed_computes:
                decision.skip_mode = SkipMode.FULL_COMPUTE
                decision.reason = "Min compute layers constraint"
                return decision
        
        # 根据策略调整
        if cfg.strategy == IntegrationStrategy.SAFE:
            # 保守：只允许KV-Only，不允许完全跳过
            if decision.skip_mode == SkipMode.FULL_SKIP:
                decision.skip_mode = SkipMode.KV_ONLY
                decision.reason = "Safe mode: downgrade to KV-Only"
        
        elif cfg.strategy == IntegrationStrategy.FAST:
            # 激进：低置信度直接Ghost KV
            if confidence < cfg.ghost_threshold and decision.skip_mode != SkipMode.FULL_COMPUTE:
                decision.skip_mode = SkipMode.FULL_SKIP  # 使用Ghost KV
                decision.reason = "Fast mode: use Ghost KV"
        
        return decision
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """
        包装整个模型
        
        自动识别并替换TransformerLayer
        """
        # 查找layer容器
        layers = None
        for name in ['layers', 'h', 'blocks', 'decoder_layers']:
            if hasattr(model, name):
                layers = getattr(model, name)
                break
        
        if layers is None:
            # 尝试在model.model中查找
            if hasattr(model, 'model'):
                for name in ['layers', 'h', 'blocks']:
                    if hasattr(model.model, name):
                        layers = getattr(model.model, name)
                        break
        
        if layers is None:
            logger.warning("Could not find transformer layers. Model not wrapped.")
            return model
        
        # 包装每一层
        wrapped_layers = nn.ModuleList([
            SEDACLayerWrapper(layer, idx, self)
            for idx, layer in enumerate(layers)
        ])
        
        # 替换
        if hasattr(model, 'layers'):
            model.layers = wrapped_layers
        elif hasattr(model, 'h'):
            model.h = wrapped_layers
        elif hasattr(model, 'blocks'):
            model.blocks = wrapped_layers
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            model.model.layers = wrapped_layers
        
        logger.info(f"Wrapped {len(wrapped_layers)} transformer layers with SEDAC")
        return model
    
    def reset_metrics(self):
        """重置指标"""
        self.metrics = InferenceMetrics()
        self._prev_hidden = None
        self._layer_decisions.clear()
    
    @contextmanager
    def inference_context(self):
        """推理上下文管理器"""
        self.reset_metrics()
        try:
            yield self
        finally:
            pass  # 可以在这里添加清理逻辑
    
    def get_summary(self) -> Dict[str, Any]:
        """获取推理摘要"""
        return {
            "config": {
                "strategy": self.config.strategy.name,
                "num_layers": self.config.num_layers,
                "anchor_interval": self.config.anchor_interval,
                "exit_threshold": self.config.exit_threshold,
            },
            "metrics": self.metrics.to_dict(),
            "components": {
                "kv_manager": True,
                "ghost_manager": self.ghost_manager is not None,
                "token_router": self.token_router is not None,
                "fused_engine": self.fused_engine is not None,
            },
        }


class PerTokenIntegrator(IndustrialIntegrator):
    """
    方案C：Per-Token级别的混合策略集成器
    
    每个Token独立决策，支持Ragged Tensor
    """
    
    def process_batch_per_token(
        self,
        hidden_states: torch.Tensor,  # [batch, seq_len, hidden]
        layer_idx: int,
        router_state: Optional[RouterState] = None,
    ) -> Tuple[torch.Tensor, RouterState]:
        """
        Per-Token处理
        
        Returns:
            (output_hidden, updated_state)
        """
        if self.token_router is None:
            raise ValueError("Token router not initialized. Use ULTIMATE or ADAPTIVE strategy.")
        
        # 1. Router分割
        active_batch, state = self.token_router.split_batch(
            hidden_states, layer_idx, router_state, 
            confidence_threshold=self.config.exit_threshold,
        )
        
        # 2. 对Active Token执行完整计算
        if active_batch.total_active > 0:
            # 提取active hidden states
            active_hidden = active_batch.hidden_states
            
            # 这里应该调用实际的layer计算
            # computed_hidden = layer(active_hidden)
            # 简化：模拟计算
            computed_hidden = active_hidden + torch.randn_like(active_hidden) * 0.01
        else:
            computed_hidden = torch.empty(0, hidden_states.shape[-1], device=hidden_states.device)
        
        # 3. 对Exit Token只计算KV（或Ghost KV）
        if state.exit_mask.any():
            exit_positions = state.exit_mask.nonzero(as_tuple=False)
            exit_hidden = hidden_states[exit_positions[:, 0], exit_positions[:, 1]]
            
            # KV-Only或Ghost KV
            if self.config.strategy == IntegrationStrategy.ULTIMATE and self.ghost_manager is not None:
                # Ghost KV
                self.ghost_manager.generate_ghost_kv(
                    exit_hidden.unsqueeze(0), layer_idx
                )
            else:
                # KV-Only
                self.kv_manager.compute_kv_only(layer_idx, exit_hidden.unsqueeze(0))
        
        # 4. 合并结果
        merged = self.token_router.merge_batch(active_batch, computed_hidden, state)
        
        return merged, state


def create_industrial_integrator(
    strategy: str = "safe",
    hidden_size: int = 4096,
    num_layers: int = 32,
    **kwargs,
) -> IndustrialIntegrator:
    """
    创建工业级集成器
    
    Args:
        strategy: "safe", "fast", "ultimate", "adaptive"
        hidden_size: 隐藏层大小
        num_layers: 层数
        **kwargs: 其他配置
    """
    strategy_map = {
        "safe": IntegrationStrategy.SAFE,
        "fast": IntegrationStrategy.FAST,
        "ultimate": IntegrationStrategy.ULTIMATE,
        "adaptive": IntegrationStrategy.ADAPTIVE,
    }
    
    config = IndustrialConfig(
        strategy=strategy_map.get(strategy.lower(), IntegrationStrategy.SAFE),
        hidden_size=hidden_size,
        num_layers=num_layers,
        **kwargs,
    )
    
    if config.strategy == IntegrationStrategy.ULTIMATE:
        return PerTokenIntegrator(config)
    return IndustrialIntegrator(config)


def demo_industrial_integrator():
    """演示工业级集成器"""
    print("=" * 70)
    print("SEDAC V9.0 Industrial Integrator Demo")
    print("=" * 70)
    
    # 测试三种策略
    strategies = ["safe", "fast", "ultimate"]
    
    for strategy in strategies:
        print(f"\n{'='*30} Strategy: {strategy.upper()} {'='*30}")
        
        # 创建集成器
        integrator = create_industrial_integrator(
            strategy=strategy,
            hidden_size=512,
            num_layers=12,
            num_heads=8,
            head_dim=64,
            anchor_interval=4,
        )
        
        # 模拟推理
        with integrator.inference_context():
            # 模拟12层的决策
            hidden = torch.randn(2, 64, 512)
            
            for layer_idx in range(12):
                decision = integrator.get_layer_decision(layer_idx, hidden)
                
                # 模拟指标更新
                if decision.skip_mode == SkipMode.FULL_COMPUTE:
                    integrator.metrics.computed_layers += 1
                elif decision.skip_mode == SkipMode.KV_ONLY:
                    integrator.metrics.kv_only_layers += 1
                else:
                    integrator.metrics.ghost_layers += 1
                integrator.metrics.total_layers += 1
                
                is_anchor = "🔒" if integrator.is_anchor_layer(layer_idx) else "  "
                print(f"  Layer {layer_idx:2d} {is_anchor}: {decision.skip_mode.name:12s} "
                      f"(conf={decision.confidence:.2f})")
                
                # 模拟hidden更新
                hidden = hidden + torch.randn_like(hidden) * 0.05
            
            # 输出摘要
            summary = integrator.get_summary()
            print(f"\n  摘要:")
            for key, value in summary["metrics"].items():
                print(f"    {key}: {value}")
    
    print("\n" + "=" * 70)
    print("Industrial Integrator: 生产级SEDAC集成方案")
    print("=" * 70)


if __name__ == "__main__":
    demo_industrial_integrator()
