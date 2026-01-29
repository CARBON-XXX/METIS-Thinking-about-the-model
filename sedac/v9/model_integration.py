"""
SEDAC V9.0 - 真实模型集成 (Model Integration)

支持将SEDAC注入到真实LLM模型中:
- LLaMA / LLaMA-2 / LLaMA-3
- Qwen / Qwen2
- Mistral
- 其他HuggingFace Transformers模型

用法:
    from sedac.v9.model_integration import inject_sedac
    
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    sedac_model = inject_sedac(model, strategy="safe")
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List, Type
from dataclasses import dataclass
import logging
import copy

logger = logging.getLogger(__name__)

# 尝试导入transformers
try:
    from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig
    from transformers.modeling_outputs import CausalLMOutputWithPast
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available. Install with: pip install transformers")

# 导入SEDAC组件
from sedac.v9.industrial_integrator import (
    IndustrialIntegrator, IndustrialConfig, IntegrationStrategy,
    create_industrial_integrator,
)
from sedac.v9.kv_cache_manager import SkipMode, LayerDecision
from sedac.v9.attention_sinks import create_attention_sink_protector
from sedac.v9.production_layer import ProductionSEDACLayer, LayerConfig


@dataclass
class SEDACModelConfig:
    """SEDAC模型配置"""
    strategy: str = "safe"          # safe, fast, ultimate
    anchor_interval: int = 4
    exit_threshold: float = 0.7
    max_skip_ratio: float = 0.5
    enable_ghost_kv: bool = False
    profile_enabled: bool = False


class SEDACLayerWrapper(nn.Module):
    """
    SEDAC层包装器
    
    包装原始TransformerLayer，注入动态计算逻辑
    """
    
    def __init__(
        self,
        original_layer: nn.Module,
        layer_idx: int,
        sedac_controller: 'SEDACController',
    ):
        super().__init__()
        self.original_layer = original_layer
        self.layer_idx = layer_idx
        self.controller = sedac_controller
        
        # 尝试提取KV投影权重
        self._extract_kv_weights()
    
    def _extract_kv_weights(self):
        """提取K/V投影权重用于KV-Only模式"""
        self.k_proj = None
        self.v_proj = None
        
        # 尝试不同的属性路径
        attn_paths = [
            'self_attn', 'attention', 'attn',
            'self_attention', 'multi_head_attention'
        ]
        
        for path in attn_paths:
            if hasattr(self.original_layer, path):
                attn = getattr(self.original_layer, path)
                
                # LLaMA/Qwen风格
                if hasattr(attn, 'k_proj'):
                    self.k_proj = attn.k_proj
                    self.v_proj = attn.v_proj
                    break
                
                # GPT风格 (fused QKV)
                if hasattr(attn, 'c_attn'):
                    # 需要分割
                    self._fused_qkv = attn.c_attn
                    break
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        """带SEDAC决策的前向传播"""
        
        # 1. 获取SEDAC决策
        decision = self.controller.get_decision(self.layer_idx, hidden_states)
        
        # 2. 根据决策执行
        if decision.skip_mode == SkipMode.FULL_COMPUTE:
            # 完整计算
            return self.original_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
        
        elif decision.skip_mode == SkipMode.KV_ONLY:
            # 只计算KV
            return self._kv_only_forward(
                hidden_states, attention_mask, position_ids,
                past_key_value, use_cache
            )
        
        elif decision.skip_mode == SkipMode.FFN_SKIP:
            # 跳过FFN
            return self._ffn_skip_forward(
                hidden_states, attention_mask, position_ids,
                past_key_value, output_attentions, use_cache, **kwargs
            )
        
        else:  # FULL_SKIP
            # 完全跳过（Ghost KV）
            return self._ghost_forward(
                hidden_states, past_key_value, use_cache
            )
    
    def _kv_only_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        past_key_value: Optional[Tuple],
        use_cache: bool,
    ):
        """KV-Only前向"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        if self.k_proj is not None:
            # 计算K和V
            key = self.k_proj(hidden_states)
            value = self.v_proj(hidden_states)
            
            # 重塑
            num_heads = self.controller.config.num_heads
            head_dim = hidden_size // num_heads
            
            key = key.view(batch_size, seq_len, -1, head_dim).transpose(1, 2)
            value = value.view(batch_size, seq_len, -1, head_dim).transpose(1, 2)
            
            # 拼接历史KV
            if past_key_value is not None:
                key = torch.cat([past_key_value[0], key], dim=2)
                value = torch.cat([past_key_value[1], value], dim=2)
            
            present_kv = (key, value) if use_cache else None
        else:
            # 没有K/V投影，fallback到完整计算
            return self.original_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
        
        # 返回残差
        return (hidden_states, present_kv, None) if use_cache else (hidden_states,)
    
    def _ffn_skip_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        past_key_value: Optional[Tuple],
        output_attentions: bool,
        use_cache: bool,
        **kwargs,
    ):
        """跳过FFN的前向"""
        # 这需要拆解原始层的forward
        # 由于不同模型结构不同，这里使用通用方法
        
        residual = hidden_states
        
        # 尝试访问LayerNorm和Attention
        input_ln = getattr(self.original_layer, 'input_layernorm', None)
        if input_ln is None:
            input_ln = getattr(self.original_layer, 'ln_1', None)
        
        attn = getattr(self.original_layer, 'self_attn', None)
        if attn is None:
            attn = getattr(self.original_layer, 'attention', None)
        
        if input_ln is not None and attn is not None:
            hidden_states = input_ln(hidden_states)
            
            attn_output = attn(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
            
            if isinstance(attn_output, tuple):
                hidden_states = attn_output[0]
                present_kv = attn_output[1] if len(attn_output) > 1 else None
            else:
                hidden_states = attn_output
                present_kv = None
            
            hidden_states = residual + hidden_states
            
            return (hidden_states, present_kv, None) if use_cache else (hidden_states,)
        
        # Fallback
        return self.original_layer(
            residual,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
    
    def _ghost_forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple],
        use_cache: bool,
    ):
        """Ghost KV前向"""
        # 使用Ghost KV生成器
        if self.controller.ghost_manager is not None:
            prev_key = past_key_value[0] if past_key_value else None
            prev_value = past_key_value[1] if past_key_value else None
            
            ghost_key, ghost_value = self.controller.ghost_manager.generate_ghost_kv(
                hidden_states, self.layer_idx, prev_key, prev_value
            )
            
            if past_key_value is not None:
                ghost_key = torch.cat([past_key_value[0], ghost_key], dim=2)
                ghost_value = torch.cat([past_key_value[1], ghost_value], dim=2)
            
            present_kv = (ghost_key, ghost_value) if use_cache else None
        else:
            present_kv = past_key_value
        
        return (hidden_states, present_kv, None) if use_cache else (hidden_states,)


class SEDACController:
    """
    SEDAC控制器
    
    管理所有SEDAC决策逻辑
    """
    
    def __init__(
        self,
        config: SEDACModelConfig,
        model_config: Any,
    ):
        self.config = config
        self.model_config = model_config
        
        # 提取模型参数
        self.num_layers = getattr(model_config, 'num_hidden_layers', 32)
        self.hidden_size = getattr(model_config, 'hidden_size', 4096)
        self.num_heads = getattr(model_config, 'num_attention_heads', 32)
        self.head_dim = self.hidden_size // self.num_heads
        
        # 初始化集成器
        strategy_map = {
            "safe": IntegrationStrategy.SAFE,
            "fast": IntegrationStrategy.FAST,
            "ultimate": IntegrationStrategy.ULTIMATE,
        }
        
        industrial_config = IndustrialConfig(
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            strategy=strategy_map.get(config.strategy, IntegrationStrategy.SAFE),
            anchor_interval=config.anchor_interval,
            exit_threshold=config.exit_threshold,
            max_skip_ratio=config.max_skip_ratio,
        )
        
        self.integrator = IndustrialIntegrator(industrial_config)
        
        # Ghost KV管理器
        self.ghost_manager = self.integrator.ghost_manager
        
        # 锚点保护器
        self.sink_protector = create_attention_sink_protector(
            num_layers=self.num_layers,
            anchor_interval=config.anchor_interval,
        )
        
        # 统计
        self.layer_stats = {i: {"computed": 0, "skipped": 0} for i in range(self.num_layers)}
        
        # 运行时状态
        self._prev_hidden = None
        self._current_step = 0
    
    def get_decision(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
    ) -> LayerDecision:
        """获取层决策"""
        # 检查锚点层
        if self.integrator.is_anchor_layer(layer_idx):
            self.layer_stats[layer_idx]["computed"] += 1
            return LayerDecision(
                layer_idx=layer_idx,
                skip_mode=SkipMode.FULL_COMPUTE,
                confidence=1.0,
                cognitive_load=0.0,
                kv_computed=True,
                computation_saved=0.0,
            )
        
        # 使用集成器决策
        decision = self.integrator.get_layer_decision(layer_idx, hidden_states)
        
        # 更新统计
        if decision.skip_mode == SkipMode.FULL_COMPUTE:
            self.layer_stats[layer_idx]["computed"] += 1
        else:
            self.layer_stats[layer_idx]["skipped"] += 1
        
        self._prev_hidden = hidden_states.detach()
        
        return decision
    
    def reset(self):
        """重置状态"""
        self._prev_hidden = None
        self._current_step = 0
        self.integrator.reset_metrics()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计"""
        total_computed = sum(s["computed"] for s in self.layer_stats.values())
        total_skipped = sum(s["skipped"] for s in self.layer_stats.values())
        total = total_computed + total_skipped
        
        return {
            "total_layers_processed": total,
            "computed_ratio": total_computed / max(total, 1),
            "skipped_ratio": total_skipped / max(total, 1),
            "layer_stats": self.layer_stats,
            "integrator_metrics": self.integrator.get_summary(),
        }


class SEDACModel(nn.Module):
    """
    SEDAC增强的模型包装器
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        sedac_config: SEDACModelConfig,
    ):
        super().__init__()
        self.base_model = base_model
        self.sedac_config = sedac_config
        
        # 获取模型配置
        if hasattr(base_model, 'config'):
            model_config = base_model.config
        else:
            # 创建默认配置
            model_config = type('Config', (), {
                'num_hidden_layers': 32,
                'hidden_size': 4096,
                'num_attention_heads': 32,
            })()
        
        # 创建SEDAC控制器
        self.controller = SEDACController(sedac_config, model_config)
        
        # 包装层
        self._wrap_layers()
    
    def _wrap_layers(self):
        """包装Transformer层"""
        # 查找层容器
        layers = None
        layers_attr = None
        
        # 尝试不同的路径
        paths = [
            ('model.layers', 'model', 'layers'),
            ('transformer.h', 'transformer', 'h'),
            ('layers', None, 'layers'),
            ('h', None, 'h'),
        ]
        
        for full_path, parent_attr, layers_name in paths:
            if parent_attr:
                parent = getattr(self.base_model, parent_attr, None)
                if parent is not None:
                    layers = getattr(parent, layers_name, None)
                    if layers is not None:
                        layers_attr = (parent_attr, layers_name)
                        break
            else:
                layers = getattr(self.base_model, layers_name, None)
                if layers is not None:
                    layers_attr = (None, layers_name)
                    break
        
        if layers is None:
            logger.warning("Could not find transformer layers. SEDAC not injected.")
            return
        
        # 包装每一层
        wrapped_layers = nn.ModuleList([
            SEDACLayerWrapper(layer, idx, self.controller)
            for idx, layer in enumerate(layers)
        ])
        
        # 替换
        parent_attr, layers_name = layers_attr
        if parent_attr:
            parent = getattr(self.base_model, parent_attr)
            setattr(parent, layers_name, wrapped_layers)
        else:
            setattr(self.base_model, layers_name, wrapped_layers)
        
        logger.info(f"Wrapped {len(wrapped_layers)} layers with SEDAC")
    
    def forward(self, *args, **kwargs):
        """前向传播"""
        return self.base_model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """生成"""
        if hasattr(self.base_model, 'generate'):
            return self.base_model.generate(*args, **kwargs)
        raise NotImplementedError("Base model does not support generate()")
    
    def reset_sedac(self):
        """重置SEDAC状态"""
        self.controller.reset()
    
    def get_sedac_stats(self) -> Dict[str, Any]:
        """获取SEDAC统计"""
        return self.controller.get_statistics()


def inject_sedac(
    model: nn.Module,
    strategy: str = "safe",
    anchor_interval: int = 4,
    exit_threshold: float = 0.7,
    **kwargs,
) -> SEDACModel:
    """
    将SEDAC注入到模型中
    
    Args:
        model: HuggingFace模型或其他PyTorch模型
        strategy: "safe", "fast", "ultimate"
        anchor_interval: 锚点层间隔
        exit_threshold: 退出阈值
        
    Returns:
        SEDACModel: 增强后的模型
    """
    config = SEDACModelConfig(
        strategy=strategy,
        anchor_interval=anchor_interval,
        exit_threshold=exit_threshold,
        **kwargs,
    )
    
    return SEDACModel(model, config)


def load_sedac_model(
    model_name_or_path: str,
    strategy: str = "safe",
    device: str = "auto",
    torch_dtype: torch.dtype = torch.float16,
    **kwargs,
) -> SEDACModel:
    """
    加载带SEDAC的模型
    
    Args:
        model_name_or_path: HuggingFace模型路径
        strategy: SEDAC策略
        device: 设备
        torch_dtype: 数据类型
        
    Returns:
        SEDACModel
    """
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers library required")
    
    logger.info(f"Loading model: {model_name_or_path}")
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    
    # 注入SEDAC
    sedac_model = inject_sedac(model, strategy=strategy, **kwargs)
    
    logger.info(f"SEDAC injected with strategy: {strategy}")
    
    return sedac_model


def demo_model_integration():
    """演示模型集成"""
    print("=" * 70)
    print("SEDAC V9.0 Model Integration Demo")
    print("=" * 70)
    
    # 创建一个简单的模拟模型
    class MockTransformerLayer(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.input_layernorm = nn.LayerNorm(hidden_size)
            self.self_attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
            self.post_attention_layernorm = nn.LayerNorm(hidden_size)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
            )
            
            # 添加k_proj和v_proj用于KV-Only模式
            self.self_attn.k_proj = nn.Linear(hidden_size, hidden_size)
            self.self_attn.v_proj = nn.Linear(hidden_size, hidden_size)
        
        def forward(self, hidden_states, **kwargs):
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            attn_output, _ = self.self_attn(hidden_states, hidden_states, hidden_states)
            hidden_states = residual + attn_output
            
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            
            return (hidden_states,)
    
    class MockModel(nn.Module):
        def __init__(self, num_layers=12, hidden_size=512):
            super().__init__()
            self.layers = nn.ModuleList([
                MockTransformerLayer(hidden_size) for _ in range(num_layers)
            ])
            self.config = type('Config', (), {
                'num_hidden_layers': num_layers,
                'hidden_size': hidden_size,
                'num_attention_heads': 8,
            })()
        
        def forward(self, hidden_states):
            for layer in self.layers:
                hidden_states = layer(hidden_states)[0]
            return hidden_states
    
    # 创建模型
    base_model = MockModel(num_layers=12, hidden_size=512)
    
    print(f"\n基础模型: 12层, hidden_size=512")
    
    # 注入SEDAC
    sedac_model = inject_sedac(
        base_model,
        strategy="safe",
        anchor_interval=4,
        exit_threshold=0.6,
    )
    
    print(f"SEDAC策略: safe")
    print(f"锚点间隔: 4")
    
    # 测试推理
    batch_size = 2
    seq_len = 64
    hidden_states = torch.randn(batch_size, seq_len, 512)
    
    print(f"\n测试推理: batch={batch_size}, seq_len={seq_len}")
    
    # 重置统计
    sedac_model.reset_sedac()
    
    # 前向传播
    output = sedac_model(hidden_states)
    
    # 获取统计
    stats = sedac_model.get_sedac_stats()
    
    print(f"\n统计:")
    print(f"  计算比例: {stats['computed_ratio']*100:.1f}%")
    print(f"  跳过比例: {stats['skipped_ratio']*100:.1f}%")
    
    print("\n" + "=" * 70)
    print("Model Integration: SEDAC成功注入模型")
    print("=" * 70)


if __name__ == "__main__":
    demo_model_integration()
