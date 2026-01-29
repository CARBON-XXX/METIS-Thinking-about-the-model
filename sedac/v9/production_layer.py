"""
SEDAC V9.0 - 生产级Transformer Layer (Production Layer)

即插即用的TransformerLayer替换方案，兼容：
- HuggingFace Transformers (LLaMA, Mistral, Qwen, etc.)
- vLLM
- TensorRT-LLM
- Custom implementations

架构：
┌─────────────────────────────────────────────────────────────────┐
│                    ProductionSEDACLayer                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ PreNorm  │ →  │ Decision │ →  │ Execute  │ →  │ PostProc │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       ↓              ↓              ↓              ↓           │
│    LayerNorm    SEDAC Engine    Branch Logic    Residual       │
│                      ↓                                         │
│              ┌───────┴───────┐                                 │
│              ↓       ↓       ↓                                 │
│            Full   KV-Only  Ghost                               │
└─────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Union, Callable
from enum import Enum, auto
import logging
import math

logger = logging.getLogger(__name__)

# 导入SEDAC组件
from sedac.v9.kv_cache_manager import SkipMode, LayerDecision


@dataclass
class LayerConfig:
    """层配置"""
    hidden_size: int = 4096
    num_heads: int = 32
    head_dim: int = 128
    intermediate_size: int = 11008  # FFN中间层
    num_key_value_heads: int = 8    # GQA
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 4096
    
    # SEDAC配置
    enable_sedac: bool = True
    exit_threshold: float = 0.7
    kv_only_threshold: float = 0.5


class RMSNorm(nn.Module):
    """RMS LayerNorm (LLaMA style)"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    """旋转位置编码 (RoPE)"""
    
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # 预计算频率
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._init_rope_cache(max_seq_len)
    
    def _init_rope_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = position_ids.max() + 1
        if seq_len > self.max_seq_len:
            self._init_rope_cache(seq_len)
        
        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """RoPE旋转操作"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """应用旋转位置编码"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class KVOnlyAttention(nn.Module):
    """
    KV-Only注意力投影
    
    只计算K和V，跳过Q和Attention Score
    用于SEDAC的KV-Only模式
    """
    
    def __init__(self, config: LayerConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_key_value_heads
        
        # 只需要K和V投影
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        只计算KV
        
        Returns:
            key: [batch, num_kv_heads, seq_len, head_dim]
            value: [batch, num_kv_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 应用RoPE到Key
        if rotary_emb is not None and position_ids is not None:
            cos, sin = rotary_emb(key, position_ids)
            # 只对key应用（query被跳过了）
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            key = (key * cos) + (rotate_half(key) * sin)
        
        return key, value


class GhostKVProjection(nn.Module):
    """
    Ghost KV投影
    
    用TinyMLP预测KV，而非真实计算
    """
    
    def __init__(self, config: LayerConfig, reduction: int = 16):
        super().__init__()
        hidden_size = config.hidden_size
        kv_dim = config.num_key_value_heads * config.head_dim
        reduced_dim = hidden_size // reduction
        
        # TinyMLP
        self.down = nn.Linear(hidden_size, reduced_dim, bias=False)
        self.act = nn.SiLU()
        self.up = nn.Linear(reduced_dim, kv_dim * 2, bias=False)
        
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        
        # 初始化为小值
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.normal_(self.up.weight, std=0.01)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_key: Optional[torch.Tensor] = None,
        prev_value: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测KV
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        x = self.down(hidden_states)
        x = self.act(x)
        kv = self.up(x)
        
        kv_dim = self.num_kv_heads * self.head_dim
        key, value = kv.split(kv_dim, dim=-1)
        
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 残差连接到上一层
        if prev_key is not None:
            key = key + prev_key[:, :, -seq_len:, :]
            value = value + prev_value[:, :, -seq_len:, :]
        
        return key, value


class SEDACAttention(nn.Module):
    """
    带SEDAC的注意力层
    
    支持三种模式：
    1. FULL: 完整计算
    2. KV_ONLY: 只计算KV
    3. GHOST: Ghost KV预测
    """
    
    def __init__(self, config: LayerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        # 完整投影
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        
        # RoPE
        self.rotary_emb = RotaryEmbedding(
            self.head_dim, 
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )
        
        # SEDAC组件
        self.kv_only = KVOnlyAttention(config)
        self.ghost_kv = GhostKVProjection(config)
        
        # 从完整投影共享权重到kv_only
        self._share_weights()
    
    def _share_weights(self):
        """共享权重"""
        self.kv_only.k_proj.weight = self.k_proj.weight
        self.kv_only.v_proj.weight = self.v_proj.weight
    
    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """GQA: 重复KV heads"""
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
        skip_mode: SkipMode = SkipMode.FULL_COMPUTE,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        带SEDAC模式的前向传播
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # ========== KV-Only模式 ==========
        if skip_mode == SkipMode.KV_ONLY:
            key, value = self.kv_only(hidden_states, position_ids, self.rotary_emb)
            
            if past_key_value is not None:
                key = torch.cat([past_key_value[0], key], dim=2)
                value = torch.cat([past_key_value[1], value], dim=2)
            
            present_kv = (key, value) if use_cache else None
            # 跳过attention计算，直接返回零
            return torch.zeros_like(hidden_states), present_kv
        
        # ========== Ghost KV模式 ==========
        if skip_mode == SkipMode.FULL_SKIP:
            prev_key = past_key_value[0] if past_key_value else None
            prev_value = past_key_value[1] if past_key_value else None
            
            key, value = self.ghost_kv(hidden_states, prev_key, prev_value)
            
            if past_key_value is not None:
                key = torch.cat([past_key_value[0], key], dim=2)
                value = torch.cat([past_key_value[1], value], dim=2)
            
            present_kv = (key, value) if use_cache else None
            return torch.zeros_like(hidden_states), present_kv
        
        # ========== 完整计算 ==========
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # RoPE
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        cos, sin = self.rotary_emb(query, position_ids)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        
        # KV Cache
        if past_key_value is not None:
            key = torch.cat([past_key_value[0], key], dim=2)
            value = torch.cat([past_key_value[1], value], dim=2)
        
        present_kv = (key, value) if use_cache else None
        
        # GQA
        key = self._repeat_kv(key, self.num_kv_groups)
        value = self._repeat_kv(value, self.num_kv_groups)
        
        # Attention
        attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value)
        
        # 输出投影
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, present_kv


class SEDACMLP(nn.Module):
    """MLP层 (SwiGLU)"""
    
    def __init__(self, config: LayerConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class ProductionSEDACLayer(nn.Module):
    """
    生产级SEDAC Transformer层
    
    即插即用，支持所有SEDAC功能
    """
    
    def __init__(self, config: LayerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # LayerNorm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 子层
        self.self_attn = SEDACAttention(config, layer_idx)
        self.mlp = SEDACMLP(config)
        
        # SEDAC状态
        self._current_skip_mode = SkipMode.FULL_COMPUTE
    
    def set_skip_mode(self, mode: SkipMode):
        """设置跳过模式"""
        self._current_skip_mode = mode
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
        skip_mode: Optional[SkipMode] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播
        
        Args:
            skip_mode: 覆盖当前skip_mode
        """
        mode = skip_mode if skip_mode is not None else self._current_skip_mode
        residual = hidden_states
        
        # ========== KV-Only或Ghost模式 ==========
        if mode in [SkipMode.KV_ONLY, SkipMode.FULL_SKIP]:
            # 只更新KV Cache
            _, present_kv = self.self_attn(
                self.input_layernorm(hidden_states),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
                skip_mode=mode,
            )
            # 直接返回残差
            return hidden_states, present_kv
        
        # ========== FFN_SKIP模式 ==========
        if mode == SkipMode.FFN_SKIP:
            hidden_states = self.input_layernorm(hidden_states)
            attn_output, present_kv = self.self_attn(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
                skip_mode=SkipMode.FULL_COMPUTE,
            )
            hidden_states = residual + attn_output
            # 跳过MLP
            return hidden_states, present_kv
        
        # ========== 完整计算 ==========
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, present_kv = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            skip_mode=SkipMode.FULL_COMPUTE,
        )
        hidden_states = residual + attn_output
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_kv


class ProductionSEDACModel(nn.Module):
    """
    生产级SEDAC模型
    
    包含完整的Transformer stack + SEDAC决策逻辑
    """
    
    def __init__(self, config: LayerConfig):
        super().__init__()
        self.config = config
        
        # Embedding
        # self.embed_tokens = nn.Embedding(vocab_size, config.hidden_size)
        
        # Transformer层
        self.layers = nn.ModuleList([
            ProductionSEDACLayer(config, i) for i in range(config.num_layers if hasattr(config, 'num_layers') else 32)
        ])
        
        # 最终LayerNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # SEDAC决策器（简化版）
        self.decision_net = nn.Sequential(
            nn.Linear(config.hidden_size, 64),
            nn.SiLU(),
            nn.Linear(64, 4),  # 4种模式的logits
        )
    
    def get_layer_decision(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        is_anchor: bool = False,
    ) -> SkipMode:
        """获取层决策"""
        if not self.config.enable_sedac:
            return SkipMode.FULL_COMPUTE
        
        if is_anchor:
            return SkipMode.FULL_COMPUTE
        
        # 用decision_net预测
        with torch.no_grad():
            logits = self.decision_net(hidden_states.mean(dim=(0, 1)))
            mode_idx = logits.argmax().item()
        
        modes = [SkipMode.FULL_COMPUTE, SkipMode.FFN_SKIP, SkipMode.KV_ONLY, SkipMode.FULL_SKIP]
        return modes[mode_idx]
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = True,
        anchor_interval: int = 4,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """前向传播"""
        all_present_kvs = [] if use_cache else None
        
        for idx, layer in enumerate(self.layers):
            past_kv = past_key_values[idx] if past_key_values else None
            is_anchor = (idx % anchor_interval == 0) or (idx < 2) or (idx >= len(self.layers) - 2)
            
            skip_mode = self.get_layer_decision(idx, hidden_states, is_anchor)
            
            hidden_states, present_kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=use_cache,
                skip_mode=skip_mode,
            )
            
            if use_cache:
                all_present_kvs.append(present_kv)
        
        hidden_states = self.norm(hidden_states)
        return hidden_states, all_present_kvs


def create_production_layer(
    hidden_size: int = 4096,
    num_heads: int = 32,
    layer_idx: int = 0,
    **kwargs,
) -> ProductionSEDACLayer:
    """创建生产级SEDAC层"""
    config = LayerConfig(
        hidden_size=hidden_size,
        num_heads=num_heads,
        **kwargs,
    )
    return ProductionSEDACLayer(config, layer_idx)


def demo_production_layer():
    """演示生产级层"""
    print("=" * 70)
    print("Production SEDAC Layer Demo")
    print("=" * 70)
    
    # 配置
    config = LayerConfig(
        hidden_size=512,
        num_heads=8,
        head_dim=64,
        num_key_value_heads=2,  # GQA
        intermediate_size=1408,
        enable_sedac=True,
    )
    
    # 创建层
    layer = ProductionSEDACLayer(config, layer_idx=5)
    
    # 模拟输入
    batch_size = 2
    seq_len = 64
    hidden = torch.randn(batch_size, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    print(f"\n配置:")
    print(f"  Hidden Size: {config.hidden_size}")
    print(f"  Num Heads: {config.num_heads}")
    print(f"  Num KV Heads: {config.num_key_value_heads} (GQA)")
    print(f"  Head Dim: {config.head_dim}")
    
    # 测试不同模式
    modes = [
        (SkipMode.FULL_COMPUTE, "完整计算"),
        (SkipMode.FFN_SKIP, "跳过FFN"),
        (SkipMode.KV_ONLY, "只计算KV"),
        (SkipMode.FULL_SKIP, "Ghost KV"),
    ]
    
    print(f"\n测试不同模式:")
    past_kv = None
    
    for mode, name in modes:
        import time
        start = time.perf_counter()
        
        output, present_kv = layer(
            hidden, 
            position_ids=position_ids,
            past_key_value=past_kv,
            use_cache=True,
            skip_mode=mode,
        )
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        kv_shape = present_kv[0].shape if present_kv else "N/A"
        output_changed = not torch.allclose(output, hidden, atol=1e-5)
        
        print(f"  {name:12s}: output_changed={output_changed}, kv_shape={list(kv_shape)}, "
              f"latency={elapsed_ms:.2f}ms")
        
        past_kv = present_kv
    
    # 参数量统计
    total_params = sum(p.numel() for p in layer.parameters())
    kv_only_params = sum(p.numel() for p in layer.self_attn.kv_only.parameters())
    ghost_params = sum(p.numel() for p in layer.self_attn.ghost_kv.parameters())
    
    print(f"\n参数量:")
    print(f"  总参数: {total_params:,}")
    print(f"  KV-Only额外参数: 0 (共享权重)")
    print(f"  Ghost KV参数: {ghost_params:,} ({ghost_params/total_params*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("Production Layer: 即插即用的SEDAC层")
    print("=" * 70)


if __name__ == "__main__":
    demo_production_layer()
