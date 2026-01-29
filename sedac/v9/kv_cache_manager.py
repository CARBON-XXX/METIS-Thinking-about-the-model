"""
SEDAC V9.0 - KV Cache 补偿机制

解决"断层危机"：当跳过中间层时，KV Cache会残缺

方案：只计算KV，跳过FFN和Attention计算
- 不跳过 W_k 和 W_v 的投影计算
- 跳过 W_q 计算
- 跳过 Attention Score (Q @ K^T) 和 Softmax
- 跳过 FFN 层
- 输出走残差连接：X_out = X_in

优点：KV Cache是真实的，后续Token能看到完整信息
缺点：加速比降低到70%-80%（因为还得算投影）
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
from enum import Enum, auto
import logging
import math

logger = logging.getLogger(__name__)


class SkipMode(Enum):
    """跳层模式"""
    FULL_COMPUTE = auto()      # 完整计算（不跳过）
    KV_ONLY = auto()           # 只计算KV，跳过Attention和FFN
    FULL_SKIP = auto()         # 完全跳过（危险，会导致KV Cache断层）
    FFN_SKIP = auto()          # 只跳过FFN，保留Attention


@dataclass
class LayerDecision:
    """层级决策"""
    layer_idx: int
    skip_mode: SkipMode
    confidence: float
    cognitive_load: float
    kv_computed: bool
    computation_saved: float  # 0-1, 节省的计算量比例


@dataclass 
class KVCacheState:
    """KV Cache状态"""
    key: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None
    seq_len: int = 0
    is_valid: bool = True  # 是否完整（无断层）


class KVOnlyProjection(nn.Module):
    """
    只计算KV的投影层
    
    当SEDAC决定跳层时，仍然计算K和V以保持Cache完整
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # K和V的投影矩阵
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        只计算K和V
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            
        Returns:
            key: [batch, num_heads, seq_len, head_dim]
            value: [batch, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 投影
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # 重塑
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        return key, value
    
    @classmethod
    def from_attention_layer(cls, attention_layer) -> "KVOnlyProjection":
        """
        从现有Attention层提取KV投影
        
        支持多种Transformer实现
        """
        # 尝试不同的属性名
        possible_k_names = ["k_proj", "key", "Wk", "key_proj"]
        possible_v_names = ["v_proj", "value", "Wv", "value_proj"]
        
        k_proj = None
        v_proj = None
        
        for name in possible_k_names:
            if hasattr(attention_layer, name):
                k_proj = getattr(attention_layer, name)
                break
        
        for name in possible_v_names:
            if hasattr(attention_layer, name):
                v_proj = getattr(attention_layer, name)
                break
        
        if k_proj is None or v_proj is None:
            raise ValueError("Cannot find K/V projection in attention layer")
        
        # 创建新的投影层
        hidden_size = k_proj.in_features
        out_features = k_proj.out_features
        
        # 推断head配置
        # 通常 out_features = num_heads * head_dim
        # 常见配置：4096 = 32 * 128
        if out_features == 4096:
            num_heads, head_dim = 32, 128
        elif out_features == 2048:
            num_heads, head_dim = 16, 128
        elif out_features == 1024:
            num_heads, head_dim = 8, 128
        else:
            # 默认假设head_dim=128
            head_dim = 128
            num_heads = out_features // head_dim
        
        proj = cls(hidden_size, num_heads, head_dim)
        proj.k_proj.weight.data = k_proj.weight.data.clone()
        proj.v_proj.weight.data = v_proj.weight.data.clone()
        
        return proj


class KVCacheManager:
    """
    KV Cache管理器
    
    负责：
    1. 维护每层的KV Cache
    2. 在跳层时只更新KV
    3. 检测和修复Cache断层
    """
    
    def __init__(
        self,
        num_layers: int,
        max_seq_len: int = 4096,
        device: torch.device = None,
    ):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 每层的Cache状态
        self.caches: List[KVCacheState] = [KVCacheState() for _ in range(num_layers)]
        
        # KV投影层（延迟初始化）
        self.kv_projections: List[Optional[KVOnlyProjection]] = [None] * num_layers
        
        # 统计
        self.total_tokens = 0
        self.skipped_layers = 0
        self.kv_only_layers = 0
    
    def register_projection(self, layer_idx: int, projection: KVOnlyProjection):
        """注册KV投影层"""
        self.kv_projections[layer_idx] = projection
    
    def update_cache(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        is_full_compute: bool = True,
    ):
        """
        更新指定层的Cache
        
        Args:
            layer_idx: 层索引
            key: [batch, num_heads, seq_len, head_dim]
            value: [batch, num_heads, seq_len, head_dim]
            is_full_compute: 是否来自完整计算（vs KV-only）
        """
        cache = self.caches[layer_idx]
        
        if cache.key is None:
            # 首次初始化
            cache.key = key
            cache.value = value
            cache.seq_len = key.shape[2]
        else:
            # 追加
            cache.key = torch.cat([cache.key, key], dim=2)
            cache.value = torch.cat([cache.value, value], dim=2)
            cache.seq_len = cache.key.shape[2]
        
        cache.is_valid = True
        
        # 截断超长序列
        if cache.seq_len > self.max_seq_len:
            excess = cache.seq_len - self.max_seq_len
            cache.key = cache.key[:, :, excess:, :]
            cache.value = cache.value[:, :, excess:, :]
            cache.seq_len = self.max_seq_len
    
    def get_cache(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """获取指定层的Cache"""
        cache = self.caches[layer_idx]
        return cache.key, cache.value
    
    def compute_kv_only(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        只计算KV（跳过Attention和FFN）
        
        Returns:
            key, value
        """
        projection = self.kv_projections[layer_idx]
        
        if projection is None:
            raise ValueError(f"KV projection not registered for layer {layer_idx}")
        
        key, value = projection(hidden_states)
        
        # 更新Cache
        self.update_cache(layer_idx, key, value, is_full_compute=False)
        self.kv_only_layers += 1
        
        return key, value
    
    def process_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        decision: LayerDecision,
        layer_fn: callable = None,
    ) -> torch.Tensor:
        """
        处理单层，根据决策选择计算模式
        
        Args:
            layer_idx: 层索引
            hidden_states: 输入hidden states
            decision: SEDAC的决策
            layer_fn: 完整层计算函数
            
        Returns:
            output hidden states
        """
        self.total_tokens += hidden_states.shape[0] * hidden_states.shape[1]
        
        if decision.skip_mode == SkipMode.FULL_COMPUTE:
            # 完整计算
            if layer_fn is not None:
                output, (key, value) = layer_fn(hidden_states, return_kv=True)
                self.update_cache(layer_idx, key, value, is_full_compute=True)
                return output
            else:
                return hidden_states
        
        elif decision.skip_mode == SkipMode.KV_ONLY:
            # 只计算KV，输出走残差
            self.compute_kv_only(layer_idx, hidden_states)
            self.skipped_layers += 1
            return hidden_states  # 残差连接
        
        elif decision.skip_mode == SkipMode.FFN_SKIP:
            # 只跳过FFN
            if layer_fn is not None:
                output, (key, value) = layer_fn(hidden_states, skip_ffn=True, return_kv=True)
                self.update_cache(layer_idx, key, value, is_full_compute=True)
                return output
            else:
                return hidden_states
        
        else:  # FULL_SKIP
            # 完全跳过（不推荐）
            self.caches[layer_idx].is_valid = False
            self.skipped_layers += 1
            return hidden_states
    
    def check_integrity(self) -> Dict[str, Any]:
        """检查Cache完整性"""
        valid_layers = sum(1 for c in self.caches if c.is_valid)
        invalid_layers = [i for i, c in enumerate(self.caches) if not c.is_valid]
        
        return {
            "total_layers": self.num_layers,
            "valid_layers": valid_layers,
            "invalid_layers": invalid_layers,
            "integrity_score": valid_layers / self.num_layers,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_tokens": self.total_tokens,
            "skipped_layers": self.skipped_layers,
            "kv_only_layers": self.kv_only_layers,
            "skip_ratio": self.skipped_layers / max(self.total_tokens * self.num_layers, 1),
            "integrity": self.check_integrity(),
        }
    
    def reset(self):
        """重置所有Cache"""
        self.caches = [KVCacheState() for _ in range(self.num_layers)]
        self.total_tokens = 0
        self.skipped_layers = 0
        self.kv_only_layers = 0


class AdaptiveLayerScheduler:
    """
    自适应层调度器
    
    结合SEDAC决策和KV Cache管理，实现工业级的动态计算
    """
    
    def __init__(
        self,
        num_layers: int,
        kv_manager: KVCacheManager,
        min_compute_ratio: float = 0.3,  # 最少计算30%的层
    ):
        self.num_layers = num_layers
        self.kv_manager = kv_manager
        self.min_compute_ratio = min_compute_ratio
        
        # 决策历史
        self.decisions: List[LayerDecision] = []
    
    def make_decision(
        self,
        layer_idx: int,
        confidence: float,
        cognitive_load: float,
        is_first_token: bool = False,
    ) -> LayerDecision:
        """
        做出层级决策
        
        Args:
            layer_idx: 当前层
            confidence: SEDAC置信度
            cognitive_load: 认知负荷
            is_first_token: 是否是序列第一个Token
            
        Returns:
            LayerDecision
        """
        layer_progress = layer_idx / (self.num_layers - 1)
        
        # 首个Token必须完整计算（建立KV Cache基础）
        if is_first_token:
            return LayerDecision(
                layer_idx=layer_idx,
                skip_mode=SkipMode.FULL_COMPUTE,
                confidence=confidence,
                cognitive_load=cognitive_load,
                kv_computed=True,
                computation_saved=0.0,
            )
        
        # 保证最小计算比例
        if layer_progress < self.min_compute_ratio:
            return LayerDecision(
                layer_idx=layer_idx,
                skip_mode=SkipMode.FULL_COMPUTE,
                confidence=confidence,
                cognitive_load=cognitive_load,
                kv_computed=True,
                computation_saved=0.0,
            )
        
        # 根据置信度和认知负荷决定
        if confidence > 0.8 and cognitive_load < 0.3:
            # 高置信 + 低负荷 → KV-only模式
            return LayerDecision(
                layer_idx=layer_idx,
                skip_mode=SkipMode.KV_ONLY,
                confidence=confidence,
                cognitive_load=cognitive_load,
                kv_computed=True,
                computation_saved=0.75,  # 节省约75%计算
            )
        elif confidence > 0.6 and cognitive_load < 0.5:
            # 中等置信 → 只跳FFN
            return LayerDecision(
                layer_idx=layer_idx,
                skip_mode=SkipMode.FFN_SKIP,
                confidence=confidence,
                cognitive_load=cognitive_load,
                kv_computed=True,
                computation_saved=0.5,  # 节省约50%计算
            )
        else:
            # 低置信或高负荷 → 完整计算
            return LayerDecision(
                layer_idx=layer_idx,
                skip_mode=SkipMode.FULL_COMPUTE,
                confidence=confidence,
                cognitive_load=cognitive_load,
                kv_computed=True,
                computation_saved=0.0,
            )
    
    def get_summary(self) -> Dict[str, Any]:
        """获取调度摘要"""
        if not self.decisions:
            return {}
        
        mode_counts = {}
        total_saved = 0.0
        
        for d in self.decisions:
            mode = d.skip_mode.name
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            total_saved += d.computation_saved
        
        avg_saved = total_saved / len(self.decisions)
        
        return {
            "total_decisions": len(self.decisions),
            "mode_distribution": mode_counts,
            "average_computation_saved": avg_saved,
            "theoretical_speedup": 1 / (1 - avg_saved) if avg_saved < 1 else float('inf'),
        }


def create_kv_cache_manager(
    num_layers: int = 36,
    max_seq_len: int = 4096,
) -> KVCacheManager:
    """创建KV Cache管理器"""
    return KVCacheManager(num_layers, max_seq_len)


def create_layer_scheduler(
    num_layers: int = 36,
    kv_manager: KVCacheManager = None,
) -> AdaptiveLayerScheduler:
    """创建层调度器"""
    if kv_manager is None:
        kv_manager = create_kv_cache_manager(num_layers)
    return AdaptiveLayerScheduler(num_layers, kv_manager)
