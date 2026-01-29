"""
SEDAC V9.0 - Attention Sinks 保护机制

解决"注意力汇聚点"问题：
- 问题：Transformer极度依赖首个Token和系统提示，错误跳层会导致崩溃
- 方案：定义锚点层+动态掩码，强制保护关键位置

基于StreamingLLM研究：
- 首个Token是"Attention Sink"，必须完整计算
- 系统提示（System Prompt）同样关键
- 锚点层强制全量计算KV

实现：
1. 锚点层定义（每K层强制计算）
2. Attention Sink Token保护
3. 动态Attention Mask修改
4. KV污染检测与修复
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List, Set
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


class ProtectionLevel(Enum):
    """保护级别"""
    NONE = auto()        # 不保护，可跳过
    ANCHOR = auto()      # 锚点层，强制计算
    SINK = auto()        # Attention Sink，绝对不可跳
    CRITICAL = auto()    # 关键Token（如系统提示结束符）


@dataclass
class TokenProtection:
    """Token保护状态"""
    position: int
    level: ProtectionLevel
    reason: str
    layers_computed: Set[int] = field(default_factory=set)
    kv_valid: bool = True


@dataclass
class LayerProtection:
    """层保护状态"""
    layer_idx: int
    is_anchor: bool
    force_compute_positions: Set[int] = field(default_factory=set)


class AttentionSinkDetector:
    """
    Attention Sink检测器
    
    识别哪些Token是"汇聚点"，需要特殊保护
    """
    
    # 系统提示结束标记（常见格式）
    SYSTEM_END_MARKERS = [
        "</s>", "[/INST]", "<|im_end|>", "<|eot_id|>",
        "###", "Human:", "User:", "Assistant:",
    ]
    
    def __init__(
        self,
        num_sink_tokens: int = 4,     # 前N个Token作为Sink
        protect_system_prompt: bool = True,
        protect_newlines: bool = False,  # 是否保护换行符
    ):
        self.num_sink_tokens = num_sink_tokens
        self.protect_system_prompt = protect_system_prompt
        self.protect_newlines = protect_newlines
        
        # 检测到的Sink位置
        self.sink_positions: Set[int] = set()
        self.system_prompt_end: int = -1
    
    def detect_sinks(
        self,
        input_ids: torch.Tensor,  # [batch, seq_len] 或 [seq_len]
        tokenizer: Any = None,
    ) -> List[TokenProtection]:
        """
        检测Attention Sinks
        
        Returns:
            保护列表
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        batch_size, seq_len = input_ids.shape
        protections = []
        
        # 1. 首N个Token作为Sink
        for i in range(min(self.num_sink_tokens, seq_len)):
            protections.append(TokenProtection(
                position=i,
                level=ProtectionLevel.SINK,
                reason=f"Attention Sink (position {i})",
            ))
            self.sink_positions.add(i)
        
        # 2. 检测系统提示结束位置
        if self.protect_system_prompt and tokenizer is not None:
            for marker in self.SYSTEM_END_MARKERS:
                try:
                    marker_ids = tokenizer.encode(marker, add_special_tokens=False)
                    # 在序列中搜索marker
                    for batch_idx in range(batch_size):
                        seq = input_ids[batch_idx].tolist()
                        for i in range(len(seq) - len(marker_ids) + 1):
                            if seq[i:i+len(marker_ids)] == marker_ids:
                                # 找到系统提示结束
                                end_pos = i + len(marker_ids) - 1
                                if end_pos not in self.sink_positions:
                                    protections.append(TokenProtection(
                                        position=end_pos,
                                        level=ProtectionLevel.CRITICAL,
                                        reason=f"System prompt end ({marker})",
                                    ))
                                    self.system_prompt_end = end_pos
                                break
                except:
                    pass
        
        return protections
    
    def is_sink(self, position: int) -> bool:
        """检查位置是否是Sink"""
        return position in self.sink_positions or position < self.num_sink_tokens
    
    def get_protected_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        获取保护mask
        
        Returns:
            [seq_len] bool tensor, True = 受保护
        """
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        
        # Sink Token
        mask[:self.num_sink_tokens] = True
        
        # 额外保护位置
        for pos in self.sink_positions:
            if pos < seq_len:
                mask[pos] = True
        
        # 系统提示结束
        if self.system_prompt_end >= 0 and self.system_prompt_end < seq_len:
            mask[self.system_prompt_end] = True
        
        return mask


class AnchorLayerManager:
    """
    锚点层管理器
    
    定义哪些层是"锚点"，强制全量计算
    """
    
    def __init__(
        self,
        num_layers: int,
        anchor_interval: int = 4,  # 每4层一个锚点
        first_n_anchors: int = 2,  # 前N层强制锚点
        last_n_anchors: int = 2,   # 后N层强制锚点
    ):
        self.num_layers = num_layers
        self.anchor_interval = anchor_interval
        self.first_n_anchors = first_n_anchors
        self.last_n_anchors = last_n_anchors
        
        # 计算锚点层
        self.anchor_layers: Set[int] = set()
        self._compute_anchors()
    
    def _compute_anchors(self):
        """计算锚点层"""
        # 前N层
        for i in range(self.first_n_anchors):
            self.anchor_layers.add(i)
        
        # 后N层
        for i in range(self.num_layers - self.last_n_anchors, self.num_layers):
            self.anchor_layers.add(i)
        
        # 中间按间隔
        for i in range(0, self.num_layers, self.anchor_interval):
            self.anchor_layers.add(i)
    
    def is_anchor(self, layer_idx: int) -> bool:
        """是否是锚点层"""
        return layer_idx in self.anchor_layers
    
    def get_anchor_mask(self, device: torch.device) -> torch.Tensor:
        """
        获取锚点mask
        
        Returns:
            [num_layers] bool tensor
        """
        mask = torch.zeros(self.num_layers, dtype=torch.bool, device=device)
        for idx in self.anchor_layers:
            mask[idx] = True
        return mask
    
    def get_skip_candidates(self) -> List[int]:
        """获取可跳过的层"""
        return [i for i in range(self.num_layers) if i not in self.anchor_layers]


class DynamicAttentionMask:
    """
    动态Attention Mask
    
    当检测到KV污染时，修改Attention Mask让模型忽略脏数据
    """
    
    def __init__(
        self,
        num_layers: int,
        max_seq_len: int = 4096,
    ):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # KV有效性追踪：[layer, seq_pos] -> 是否有效
        self.kv_validity: Dict[Tuple[int, int], bool] = {}
        
        # 污染计数
        self.pollution_count = 0
    
    def mark_valid(self, layer_idx: int, positions: torch.Tensor):
        """标记KV为有效"""
        for pos in positions.tolist():
            self.kv_validity[(layer_idx, pos)] = True
    
    def mark_invalid(self, layer_idx: int, positions: torch.Tensor):
        """标记KV为无效（被跳过）"""
        for pos in positions.tolist():
            self.kv_validity[(layer_idx, pos)] = False
            self.pollution_count += 1
    
    def is_valid(self, layer_idx: int, position: int) -> bool:
        """检查KV是否有效"""
        return self.kv_validity.get((layer_idx, position), True)
    
    def get_attention_mask(
        self,
        layer_idx: int,
        seq_len: int,
        device: torch.device,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        获取动态Attention Mask
        
        Returns:
            [seq_len, seq_len] mask tensor
            True = 可以attend, False = 被mask掉
        """
        # 基础causal mask
        if causal:
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        else:
            mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        
        # 标记无效位置
        for pos in range(seq_len):
            if not self.is_valid(layer_idx, pos):
                # 这个位置的KV被跳过了，mask掉对它的attention
                mask[:, pos] = False
        
        return mask
    
    def find_nearest_valid(
        self,
        layer_idx: int,
        position: int,
        anchor_manager: AnchorLayerManager,
    ) -> int:
        """
        找到最近的有效锚点层
        
        当某层KV无效时，重定向到最近的锚点层
        """
        # 向前搜索
        for l in range(layer_idx, -1, -1):
            if anchor_manager.is_anchor(l) and self.is_valid(l, position):
                return l
        
        # 向后搜索（不太可能，但作为fallback）
        for l in range(layer_idx, self.num_layers):
            if anchor_manager.is_anchor(l) and self.is_valid(l, position):
                return l
        
        return layer_idx  # 没找到，返回原层
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计"""
        total_entries = len(self.kv_validity)
        valid_count = sum(1 for v in self.kv_validity.values() if v)
        
        return {
            "total_entries": total_entries,
            "valid_entries": valid_count,
            "invalid_entries": total_entries - valid_count,
            "pollution_count": self.pollution_count,
            "validity_ratio": valid_count / max(total_entries, 1),
        }
    
    def reset(self):
        """重置"""
        self.kv_validity.clear()
        self.pollution_count = 0


class AttentionSinkProtector:
    """
    Attention Sink保护器
    
    整合所有保护机制
    """
    
    def __init__(
        self,
        num_layers: int,
        max_seq_len: int = 4096,
        anchor_interval: int = 4,
        num_sink_tokens: int = 4,
    ):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # 子组件
        self.sink_detector = AttentionSinkDetector(num_sink_tokens=num_sink_tokens)
        self.anchor_manager = AnchorLayerManager(num_layers, anchor_interval)
        self.dynamic_mask = DynamicAttentionMask(num_layers, max_seq_len)
        
        # Token保护状态
        self.token_protections: Dict[int, TokenProtection] = {}
    
    def initialize(
        self,
        input_ids: torch.Tensor,
        tokenizer: Any = None,
    ):
        """初始化保护状态"""
        # 检测Sinks
        protections = self.sink_detector.detect_sinks(input_ids, tokenizer)
        
        for p in protections:
            self.token_protections[p.position] = p
        
        logger.debug(f"Initialized protection for {len(protections)} tokens")
    
    def should_compute_layer(
        self,
        layer_idx: int,
        position: int,
        confidence: float = 0.0,
    ) -> Tuple[bool, str]:
        """
        决定是否应该计算该层
        
        Returns:
            (should_compute, reason)
        """
        # 锚点层：强制计算
        if self.anchor_manager.is_anchor(layer_idx):
            return True, "Anchor layer"
        
        # Sink Token：强制计算
        if self.sink_detector.is_sink(position):
            return True, "Attention Sink token"
        
        # 受保护Token
        if position in self.token_protections:
            prot = self.token_protections[position]
            if prot.level in [ProtectionLevel.SINK, ProtectionLevel.CRITICAL]:
                return True, prot.reason
        
        # 否则由SEDAC决定
        return False, "SEDAC decision"
    
    def on_layer_computed(
        self,
        layer_idx: int,
        positions: torch.Tensor,
    ):
        """层计算完成回调"""
        self.dynamic_mask.mark_valid(layer_idx, positions)
        
        # 更新Token保护状态
        for pos in positions.tolist():
            if pos in self.token_protections:
                self.token_protections[pos].layers_computed.add(layer_idx)
    
    def on_layer_skipped(
        self,
        layer_idx: int,
        positions: torch.Tensor,
    ):
        """层跳过回调"""
        self.dynamic_mask.mark_invalid(layer_idx, positions)
    
    def get_attention_mask(
        self,
        layer_idx: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """获取该层的Attention Mask"""
        return self.dynamic_mask.get_attention_mask(layer_idx, seq_len, device)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            "num_anchor_layers": len(self.anchor_manager.anchor_layers),
            "num_sink_tokens": len(self.sink_detector.sink_positions),
            "protected_tokens": len(self.token_protections),
            "mask_stats": self.dynamic_mask.get_statistics(),
        }
    
    def reset(self):
        """重置状态"""
        self.token_protections.clear()
        self.dynamic_mask.reset()


def create_attention_sink_protector(
    num_layers: int = 32,
    anchor_interval: int = 4,
    num_sink_tokens: int = 4,
) -> AttentionSinkProtector:
    """创建Attention Sink保护器"""
    return AttentionSinkProtector(
        num_layers=num_layers,
        anchor_interval=anchor_interval,
        num_sink_tokens=num_sink_tokens,
    )


def demo_attention_sinks():
    """演示Attention Sinks保护"""
    print("=" * 60)
    print("Attention Sinks Demo: 关键Token保护机制")
    print("=" * 60)
    
    # 配置
    num_layers = 32
    seq_len = 128
    anchor_interval = 4
    num_sink_tokens = 4
    
    # 创建保护器
    protector = create_attention_sink_protector(
        num_layers=num_layers,
        anchor_interval=anchor_interval,
        num_sink_tokens=num_sink_tokens,
    )
    
    # 模拟输入
    input_ids = torch.randint(0, 32000, (1, seq_len))
    protector.initialize(input_ids)
    
    print(f"\n配置:")
    print(f"  总层数: {num_layers}")
    print(f"  锚点间隔: {anchor_interval}")
    print(f"  Sink Token数: {num_sink_tokens}")
    
    # 显示锚点层
    anchor_layers = sorted(protector.anchor_manager.anchor_layers)
    print(f"\n锚点层 ({len(anchor_layers)}个):")
    print(f"  {anchor_layers}")
    
    # 模拟推理过程
    print(f"\n模拟推理:")
    device = input_ids.device
    
    computed_count = 0
    skipped_count = 0
    
    for layer_idx in range(num_layers):
        # 检查几个关键位置
        test_positions = [0, 1, 2, 3, 10, 50, 100]
        
        for pos in test_positions:
            if pos >= seq_len:
                continue
            
            should_compute, reason = protector.should_compute_layer(layer_idx, pos)
            
            if should_compute:
                computed_count += 1
                protector.on_layer_computed(layer_idx, torch.tensor([pos]))
            else:
                skipped_count += 1
                protector.on_layer_skipped(layer_idx, torch.tensor([pos]))
        
        if layer_idx < 3 or layer_idx >= num_layers - 2:
            is_anchor = "🔒" if protector.anchor_manager.is_anchor(layer_idx) else "  "
            print(f"  Layer {layer_idx:2d} {is_anchor}")
        elif layer_idx == 3:
            print(f"  ...")
    
    # 统计
    stats = protector.get_statistics()
    print(f"\n统计:")
    print(f"  锚点层数: {stats['num_anchor_layers']}")
    print(f"  Sink Token: {stats['num_sink_tokens']}")
    print(f"  受保护Token: {stats['protected_tokens']}")
    print(f"  KV有效率: {stats['mask_stats']['validity_ratio']*100:.1f}%")
    
    # 显示Attention Mask示例
    print(f"\n动态Attention Mask示例 (Layer 5, seq_len=8):")
    mask = protector.get_attention_mask(5, 8, device)
    print(mask.int())
    
    print("\n" + "=" * 60)
    print("Attention Sinks: 保护关键Token，防止长文本崩溃")
    print("=" * 60)


if __name__ == "__main__":
    demo_attention_sinks()
