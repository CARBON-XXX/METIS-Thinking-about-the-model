"""
SEDAC V9.0 - Token Router (工业级动态分组)

解决"Batch平均主义陷阱"：
- 问题：对Batch取平均决定是否跳层，难样本拖死简单样本
- 方案：Token-level Router，每个Token独立决策，动态分组

核心机制：
1. Split: 每层前将Batch分为 Group_Exit 和 Group_Continue
2. Execute: 只对 Group_Continue 执行GPU计算
3. Merge: 层后将两组数据拼回

支持：
- Ragged Tensor（参差张量）
- Continuous Batching（连续批处理）
- Per-Token决策，无"陪跑"
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, NamedTuple
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


class TokenState(Enum):
    """Token状态"""
    ACTIVE = auto()      # 活跃，继续计算
    EXITED = auto()      # 已退出，等待合并
    ANCHORED = auto()    # 锚点Token，强制计算


@dataclass
class TokenMetadata:
    """Token元数据"""
    token_idx: int
    batch_idx: int
    state: TokenState
    exit_layer: int = -1
    confidence: float = 0.0
    cognitive_load: float = 0.0


class RaggedBatch(NamedTuple):
    """
    参差张量批次
    
    支持同一Batch内不同Token处于不同计算深度
    """
    hidden_states: torch.Tensor      # [total_active, hidden_size]
    indices: torch.Tensor            # [total_active] - 原始位置索引
    batch_ids: torch.Tensor          # [total_active] - 属于哪个batch
    seq_positions: torch.Tensor      # [total_active] - 序列内位置
    
    @property
    def total_active(self) -> int:
        return self.hidden_states.shape[0]


@dataclass
class RouterState:
    """Router状态"""
    original_shape: Tuple[int, int, int]  # [batch, seq_len, hidden]
    active_mask: torch.Tensor             # [batch, seq_len] bool
    exit_mask: torch.Tensor               # [batch, seq_len] bool
    exit_hidden: torch.Tensor             # 已退出Token的hidden states
    exit_layers: torch.Tensor             # [batch, seq_len] 退出层号
    confidences: torch.Tensor             # [batch, seq_len] 置信度


class TokenRouter(nn.Module):
    """
    Token级别路由器
    
    实现真正的Per-Token动态计算，解决Batch平均主义问题
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        anchor_interval: int = 4,  # 每4层一个锚点
        min_active_ratio: float = 0.1,  # 最少保持10%活跃
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.anchor_interval = anchor_interval
        self.min_active_ratio = min_active_ratio
        
        # 轻量级Router网络（每个Token独立决策）
        self.router_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.SiLU(),
            nn.Linear(hidden_size // 4, 2),  # [continue_logit, exit_logit]
        )
        
        # 层级偏置（深层更倾向退出）
        self.layer_bias = nn.Parameter(torch.zeros(num_layers))
        
        # 统计
        self.total_tokens = 0
        self.exited_tokens = 0
        self.layer_exit_counts = [0] * num_layers
    
    def _is_anchor_layer(self, layer_idx: int) -> bool:
        """是否是锚点层（强制计算）"""
        return layer_idx % self.anchor_interval == 0
    
    def compute_exit_scores(
        self,
        hidden_states: torch.Tensor,  # [batch, seq_len, hidden] 或 [N, hidden]
        layer_idx: int,
    ) -> torch.Tensor:
        """
        计算每个Token的退出分数
        
        Returns:
            exit_probs: [batch, seq_len] 或 [N] - 退出概率
        """
        original_shape = hidden_states.shape[:-1]
        flat_hidden = hidden_states.view(-1, self.hidden_size)
        
        # Router前向
        logits = self.router_net(flat_hidden)  # [N, 2]
        
        # 加入层级偏置（深层更倾向退出）
        logits[:, 1] += self.layer_bias[layer_idx]
        
        # Softmax得到概率
        probs = F.softmax(logits, dim=-1)
        exit_probs = probs[:, 1]  # 退出概率
        
        return exit_probs.view(*original_shape)
    
    def split_batch(
        self,
        hidden_states: torch.Tensor,  # [batch, seq_len, hidden]
        layer_idx: int,
        state: Optional[RouterState] = None,
        confidence_threshold: float = 0.7,
    ) -> Tuple[RaggedBatch, RouterState]:
        """
        将Batch分割为继续计算组和退出组
        
        Args:
            hidden_states: 输入hidden states
            layer_idx: 当前层
            state: 上一层的Router状态（首层为None）
            confidence_threshold: 退出阈值
            
        Returns:
            (active_batch, updated_state)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        # 首层初始化状态
        if state is None:
            state = RouterState(
                original_shape=(batch_size, seq_len, hidden_size),
                active_mask=torch.ones(batch_size, seq_len, dtype=torch.bool, device=device),
                exit_mask=torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device),
                exit_hidden=torch.zeros(batch_size, seq_len, hidden_size, device=device),
                exit_layers=torch.full((batch_size, seq_len), -1, dtype=torch.long, device=device),
                confidences=torch.zeros(batch_size, seq_len, device=device),
            )
        
        # 锚点层：全部继续
        if self._is_anchor_layer(layer_idx):
            # 只处理当前活跃的Token
            active_indices = state.active_mask.nonzero(as_tuple=False)
            
            if active_indices.shape[0] == 0:
                # 所有Token都已退出
                return RaggedBatch(
                    hidden_states=torch.empty(0, hidden_size, device=device),
                    indices=torch.empty(0, dtype=torch.long, device=device),
                    batch_ids=torch.empty(0, dtype=torch.long, device=device),
                    seq_positions=torch.empty(0, dtype=torch.long, device=device),
                ), state
            
            batch_ids = active_indices[:, 0]
            seq_positions = active_indices[:, 1]
            active_hidden = hidden_states[batch_ids, seq_positions]
            indices = batch_ids * seq_len + seq_positions
            
            return RaggedBatch(
                hidden_states=active_hidden,
                indices=indices,
                batch_ids=batch_ids,
                seq_positions=seq_positions,
            ), state
        
        # 非锚点层：计算退出分数
        exit_probs = self.compute_exit_scores(hidden_states, layer_idx)
        
        # 决定哪些Token退出
        should_exit = (exit_probs > confidence_threshold) & state.active_mask
        
        # 保证最小活跃比例
        total_active = state.active_mask.sum().item()
        num_exiting = should_exit.sum().item()
        max_exits = int(total_active * (1 - self.min_active_ratio))
        
        if num_exiting > max_exits:
            # 限制退出数量：只让置信度最高的退出
            exit_probs_masked = exit_probs.clone()
            exit_probs_masked[~state.active_mask] = -1
            
            flat_probs = exit_probs_masked.view(-1)
            _, top_indices = flat_probs.topk(max_exits)
            
            should_exit = torch.zeros_like(should_exit)
            should_exit.view(-1)[top_indices] = True
            should_exit = should_exit & state.active_mask
        
        # 更新状态
        new_exits = should_exit & ~state.exit_mask
        
        if new_exits.any():
            # 记录退出的hidden states
            state.exit_hidden[new_exits] = hidden_states[new_exits]
            state.exit_layers[new_exits] = layer_idx
            state.confidences[new_exits] = exit_probs[new_exits]
            
            # 更新mask
            state.exit_mask = state.exit_mask | new_exits
            state.active_mask = state.active_mask & ~new_exits
            
            # 统计
            self.exited_tokens += new_exits.sum().item()
            self.layer_exit_counts[layer_idx] += new_exits.sum().item()
        
        self.total_tokens += state.active_mask.numel()
        
        # 提取活跃Token
        active_indices = state.active_mask.nonzero(as_tuple=False)
        
        if active_indices.shape[0] == 0:
            return RaggedBatch(
                hidden_states=torch.empty(0, hidden_size, device=device),
                indices=torch.empty(0, dtype=torch.long, device=device),
                batch_ids=torch.empty(0, dtype=torch.long, device=device),
                seq_positions=torch.empty(0, dtype=torch.long, device=device),
            ), state
        
        batch_ids = active_indices[:, 0]
        seq_positions = active_indices[:, 1]
        active_hidden = hidden_states[batch_ids, seq_positions]
        indices = batch_ids * seq_len + seq_positions
        
        return RaggedBatch(
            hidden_states=active_hidden,
            indices=indices,
            batch_ids=batch_ids,
            seq_positions=seq_positions,
        ), state
    
    def merge_batch(
        self,
        active_batch: RaggedBatch,
        computed_hidden: torch.Tensor,  # [N_active, hidden]
        state: RouterState,
    ) -> torch.Tensor:
        """
        合并计算结果和已退出Token
        
        Returns:
            merged_hidden: [batch, seq_len, hidden]
        """
        batch_size, seq_len, hidden_size = state.original_shape
        device = computed_hidden.device if computed_hidden.numel() > 0 else state.exit_hidden.device
        
        # 初始化输出
        merged = state.exit_hidden.clone()
        
        # 填入活跃Token的计算结果
        if active_batch.total_active > 0:
            merged[active_batch.batch_ids, active_batch.seq_positions] = computed_hidden
        
        return merged
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取路由统计"""
        total = max(self.total_tokens, 1)
        return {
            "total_tokens": self.total_tokens,
            "exited_tokens": self.exited_tokens,
            "exit_ratio": self.exited_tokens / total,
            "layer_exit_distribution": {
                i: count / max(sum(self.layer_exit_counts), 1)
                for i, count in enumerate(self.layer_exit_counts)
                if count > 0
            },
            "theoretical_speedup": total / max(total - self.exited_tokens, 1),
        }
    
    def reset_statistics(self):
        """重置统计"""
        self.total_tokens = 0
        self.exited_tokens = 0
        self.layer_exit_counts = [0] * self.num_layers


class BatchScheduler:
    """
    Batch调度器
    
    实现Continuous Batching，支持动态添加/移除请求
    """
    
    def __init__(
        self,
        max_batch_size: int = 64,
        max_seq_len: int = 4096,
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        
        # 请求队列
        self.pending_requests: List[Dict] = []
        self.active_requests: Dict[int, Dict] = {}
        self.completed_requests: List[Dict] = []
        
        self.request_counter = 0
    
    def add_request(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        priority: int = 0,
    ) -> int:
        """添加新请求"""
        request_id = self.request_counter
        self.request_counter += 1
        
        self.pending_requests.append({
            "id": request_id,
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "priority": priority,
            "generated_tokens": 0,
            "status": "pending",
        })
        
        return request_id
    
    def schedule_batch(self) -> Tuple[List[int], torch.Tensor]:
        """
        调度下一个batch
        
        Returns:
            (request_ids, batched_input_ids)
        """
        # 按优先级排序
        self.pending_requests.sort(key=lambda x: -x["priority"])
        
        # 选择请求
        selected = []
        total_tokens = 0
        
        for req in self.pending_requests[:]:
            seq_len = req["input_ids"].shape[-1]
            if len(selected) < self.max_batch_size and total_tokens + seq_len <= self.max_seq_len * self.max_batch_size:
                selected.append(req)
                self.pending_requests.remove(req)
                total_tokens += seq_len
        
        if not selected:
            return [], None
        
        # Pad到相同长度
        max_len = max(req["input_ids"].shape[-1] for req in selected)
        batched = []
        request_ids = []
        
        for req in selected:
            req_id = req["id"]
            request_ids.append(req_id)
            self.active_requests[req_id] = req
            
            # Pad
            seq_len = req["input_ids"].shape[-1]
            if seq_len < max_len:
                padding = torch.zeros(max_len - seq_len, dtype=req["input_ids"].dtype, device=req["input_ids"].device)
                padded = torch.cat([req["input_ids"].squeeze(0), padding])
            else:
                padded = req["input_ids"].squeeze(0)
            batched.append(padded)
        
        return request_ids, torch.stack(batched)
    
    def complete_request(self, request_id: int, output: torch.Tensor):
        """完成请求"""
        if request_id in self.active_requests:
            req = self.active_requests.pop(request_id)
            req["output"] = output
            req["status"] = "completed"
            self.completed_requests.append(req)
    
    def get_completed(self) -> List[Dict]:
        """获取已完成的请求"""
        completed = self.completed_requests[:]
        self.completed_requests.clear()
        return completed


def create_token_router(
    hidden_size: int = 4096,
    num_layers: int = 32,
    anchor_interval: int = 4,
) -> TokenRouter:
    """创建Token Router"""
    return TokenRouter(
        hidden_size=hidden_size,
        num_layers=num_layers,
        anchor_interval=anchor_interval,
    )


def demo_token_router():
    """演示Token Router"""
    print("=" * 60)
    print("Token Router Demo: Per-Token Dynamic Computation")
    print("=" * 60)
    
    # 配置
    batch_size = 4
    seq_len = 16
    hidden_size = 256
    num_layers = 12
    
    # 创建Router
    router = create_token_router(
        hidden_size=hidden_size,
        num_layers=num_layers,
        anchor_interval=4,
    )
    
    # 模拟输入（不同难度）
    hidden = torch.randn(batch_size, seq_len, hidden_size)
    
    # 让第一个batch的Token更"确定"（低熵，应该早退）
    hidden[0] = hidden[0] * 0.1  # 低方差
    # 让最后一个batch的Token更"困难"（高熵，应该继续）
    hidden[-1] = hidden[-1] * 2.0  # 高方差
    
    print(f"\n输入: batch_size={batch_size}, seq_len={seq_len}")
    print(f"Layer 0 (Anchor): 强制全部计算")
    
    state = None
    
    for layer_idx in range(num_layers):
        active_batch, state = router.split_batch(hidden, layer_idx, state, confidence_threshold=0.6)
        
        # 模拟层计算
        if active_batch.total_active > 0:
            computed = active_batch.hidden_states + torch.randn_like(active_batch.hidden_states) * 0.01
            hidden = router.merge_batch(active_batch, computed, state)
        
        active_ratio = state.active_mask.sum().item() / state.active_mask.numel()
        is_anchor = "🔒" if router._is_anchor_layer(layer_idx) else "  "
        
        print(f"  Layer {layer_idx:2d} {is_anchor}: active={active_batch.total_active:3d}/{batch_size*seq_len}, "
              f"ratio={active_ratio*100:.1f}%")
    
    # 统计
    stats = router.get_statistics()
    print(f"\n统计:")
    print(f"  退出比例: {stats['exit_ratio']*100:.1f}%")
    print(f"  理论加速: {stats['theoretical_speedup']:.2f}x")
    print(f"\n退出层分布:")
    for layer, ratio in stats['layer_exit_distribution'].items():
        bar = "█" * int(ratio * 30)
        print(f"    Layer {layer:2d}: {bar} {ratio*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("Token Router: 每个Token独立决策，无'陪跑'问题")
    print("=" * 60)


if __name__ == "__main__":
    demo_token_router()
