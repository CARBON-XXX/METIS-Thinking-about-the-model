"""
SEDAC V7.4 - Dual-Signal Exit Guard

核心发现:
- entropy_mean 是最强区分特征 (|t| = 44.29)
- High-risk token: 高熵 (3.24) + 高稳定性 (0.97)
- Low-risk token: 低熵 (0.84) + 高稳定性 (0.97)

结论: 仅靠 stability 无法区分，必须结合 entropy

V7.4 策略:
- 退出条件: stability >= tau AND entropy < entropy_threshold
- 阻止条件: stability >= tau BUT entropy >= entropy_threshold (稳定但不确定)
"""

import torch
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum


class ExitDecision(Enum):
    CONTINUE = 0      # 继续计算
    EXIT = 1          # 可以退出
    BLOCKED = 2       # 被阻止退出 (稳定但高熵)


@dataclass
class DualSignalConfig:
    """V7.4 双信号配置"""
    # Stability thresholds
    tau: float = 0.95
    k: int = 3
    min_layer_ratio: float = 0.20
    
    # Entropy thresholds (核心新增)
    entropy_threshold: float = 1.5      # 低于此值才允许退出
    entropy_percentile: float = 0.5     # 或使用 percentile 动态计算
    use_dynamic_entropy: bool = True    # 是否动态计算 entropy 阈值
    
    # Blocking behavior
    block_high_entropy: bool = True     # 是否阻止高熵退出
    max_blocked_layers: int = 5         # 被阻止后最多再计算几层
    
    @classmethod
    def conservative(cls) -> "DualSignalConfig":
        return cls(
            tau=0.97, k=4, min_layer_ratio=0.25,
            entropy_threshold=1.0, entropy_percentile=0.4
        )
    
    @classmethod
    def balanced(cls) -> "DualSignalConfig":
        return cls(
            tau=0.95, k=3, min_layer_ratio=0.20,
            entropy_threshold=1.5, entropy_percentile=0.5
        )
    
    @classmethod
    def aggressive(cls) -> "DualSignalConfig":
        return cls(
            tau=0.92, k=2, min_layer_ratio=0.15,
            entropy_threshold=2.0, entropy_percentile=0.6
        )


class DualSignalExitGuard:
    """
    双信号退出守卫
    
    结合 stability 和 entropy 两个正交信号，
    解决 "稳定但错误" (Confidently Wrong) 的问题
    """
    
    def __init__(
        self,
        config: DualSignalConfig = None,
        device: torch.device = None
    ):
        self.config = config or DualSignalConfig.balanced()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # State
        self.prev_hidden: Optional[torch.Tensor] = None
        self.consecutive_stable: Optional[torch.Tensor] = None
        self.blocked_count: Optional[torch.Tensor] = None
        self.exited_mask: Optional[torch.Tensor] = None
        self.exit_layer: Optional[torch.Tensor] = None
        
        # Statistics
        self.stats = {
            "total_exits": 0,
            "blocked_exits": 0,
            "high_entropy_blocks": 0,
        }
        
        # Dynamic entropy threshold
        self.entropy_history: List[torch.Tensor] = []
        self.dynamic_entropy_threshold: Optional[float] = None
    
    def reset(self, batch_size: int):
        """重置状态"""
        self.prev_hidden = None
        self.consecutive_stable = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.blocked_count = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.exited_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        self.exit_layer = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)
        self.entropy_history = []
        self.dynamic_entropy_threshold = None
    
    def _compute_stability(self, hidden: torch.Tensor) -> torch.Tensor:
        """计算当前层与上一层的稳定性"""
        if self.prev_hidden is None:
            return torch.ones(hidden.shape[0], device=self.device)
        
        cos_sim = torch.nn.functional.cosine_similarity(
            self.prev_hidden.float(), hidden.float(), dim=1
        )
        stability = (cos_sim + 1.0) / 2.0
        return stability
    
    def _update_entropy_threshold(self, entropy: torch.Tensor):
        """动态更新 entropy 阈值"""
        self.entropy_history.append(entropy.detach())
        
        if len(self.entropy_history) >= 3:
            all_entropy = torch.cat(self.entropy_history, dim=0)
            self.dynamic_entropy_threshold = torch.quantile(
                all_entropy, self.config.entropy_percentile
            ).item()
    
    def step(
        self,
        hidden: torch.Tensor,
        entropy: torch.Tensor,
        layer_idx: int,
        total_layers: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        单步判断
        
        Args:
            hidden: [batch_size, hidden_dim]
            entropy: [batch_size] 当前层的熵
            layer_idx: 当前层索引
            total_layers: 总层数
        
        Returns:
            (exit_mask, decision_codes)
            exit_mask: 哪些 token 可以退出
            decision_codes: 0=继续, 1=退出, 2=被阻止
        """
        batch_size = hidden.shape[0]
        
        # Initialize if needed
        if self.consecutive_stable is None:
            self.reset(batch_size)
        
        min_layer = int(total_layers * self.config.min_layer_ratio)
        
        # Compute stability
        stability = self._compute_stability(hidden)
        self.prev_hidden = hidden.detach()
        
        # Update dynamic entropy threshold
        if self.config.use_dynamic_entropy:
            self._update_entropy_threshold(entropy)
        
        # Get entropy threshold
        entropy_thresh = (
            self.dynamic_entropy_threshold 
            if self.dynamic_entropy_threshold is not None 
            else self.config.entropy_threshold
        )
        
        # Update consecutive stable count
        stable_mask = stability >= self.config.tau
        self.consecutive_stable = torch.where(
            stable_mask & ~self.exited_mask,
            self.consecutive_stable + 1,
            torch.zeros_like(self.consecutive_stable)
        )
        
        # Decision logic
        decision = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        if layer_idx >= min_layer:
            # Want to exit: stable enough
            want_exit = (self.consecutive_stable >= self.config.k) & ~self.exited_mask
            
            # Low entropy: can exit
            low_entropy = entropy < entropy_thresh
            
            # High entropy: blocked
            high_entropy = entropy >= entropy_thresh
            
            # Final decisions
            can_exit = want_exit & low_entropy
            blocked = want_exit & high_entropy & self.config.block_high_entropy
            
            # Update blocked count
            self.blocked_count = torch.where(blocked, self.blocked_count + 1, self.blocked_count)
            
            # Force exit if blocked too many times
            force_exit = (self.blocked_count >= self.config.max_blocked_layers) & ~self.exited_mask
            
            # Apply exits
            exit_now = can_exit | force_exit
            self.exited_mask = self.exited_mask | exit_now
            self.exit_layer = torch.where(
                exit_now & (self.exit_layer == -1),
                torch.tensor(layer_idx, device=self.device),
                self.exit_layer
            )
            
            # Set decision codes
            decision = torch.where(exit_now, torch.tensor(1, device=self.device), decision)
            decision = torch.where(blocked & ~exit_now, torch.tensor(2, device=self.device), decision)
            
            # Update stats
            self.stats["total_exits"] += exit_now.sum().item()
            self.stats["blocked_exits"] += blocked.sum().item()
            self.stats["high_entropy_blocks"] += (blocked & high_entropy).sum().item()
        
        return self.exited_mask.clone(), decision
    
    def get_stats(self) -> dict:
        return self.stats.copy()


def evaluate_dual_signal(
    hidden_states: List[torch.Tensor],
    entropies: List[torch.Tensor],
    high_risk_mask: torch.Tensor,
    config: DualSignalConfig,
    device: torch.device
) -> dict:
    """
    评估双信号策略
    
    Returns:
        dict: speedup, risk_rate, blocked_count, etc.
    """
    num_layers = len(hidden_states)
    num_tokens = hidden_states[0].shape[0]
    
    guard = DualSignalExitGuard(config, device)
    guard.reset(num_tokens)
    
    for layer_idx in range(num_layers):
        hidden = hidden_states[layer_idx]
        entropy = entropies[layer_idx] if layer_idx < len(entropies) else torch.zeros(num_tokens, device=device)
        
        exit_mask, decisions = guard.step(hidden, entropy, layer_idx, num_layers)
    
    # Compute metrics
    exited_mask = guard.exited_mask
    exit_layers = guard.exit_layer
    
    exited_count = exited_mask.sum().item()
    
    if exited_count > 0:
        exit_layer_sum = exit_layers[exited_mask].sum().item()
        high_risk_exits = (exited_mask & high_risk_mask).sum().item()
        risk_rate = high_risk_exits / exited_count
    else:
        exit_layer_sum = 0
        risk_rate = 0.0
    
    total_layers_used = exit_layer_sum + (num_tokens - exited_count) * num_layers
    baseline_layers = num_tokens * num_layers
    speedup = baseline_layers / total_layers_used if total_layers_used > 0 else 1.0
    
    return {
        "speedup": speedup,
        "risk_rate": risk_rate,
        "exit_rate": exited_count / num_tokens,
        "blocked_count": guard.stats["blocked_exits"],
        "high_entropy_blocks": guard.stats["high_entropy_blocks"],
        "config": {
            "tau": config.tau,
            "k": config.k,
            "entropy_threshold": config.entropy_threshold,
        }
    }
