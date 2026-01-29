"""
SEDAC V8.0 - Intuition Network (直觉网络)

可训练的门控网络，学习"直觉"而非手工阈值

输入 (Sensory Inputs):
- Entropy distribution (不仅是均值)
- Stability (cosine similarity)
- Hidden State L2 Norm
- Attention Head Sparsity (if available)
- Layer-wise KL Divergence

输出 (Intuition Signal):
- p_confident: 掌控感 [0, 1]
- p_hallucination: 幻觉风险 [0, 1]
- p_ood: 超纲检测 [0, 1]

架构: 轻量级 MLP (2层, 128 hidden units)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum
import math


@dataclass
class IntuitionConfig:
    """直觉网络配置"""
    # 输入特征维度
    num_features: int = 8
    
    # 网络架构
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    
    # 输出头
    num_outputs: int = 3  # confident, hallucination, ood
    
    # 温度参数 (用于校准)
    temperature: float = 1.0
    
    # 决策阈值
    confident_threshold: float = 0.7
    hallucination_threshold: float = 0.5
    ood_threshold: float = 0.6
    
    # 训练参数
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Loss 权重
    lambda_speed: float = 1.0
    lambda_accuracy: float = 2.0
    lambda_calibration: float = 0.5


@dataclass
class IntuitionSignal:
    """直觉信号输出"""
    p_confident: torch.Tensor      # 掌控感 [batch_size]
    p_hallucination: torch.Tensor  # 幻觉风险 [batch_size]
    p_ood: torch.Tensor            # 超纲检测 [batch_size]
    
    # 原始 logits (用于训练)
    logits: Optional[torch.Tensor] = None
    
    # 辅助信息
    features: Optional[torch.Tensor] = None
    layer_idx: int = 0


class SensoryEncoder(nn.Module):
    """
    感知器编码器
    
    将原始特征编码为直觉网络可理解的表示
    """
    
    def __init__(self, config: IntuitionConfig):
        super().__init__()
        self.config = config
        
        # 特征归一化
        self.layer_norm = nn.LayerNorm(config.num_features)
        
        # 位置编码 (用于区分不同层)
        self.max_layers = 128
        self.layer_embedding = nn.Embedding(self.max_layers, config.hidden_dim // 4)
        
        # 特征投影
        self.feature_proj = nn.Linear(config.num_features, config.hidden_dim * 3 // 4)
    
    def forward(
        self, 
        features: torch.Tensor,  # [batch_size, num_features]
        layer_idx: int
    ) -> torch.Tensor:
        # 归一化
        x = self.layer_norm(features)
        
        # 特征投影
        x = self.feature_proj(x)  # [batch_size, hidden_dim * 3/4]
        
        # 层位置编码
        layer_idx_tensor = torch.tensor([layer_idx], device=features.device)
        layer_emb = self.layer_embedding(layer_idx_tensor)  # [1, hidden_dim/4]
        layer_emb = layer_emb.expand(features.shape[0], -1)  # [batch_size, hidden_dim/4]
        
        # 拼接
        x = torch.cat([x, layer_emb], dim=-1)  # [batch_size, hidden_dim]
        
        return x


class IntuitionMLP(nn.Module):
    """
    直觉 MLP
    
    轻量级网络，学习从感知到直觉的映射
    """
    
    def __init__(self, config: IntuitionConfig):
        super().__init__()
        self.config = config
        
        layers = []
        in_dim = config.hidden_dim
        
        for i in range(config.num_layers):
            out_dim = config.hidden_dim if i < config.num_layers - 1 else config.hidden_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ])
            in_dim = out_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # 输出头
        self.output_head = nn.Linear(config.hidden_dim, config.num_outputs)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (logits, hidden)
        """
        hidden = self.mlp(x)
        logits = self.output_head(hidden)
        return logits, hidden


class IntuitionNetwork(nn.Module):
    """
    直觉网络 (The Intuition Layer)
    
    SEDAC V8.0 的核心组件
    
    功能:
    1. 从多维特征中提取"直觉"信号
    2. 输出三个概率: confident, hallucination, ood
    3. 可通过监督学习进行训练
    
    使用:
        network = IntuitionNetwork(config)
        signal = network(features, layer_idx=10)
        
        if signal.p_confident > 0.7:
            # Early exit
        elif signal.p_hallucination > 0.5:
            # Trigger intervention
    """
    
    def __init__(self, config: IntuitionConfig = None):
        super().__init__()
        self.config = config or IntuitionConfig()
        
        # 感知器编码器
        self.encoder = SensoryEncoder(self.config)
        
        # 直觉 MLP
        self.mlp = IntuitionMLP(self.config)
        
        # 温度参数 (可学习)
        self.temperature = nn.Parameter(torch.tensor(self.config.temperature))
    
    def forward(
        self,
        features: torch.Tensor,  # [batch_size, num_features]
        layer_idx: int = 0
    ) -> IntuitionSignal:
        """
        前向传播
        
        Args:
            features: 感知特征 [batch_size, num_features]
            layer_idx: 当前层索引
        
        Returns:
            IntuitionSignal 包含三个概率
        """
        # 编码
        encoded = self.encoder(features, layer_idx)
        
        # MLP
        logits, hidden = self.mlp(encoded)
        
        # 温度缩放 + Sigmoid
        probs = torch.sigmoid(logits / self.temperature)
        
        return IntuitionSignal(
            p_confident=probs[:, 0],
            p_hallucination=probs[:, 1],
            p_ood=probs[:, 2],
            logits=logits,
            features=features,
            layer_idx=layer_idx,
        )
    
    def get_decision(self, signal: IntuitionSignal) -> torch.Tensor:
        """
        根据直觉信号做出决策
        
        Returns:
            decision: [batch_size] 
                0 = CONTINUE (继续计算)
                1 = EXIT (早退)
                2 = INTERVENE (干预)
        """
        batch_size = signal.p_confident.shape[0]
        decision = torch.zeros(batch_size, dtype=torch.long, device=signal.p_confident.device)
        
        # 优先级: Hallucination > OOD > Confident
        
        # 1. 幻觉风险高 -> 干预
        halluc_mask = signal.p_hallucination > self.config.hallucination_threshold
        decision[halluc_mask] = 2
        
        # 2. 超纲检测 -> 干预 (如果没有幻觉风险)
        ood_mask = (signal.p_ood > self.config.ood_threshold) & ~halluc_mask
        decision[ood_mask] = 2
        
        # 3. 高置信 -> 退出 (如果没有风险)
        confident_mask = (
            (signal.p_confident > self.config.confident_threshold) &
            ~halluc_mask & ~ood_mask
        )
        decision[confident_mask] = 1
        
        return decision


class FeatureExtractor:
    """
    特征提取器
    
    从 hidden states 和 entropies 中提取直觉网络所需的特征
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prev_hidden: Optional[torch.Tensor] = None
        self.prev_entropy: Optional[torch.Tensor] = None
        self.history_norms: List[torch.Tensor] = []
    
    def reset(self):
        self.prev_hidden = None
        self.prev_entropy = None
        self.history_norms = []
    
    def extract(
        self,
        hidden: torch.Tensor,           # [batch_size, hidden_dim]
        entropy: Optional[torch.Tensor] = None,  # [batch_size]
        attention_weights: Optional[torch.Tensor] = None,  # [batch_size, num_heads, seq_len, seq_len]
    ) -> torch.Tensor:
        """
        提取 8 维特征向量
        
        Features:
        0. entropy (当前层熵)
        1. entropy_delta (熵变化)
        2. stability (与上一层的相似度)
        3. hidden_norm (隐状态范数)
        4. norm_delta (范数变化)
        5. norm_acceleration (范数加速度)
        6. attention_entropy (注意力熵, 如果有)
        7. layer_progress (层进度比例)
        """
        batch_size = hidden.shape[0]
        features = torch.zeros(batch_size, 8, device=self.device)
        
        # 0. Entropy
        if entropy is not None:
            features[:, 0] = entropy
        
        # 1. Entropy delta
        if entropy is not None and self.prev_entropy is not None:
            features[:, 1] = entropy - self.prev_entropy
        
        # 2. Stability
        if self.prev_hidden is not None:
            cos_sim = F.cosine_similarity(self.prev_hidden.float(), hidden.float(), dim=1)
            features[:, 2] = (cos_sim + 1.0) / 2.0
        else:
            features[:, 2] = 1.0
        
        # 3. Hidden norm
        hidden_norm = torch.norm(hidden.float(), dim=1)
        features[:, 3] = hidden_norm / 1000.0  # 归一化
        
        # 4. Norm delta
        if len(self.history_norms) > 0:
            features[:, 4] = (hidden_norm - self.history_norms[-1]) / 100.0
        
        # 5. Norm acceleration
        if len(self.history_norms) >= 2:
            prev_delta = self.history_norms[-1] - self.history_norms[-2]
            curr_delta = hidden_norm - self.history_norms[-1]
            features[:, 5] = (curr_delta - prev_delta) / 100.0
        
        # 6. Attention entropy (if available)
        if attention_weights is not None:
            # 计算每个 head 的注意力熵，然后平均
            attn = attention_weights.float()  # [batch, heads, seq, seq]
            attn_entropy = -torch.sum(attn * torch.log(attn + 1e-10), dim=-1)  # [batch, heads, seq]
            features[:, 6] = attn_entropy.mean(dim=(1, 2)) / 10.0  # 归一化
        
        # 7. 保留位 (可用于层进度等)
        features[:, 7] = 0.0
        
        # 更新历史
        self.prev_hidden = hidden.detach()
        if entropy is not None:
            self.prev_entropy = entropy.detach()
        self.history_norms.append(hidden_norm.detach())
        
        return features


class IntuitionLoss(nn.Module):
    """
    直觉网络损失函数
    
    L = L_speed + λ * L_accuracy + γ * L_calibration
    
    - L_speed: 鼓励早退 (BCE with speedup reward)
    - L_accuracy: 惩罚错误退出 (BCE with risk penalty)
    - L_calibration: 校准损失 (ECE)
    """
    
    def __init__(self, config: IntuitionConfig):
        super().__init__()
        self.config = config
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(
        self,
        signal: IntuitionSignal,
        targets: Dict[str, torch.Tensor],
        layer_idx: int,
        total_layers: int
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失
        
        Args:
            signal: 直觉信号
            targets: {
                'is_correct': [batch_size] bool - 最终预测是否正确
                'is_ood': [batch_size] bool - 是否超纲
                'optimal_exit_layer': [batch_size] int - 最优退出层
            }
            layer_idx: 当前层
            total_layers: 总层数
        
        Returns:
            Dict with 'total', 'speed', 'accuracy', 'calibration'
        """
        batch_size = signal.p_confident.shape[0]
        device = signal.p_confident.device
        
        is_correct = targets['is_correct'].float()
        is_ood = targets.get('is_ood', torch.zeros(batch_size, device=device)).float()
        optimal_layer = targets.get('optimal_exit_layer', 
                                   torch.full((batch_size,), total_layers, device=device)).float()
        
        # === L_speed: 鼓励在正确时早退 ===
        # 如果预测正确，应该 p_confident 高
        # 越早的层，reward 越高
        layer_progress = layer_idx / total_layers
        speed_reward = (1 - layer_progress) * is_correct  # 早退且正确 = 高 reward
        
        speed_loss = self.bce(signal.logits[:, 0], speed_reward)
        
        # === L_accuracy: 惩罚错误退出 ===
        # 如果预测错误但 p_confident 高 -> 惩罚
        # 如果预测错误，应该 p_hallucination 高
        accuracy_target = 1 - is_correct  # 错误时应该报警
        accuracy_loss = self.bce(signal.logits[:, 1], accuracy_target)
        
        # === L_calibration: 校准损失 ===
        # p_confident 应该与实际准确率匹配
        # 使用 focal loss 形式来处理校准
        p_confident = signal.p_confident.detach()
        calibration_error = torch.abs(p_confident - is_correct)
        calibration_loss = calibration_error ** 2  # 平方误差
        
        # === OOD Loss ===
        ood_loss = self.bce(signal.logits[:, 2], is_ood)
        
        # === 总损失 ===
        total_loss = (
            self.config.lambda_speed * speed_loss.mean() +
            self.config.lambda_accuracy * accuracy_loss.mean() +
            self.config.lambda_calibration * calibration_loss.mean() +
            0.5 * ood_loss.mean()
        )
        
        return {
            'total': total_loss,
            'speed': speed_loss.mean(),
            'accuracy': accuracy_loss.mean(),
            'calibration': calibration_loss.mean(),
            'ood': ood_loss.mean(),
        }
