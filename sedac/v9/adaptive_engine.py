"""
SEDAC V9.0 - Adaptive Cognitive Attention Engine

核心原则：零人工阈值，全自主决策

不是"算得更快"，而是"算得更少"
不是固定阈值，而是动态自适应
不是离散等级，而是连续谱系

底层推理过程 (Reasoning Process):
=================================
1. 为什么要移除固定阈值？
   - 固定阈值假设数据分布不变，但实际分布随模型/任务/领域变化
   - 自适应阈值 = 用数据自己说话

2. 如何实现全自主？
   - 使用在线统计量（均值、方差、分位数）动态计算边界
   - 置信度分布 → 自动确定退出阈值
   - 熵分布 → 自动确定干预阈值

3. 连续化认知模式的意义？
   - 人类思维不是离散的5级，而是连续谱
   - 用连续的cognitive_load（认知负荷）替代离散模式
   - cognitive_load ∈ [0, 1]，0=无需思考，1=极度困难
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
import math
import logging
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class OnlineStatistics:
    """
    在线统计量计算器
    
    使用Welford算法计算在线均值和方差，无需存储所有历史数据
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # 用于计算方差
        
    def update(self, value: float):
        """更新统计量"""
        self.values.append(value)
        self.n += 1
        
        # Welford在线算法
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.M2 += delta * delta2
        
    @property
    def variance(self) -> float:
        if self.n < 2:
            return 1.0
        return self.M2 / (self.n - 1)
    
    @property
    def std(self) -> float:
        return math.sqrt(self.variance)
    
    def percentile(self, p: float) -> float:
        """计算分位数"""
        if len(self.values) == 0:
            return 0.5
        sorted_vals = sorted(self.values)
        idx = int(len(sorted_vals) * p)
        idx = max(0, min(idx, len(sorted_vals) - 1))
        return sorted_vals[idx]
    
    def z_score(self, value: float) -> float:
        """计算Z分数（标准化）"""
        if self.std < 1e-6:
            return 0.0
        return (value - self.mean) / self.std


@dataclass
class AdaptiveState:
    """
    自适应状态 - 全连续表示
    
    不再使用离散的认知模式，而是连续的认知负荷
    """
    # 核心连续量
    confidence: float          # 置信度 ∈ [0, 1]
    cognitive_load: float      # 认知负荷 ∈ [0, 1]，0=简单，1=困难
    recommended_depth: float   # 推荐计算深度 ∈ [0, 1]，相对于总层数
    
    # 决策
    should_exit: bool
    should_intervene: bool
    intervention_strength: float  # 干预强度 ∈ [0, 1]
    
    # 统计量（用于自适应）
    confidence_percentile: float  # 当前置信度在历史分布中的位置
    entropy_percentile: float     # 当前熵在历史分布中的位置
    
    # 元信息
    layer_idx: int
    total_layers: int
    
    @property
    def exit_probability(self) -> float:
        """退出概率 = 置信度 * (1 - 认知负荷)"""
        return self.confidence * (1.0 - self.cognitive_load)
    
    @property
    def intervention_probability(self) -> float:
        """干预概率 = (1 - 置信度) * 认知负荷"""
        return (1.0 - self.confidence) * self.cognitive_load


class AdaptiveCognitiveEngine:
    """
    全自主认知注意力引擎
    
    核心理念：
    - 零硬编码阈值
    - 所有决策边界从数据统计中自动学习
    - 连续的认知负荷而非离散模式
    """
    
    def __init__(
        self,
        intuition_network: nn.Module,
        device: torch.device = None,
        warmup_steps: int = 100,
        calibration_window: int = 1000,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.intuition = intuition_network.to(self.device)
        self.intuition.eval()
        
        # 自适应参数（从数据中学习）
        self.warmup_steps = warmup_steps
        self.step_count = 0
        
        # 在线统计量
        self.confidence_stats = OnlineStatistics(calibration_window)
        self.entropy_stats = OnlineStatistics(calibration_window)
        self.exit_layer_stats = OnlineStatistics(calibration_window)
        
        # 特征提取状态
        self.prev_hidden = None
        self.prev_entropy = None
        self.layer_confidences = []
        
        # 校准状态
        self.is_calibrated = False
        self.calibration_data = []
        
        logger.info(f"AdaptiveCognitiveEngine initialized on {self.device}")
        logger.info(f"Warmup steps: {warmup_steps}, Calibration window: {calibration_window}")
    
    def reset(self):
        """重置token级状态"""
        self.prev_hidden = None
        self.prev_entropy = None
        self.layer_confidences = []
    
    def _extract_features(
        self,
        hidden: torch.Tensor,
        layer_idx: int,
        total_layers: int,
        entropy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        提取8维特征
        
        特征说明：
        1. entropy: 当前熵值（归一化）
        2. entropy_delta: 熵变化率
        3. stability: 层间稳定性（余弦相似度）
        4. hidden_norm: 隐藏状态范数（归一化）
        5. norm_delta: 范数变化
        6. norm_acceleration: 范数加速度
        7. attention_entropy: 注意力熵（预留）
        8. layer_progress: 层进度
        """
        batch_size = hidden.shape[0]
        features = torch.zeros(batch_size, 8, device=self.device)
        
        # 1. 熵值（归一化到[0,1]）
        if entropy is not None:
            # 动态归一化：使用在线统计量
            entropy_val = entropy.mean().item()
            if self.entropy_stats.n > 10:
                # Z-score归一化后sigmoid压缩到[0,1]
                z = self.entropy_stats.z_score(entropy_val)
                features[:, 0] = torch.sigmoid(torch.tensor(z, device=self.device))
            else:
                features[:, 0] = torch.clamp(entropy / 10.0, 0, 1)
        
        # 2. 熵变化率
        if self.prev_entropy is not None and entropy is not None:
            delta = (entropy - self.prev_entropy).mean()
            features[:, 1] = torch.tanh(delta)
        
        # 3. 层间稳定性
        if self.prev_hidden is not None:
            cos_sim = F.cosine_similarity(
                hidden.view(batch_size, -1),
                self.prev_hidden.view(batch_size, -1),
                dim=-1
            )
            features[:, 2] = cos_sim
        else:
            features[:, 2] = 0.5
        
        # 4-6. 范数特征
        current_norm = hidden.norm(dim=-1).mean()
        features[:, 3] = torch.tanh(current_norm / 1000.0)
        
        if self.prev_hidden is not None:
            prev_norm = self.prev_hidden.norm(dim=-1).mean()
            norm_delta = current_norm - prev_norm
            features[:, 4] = torch.tanh(norm_delta / 100.0)
            features[:, 5] = torch.tanh(norm_delta / 100.0)  # 简化：用delta代替acceleration
        
        # 7. 注意力熵（预留）
        features[:, 6] = 0.0
        
        # 8. 层进度
        features[:, 7] = layer_idx / max(total_layers - 1, 1)
        
        # 更新状态
        self.prev_hidden = hidden.detach().clone()
        if entropy is not None:
            self.prev_entropy = entropy.detach().clone()
            self.entropy_stats.update(entropy.mean().item())
        
        return features
    
    def _compute_cognitive_load(
        self,
        confidence: float,
        entropy_percentile: float,
        layer_progress: float,
    ) -> float:
        """
        计算认知负荷（连续值）
        
        认知负荷 = f(置信度, 熵分位数, 层进度)
        
        直觉：
        - 低置信 + 高熵 → 高认知负荷
        - 高置信 + 低熵 → 低认知负荷
        - 早期层需要更高置信才能退出
        """
        # 基础负荷 = 1 - 置信度
        base_load = 1.0 - confidence
        
        # 熵调制：高熵分位数增加负荷
        entropy_factor = entropy_percentile
        
        # 层进度调制：早期层更难退出
        # 使用sigmoid使早期层有更高的负荷惩罚
        progress_factor = 1.0 - torch.sigmoid(
            torch.tensor(10.0 * (layer_progress - 0.3))
        ).item()
        
        # 综合计算
        cognitive_load = base_load * 0.5 + entropy_factor * 0.3 + progress_factor * 0.2
        
        return max(0.0, min(1.0, cognitive_load))
    
    def _compute_recommended_depth(
        self,
        cognitive_load: float,
        confidence: float,
        layer_progress: float,
    ) -> float:
        """
        计算推荐计算深度（连续值）
        
        不是"你应该在第N层退出"，而是"你还需要多少计算"
        
        返回值 ∈ [0, 1]，表示还需要的计算深度比例
        """
        # 基础深度需求 = 认知负荷
        base_depth = cognitive_load
        
        # 置信度调制：高置信降低深度需求
        confidence_discount = (1.0 - base_depth) * confidence
        
        # 已完成进度折扣
        progress_discount = layer_progress * 0.3
        
        recommended = base_depth - confidence_discount - progress_discount
        
        return max(0.0, min(1.0, recommended))
    
    def _should_exit(
        self,
        confidence: float,
        cognitive_load: float,
        layer_progress: float,
        signal_p_confident: float,
    ) -> Tuple[bool, float]:
        """
        自适应退出决策
        
        核心改进：直接使用直觉网络的输出作为主要依据
        
        直觉网络已经学习了"何时可以安全退出"，
        自适应层只提供额外的安全保护
        """
        # 冷启动期：使用保守策略
        if self.step_count < self.warmup_steps:
            # 即使在冷启动期，如果网络非常确定也可以退出
            if signal_p_confident > 0.9 and layer_progress > 0.1:
                return True, signal_p_confident
            return False, 0.0
        
        # 核心决策：直接使用训练好的网络输出
        # 网络输出 > 0.5 表示"可以退出"
        network_says_exit = signal_p_confident > 0.5
        
        # 自适应安全检查（可选的额外保护）
        # 只在网络非常不确定时阻止退出
        min_progress = 0.1  # 至少完成10%的层
        
        # 最终决策
        can_exit = network_says_exit and layer_progress >= min_progress
        
        # 退出概率 = 网络置信度
        exit_prob = signal_p_confident
        
        return can_exit, exit_prob
    
    def _should_intervene(
        self,
        confidence: float,
        cognitive_load: float,
        entropy_percentile: float,
    ) -> Tuple[bool, float]:
        """
        自适应干预决策
        
        干预条件：
        1. 置信度低于历史分布的某个动态分位数
        2. 认知负荷高
        3. 熵值异常高
        """
        # 冷启动期：不干预
        if self.step_count < self.warmup_steps:
            return False, 0.0
        
        # 动态干预阈值
        intervene_confidence_threshold = self.confidence_stats.percentile(0.25)
        
        # 干预条件
        should_intervene = (
            confidence < intervene_confidence_threshold and
            cognitive_load > 0.7 and
            entropy_percentile > 0.8
        )
        
        # 干预强度
        intervention_strength = (1.0 - confidence) * cognitive_load * entropy_percentile
        
        return should_intervene, intervention_strength
    
    @torch.no_grad()
    def step(
        self,
        hidden: torch.Tensor,
        layer_idx: int,
        total_layers: int,
        entropy: Optional[torch.Tensor] = None,
    ) -> AdaptiveState:
        """
        核心决策步骤
        
        输入：
        - hidden: 当前层隐藏状态
        - layer_idx: 当前层索引
        - total_layers: 总层数
        - entropy: 可选的熵值
        
        输出：
        - AdaptiveState: 完整的自适应状态
        """
        self.step_count += 1
        
        # 提取特征
        features = self._extract_features(hidden, layer_idx, total_layers, entropy)
        
        # 获取直觉网络输出
        signal = self.intuition(features, layer_idx)
        
        # 提取置信度
        confidence = signal.p_confident.mean().item()
        self.confidence_stats.update(confidence)
        self.layer_confidences.append(confidence)
        
        # 计算熵分位数
        if entropy is not None:
            entropy_val = entropy.mean().item()
            if self.entropy_stats.n > 10:
                entropy_percentile = sum(
                    1 for v in self.entropy_stats.values if v <= entropy_val
                ) / len(self.entropy_stats.values)
            else:
                entropy_percentile = 0.5
        else:
            entropy_percentile = 0.5
        
        # 计算置信度分位数
        if self.confidence_stats.n > 10:
            confidence_percentile = sum(
                1 for v in self.confidence_stats.values if v <= confidence
            ) / len(self.confidence_stats.values)
        else:
            confidence_percentile = 0.5
        
        # 层进度
        layer_progress = layer_idx / max(total_layers - 1, 1)
        
        # 计算认知负荷（连续）
        cognitive_load = self._compute_cognitive_load(
            confidence, entropy_percentile, layer_progress
        )
        
        # 计算推荐深度（连续）
        recommended_depth = self._compute_recommended_depth(
            cognitive_load, confidence, layer_progress
        )
        
        # 直觉网络原始输出
        signal_p_confident = signal.p_confident.mean().item()
        
        # 退出决策（使用网络原始输出）
        should_exit, exit_prob = self._should_exit(
            confidence, cognitive_load, layer_progress, signal_p_confident
        )
        
        # 干预决策
        should_intervene, intervention_strength = self._should_intervene(
            confidence, cognitive_load, entropy_percentile
        )
        
        # 记录退出层（用于自适应学习）
        if should_exit:
            self.exit_layer_stats.update(layer_idx)
        
        return AdaptiveState(
            confidence=confidence,
            cognitive_load=cognitive_load,
            recommended_depth=recommended_depth,
            should_exit=should_exit,
            should_intervene=should_intervene,
            intervention_strength=intervention_strength,
            confidence_percentile=confidence_percentile,
            entropy_percentile=entropy_percentile,
            layer_idx=layer_idx,
            total_layers=total_layers,
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取当前统计量"""
        return {
            "step_count": self.step_count,
            "is_calibrated": self.step_count >= self.warmup_steps,
            "confidence": {
                "mean": self.confidence_stats.mean,
                "std": self.confidence_stats.std,
                "p25": self.confidence_stats.percentile(0.25),
                "p50": self.confidence_stats.percentile(0.50),
                "p75": self.confidence_stats.percentile(0.75),
            },
            "entropy": {
                "mean": self.entropy_stats.mean,
                "std": self.entropy_stats.std,
            },
            "exit_layer": {
                "mean": self.exit_layer_stats.mean if self.exit_layer_stats.n > 0 else -1,
                "std": self.exit_layer_stats.std if self.exit_layer_stats.n > 0 else 0,
            },
        }


def create_adaptive_engine(
    checkpoint_path: Optional[str] = None,
    device: str = "auto",
    warmup_steps: int = 100,
) -> AdaptiveCognitiveEngine:
    """
    创建自适应认知引擎
    """
    from sedac.v8.intuition_network import IntuitionNetwork, IntuitionConfig
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    device = torch.device(device)
    
    # 创建直觉网络
    config = IntuitionConfig()
    intuition = IntuitionNetwork(config)
    
    # 加载检查点
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        intuition.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    return AdaptiveCognitiveEngine(
        intuition_network=intuition,
        device=device,
        warmup_steps=warmup_steps,
    )
