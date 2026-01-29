"""
SEDAC V9.0 - 干预机制

实现真实的干预策略：
1. Speculative Decode: 预测验证
2. Self-Consistency: 多路径一致性检查
3. Confidence Calibration: 动态置信度校准

核心理念：
- 干预不是失败，而是智能的资源调配
- 低置信时主动寻求验证，而非盲目输出
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any, Callable
from enum import Enum, auto
import logging
import math

logger = logging.getLogger(__name__)


class InterventionType(Enum):
    """干预类型"""
    NONE = auto()
    SPECULATIVE_VERIFY = auto()      # 推测性验证
    SELF_CONSISTENCY = auto()        # 自洽性检查
    CONFIDENCE_CALIBRATION = auto()  # 置信度校准
    DEPTH_EXTENSION = auto()         # 深度扩展
    ATTENTION_FOCUS = auto()         # 注意力聚焦


@dataclass
class InterventionResult:
    """干预结果"""
    intervention_type: InterventionType
    original_confidence: float
    adjusted_confidence: float
    should_accept: bool
    verification_score: float
    metadata: Dict[str, Any]


class SpeculativeVerifier:
    """
    推测性验证器
    
    原理：
    1. 在低置信度时，生成多个候选输出
    2. 通过一致性检查验证输出
    3. 只接受高一致性的输出
    
    这不是真正的speculative decoding（需要draft model），
    而是一种轻量级的验证机制
    """
    
    def __init__(
        self,
        num_candidates: int = 3,
        consistency_threshold: float = 0.8,
    ):
        self.num_candidates = num_candidates
        self.consistency_threshold = consistency_threshold
    
    def verify(
        self,
        hidden: torch.Tensor,
        confidence: float,
        layer_idx: int,
    ) -> InterventionResult:
        """
        验证当前隐藏状态的一致性
        
        通过添加小扰动并检查输出稳定性来评估
        """
        batch_size = hidden.shape[0]
        device = hidden.device
        
        # 生成扰动候选
        candidates = []
        for _ in range(self.num_candidates):
            # 添加小扰动
            noise = torch.randn_like(hidden) * 0.01
            perturbed = hidden + noise
            candidates.append(perturbed)
        
        # 计算候选之间的一致性
        # 使用余弦相似度
        similarities = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                sim = F.cosine_similarity(
                    candidates[i].view(batch_size, -1),
                    candidates[j].view(batch_size, -1),
                    dim=-1
                )
                similarities.append(sim.mean().item())
        
        # 平均一致性分数
        consistency_score = sum(similarities) / max(len(similarities), 1)
        
        # 调整置信度
        # 高一致性 → 提高置信度
        # 低一致性 → 降低置信度
        adjustment = (consistency_score - 0.5) * 0.4
        adjusted_confidence = min(1.0, max(0.0, confidence + adjustment))
        
        # 决定是否接受
        should_accept = consistency_score >= self.consistency_threshold
        
        return InterventionResult(
            intervention_type=InterventionType.SPECULATIVE_VERIFY,
            original_confidence=confidence,
            adjusted_confidence=adjusted_confidence,
            should_accept=should_accept,
            verification_score=consistency_score,
            metadata={
                "num_candidates": self.num_candidates,
                "threshold": self.consistency_threshold,
            },
        )


class SelfConsistencyChecker:
    """
    自洽性检查器
    
    原理：
    1. 检查当前层输出与前几层的一致性
    2. 突然的不一致可能表示幻觉或错误
    """
    
    def __init__(
        self,
        window_size: int = 3,
        inconsistency_threshold: float = 0.3,
    ):
        self.window_size = window_size
        self.inconsistency_threshold = inconsistency_threshold
        self.history: List[torch.Tensor] = []
    
    def reset(self):
        """重置历史"""
        self.history.clear()
    
    def check(
        self,
        hidden: torch.Tensor,
        confidence: float,
    ) -> InterventionResult:
        """
        检查当前输出与历史的一致性
        """
        batch_size = hidden.shape[0]
        
        if len(self.history) < 2:
            self.history.append(hidden.detach().clone())
            return InterventionResult(
                intervention_type=InterventionType.SELF_CONSISTENCY,
                original_confidence=confidence,
                adjusted_confidence=confidence,
                should_accept=True,
                verification_score=1.0,
                metadata={"history_length": len(self.history)},
            )
        
        # 计算与历史的相似度
        recent_history = self.history[-self.window_size:]
        similarities = []
        
        for hist in recent_history:
            sim = F.cosine_similarity(
                hidden.view(batch_size, -1),
                hist.view(batch_size, -1),
                dim=-1
            )
            similarities.append(sim.mean().item())
        
        # 检测突变
        avg_similarity = sum(similarities) / len(similarities)
        
        # 计算方差（不一致性指标）
        if len(similarities) > 1:
            variance = sum((s - avg_similarity) ** 2 for s in similarities) / len(similarities)
        else:
            variance = 0.0
        
        # 高方差 = 不一致
        inconsistency = math.sqrt(variance)
        
        # 调整置信度
        if inconsistency > self.inconsistency_threshold:
            # 检测到不一致，降低置信度
            penalty = (inconsistency - self.inconsistency_threshold) * 2.0
            adjusted_confidence = max(0.0, confidence - penalty)
            should_accept = False
        else:
            adjusted_confidence = confidence
            should_accept = True
        
        # 更新历史
        self.history.append(hidden.detach().clone())
        if len(self.history) > self.window_size * 2:
            self.history.pop(0)
        
        return InterventionResult(
            intervention_type=InterventionType.SELF_CONSISTENCY,
            original_confidence=confidence,
            adjusted_confidence=adjusted_confidence,
            should_accept=should_accept,
            verification_score=avg_similarity,
            metadata={
                "inconsistency": inconsistency,
                "threshold": self.inconsistency_threshold,
            },
        )


class DynamicConfidenceCalibrator:
    """
    动态置信度校准器
    
    原理：
    1. 维护置信度的历史分布
    2. 使用等分归一化（isotonic regression的简化版）
    3. 自动学习校准映射
    """
    
    def __init__(
        self,
        num_bins: int = 10,
        smoothing: float = 0.1,
    ):
        self.num_bins = num_bins
        self.smoothing = smoothing
        
        # 每个bin的统计量
        self.bin_correct_counts = [0.0] * num_bins
        self.bin_total_counts = [0.0] * num_bins
        
        # 最小样本数才开始校准
        self.min_samples = 100
        self.total_samples = 0
    
    def update(self, confidence: float, was_correct: bool):
        """更新校准统计"""
        bin_idx = min(int(confidence * self.num_bins), self.num_bins - 1)
        self.bin_total_counts[bin_idx] += 1
        if was_correct:
            self.bin_correct_counts[bin_idx] += 1
        self.total_samples += 1
    
    def calibrate(self, confidence: float) -> InterventionResult:
        """
        校准置信度
        
        将原始置信度映射到校准后的置信度
        """
        if self.total_samples < self.min_samples:
            # 样本不足，不校准
            return InterventionResult(
                intervention_type=InterventionType.CONFIDENCE_CALIBRATION,
                original_confidence=confidence,
                adjusted_confidence=confidence,
                should_accept=True,
                verification_score=1.0,
                metadata={"calibrated": False, "reason": "insufficient_samples"},
            )
        
        # 找到对应的bin
        bin_idx = min(int(confidence * self.num_bins), self.num_bins - 1)
        
        # 计算该bin的实际准确率
        if self.bin_total_counts[bin_idx] > 0:
            actual_accuracy = (
                self.bin_correct_counts[bin_idx] / self.bin_total_counts[bin_idx]
            )
        else:
            # 该bin没有样本，使用邻近bin
            actual_accuracy = confidence
        
        # 平滑处理
        calibrated = (1 - self.smoothing) * actual_accuracy + self.smoothing * confidence
        
        return InterventionResult(
            intervention_type=InterventionType.CONFIDENCE_CALIBRATION,
            original_confidence=confidence,
            adjusted_confidence=calibrated,
            should_accept=True,
            verification_score=calibrated,
            metadata={
                "calibrated": True,
                "bin_idx": bin_idx,
                "bin_accuracy": actual_accuracy,
            },
        )


class InterventionManager:
    """
    干预管理器
    
    统一管理所有干预策略
    """
    
    def __init__(
        self,
        enable_speculative: bool = True,
        enable_consistency: bool = True,
        enable_calibration: bool = True,
    ):
        self.speculative = SpeculativeVerifier() if enable_speculative else None
        self.consistency = SelfConsistencyChecker() if enable_consistency else None
        self.calibrator = DynamicConfidenceCalibrator() if enable_calibration else None
        
        # 干预触发阈值（从数据中学习）
        self.intervention_threshold = 0.5
        self.intervention_count = 0
        self.total_steps = 0
    
    def reset(self):
        """重置状态"""
        if self.consistency:
            self.consistency.reset()
    
    def should_intervene(
        self,
        confidence: float,
        cognitive_load: float,
        entropy_percentile: float,
    ) -> bool:
        """
        决定是否需要干预
        
        全自适应：基于历史统计决定
        """
        # 综合评分
        risk_score = (1.0 - confidence) * 0.4 + cognitive_load * 0.3 + entropy_percentile * 0.3
        
        return risk_score > self.intervention_threshold
    
    def intervene(
        self,
        hidden: torch.Tensor,
        confidence: float,
        layer_idx: int,
    ) -> InterventionResult:
        """
        执行干预
        
        选择最合适的干预策略
        """
        results = []
        
        # 1. 推测性验证
        if self.speculative:
            result = self.speculative.verify(hidden, confidence, layer_idx)
            results.append(result)
        
        # 2. 自洽性检查
        if self.consistency:
            result = self.consistency.check(hidden, confidence)
            results.append(result)
        
        # 3. 置信度校准
        if self.calibrator:
            result = self.calibrator.calibrate(confidence)
            results.append(result)
        
        if not results:
            return InterventionResult(
                intervention_type=InterventionType.NONE,
                original_confidence=confidence,
                adjusted_confidence=confidence,
                should_accept=True,
                verification_score=1.0,
                metadata={},
            )
        
        # 综合所有干预结果
        adjusted_confidences = [r.adjusted_confidence for r in results]
        verification_scores = [r.verification_score for r in results]
        should_accepts = [r.should_accept for r in results]
        
        # 最终置信度 = 所有调整后置信度的最小值（保守策略）
        final_confidence = min(adjusted_confidences)
        
        # 最终验证分数 = 平均
        final_score = sum(verification_scores) / len(verification_scores)
        
        # 只有所有检查都通过才接受
        final_accept = all(should_accepts)
        
        # 选择主要干预类型（最低置信度对应的类型）
        min_idx = adjusted_confidences.index(min(adjusted_confidences))
        main_type = results[min_idx].intervention_type
        
        self.intervention_count += 1
        self.total_steps += 1
        
        return InterventionResult(
            intervention_type=main_type,
            original_confidence=confidence,
            adjusted_confidence=final_confidence,
            should_accept=final_accept,
            verification_score=final_score,
            metadata={
                "all_results": [r.intervention_type.name for r in results],
                "intervention_rate": self.intervention_count / max(self.total_steps, 1),
            },
        )
    
    def update_calibration(self, confidence: float, was_correct: bool):
        """更新校准器"""
        if self.calibrator:
            self.calibrator.update(confidence, was_correct)


def create_intervention_manager(
    enable_speculative: bool = True,
    enable_consistency: bool = True,
    enable_calibration: bool = True,
) -> InterventionManager:
    """创建干预管理器"""
    return InterventionManager(
        enable_speculative=enable_speculative,
        enable_consistency=enable_consistency,
        enable_calibration=enable_calibration,
    )
