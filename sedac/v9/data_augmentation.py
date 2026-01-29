"""
SEDAC V9.0 - 数据增强模块

生成多样化的训练数据，覆盖：
- 数学推理
- 代码理解
- 逻辑推理
- 事实检索
- 创意生成

核心理念：
- 不同任务类型有不同的认知负荷分布
- 数据多样性是泛化能力的关键
"""

from __future__ import annotations
import torch
import json
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """任务类型"""
    MATH = "math"           # 数学推理
    CODE = "code"           # 代码理解
    LOGIC = "logic"         # 逻辑推理
    FACT = "fact"           # 事实检索
    CREATIVE = "creative"   # 创意生成


@dataclass
class SyntheticSample:
    """合成样本"""
    token_idx: int
    task_type: str
    difficulty: float           # 难度 ∈ [0, 1]
    features_per_layer: List[List[float]]
    is_correct: bool
    is_ood: bool
    optimal_exit_layer: int
    final_entropy: float
    cognitive_load: float       # 真实认知负荷
    metadata: Dict[str, Any]


class DataAugmentor:
    """
    数据增强器
    
    基于任务类型和难度生成合成训练数据
    """
    
    def __init__(
        self,
        num_layers: int = 36,
        num_features: int = 8,
        seed: int = 42,
    ):
        self.num_layers = num_layers
        self.num_features = num_features
        self.rng = np.random.RandomState(seed)
        
        # 任务类型的特征模式
        self.task_patterns = {
            TaskType.MATH: self._math_pattern,
            TaskType.CODE: self._code_pattern,
            TaskType.LOGIC: self._logic_pattern,
            TaskType.FACT: self._fact_pattern,
            TaskType.CREATIVE: self._creative_pattern,
        }
    
    def _base_entropy_curve(self, difficulty: float, layer_progress: np.ndarray) -> np.ndarray:
        """
        基础熵曲线
        
        高难度：熵值整体较高，下降缓慢
        低难度：熵值整体较低，下降快速
        """
        # 初始熵与难度正相关
        initial_entropy = 1.0 + difficulty * 4.0  # [1, 5]
        
        # 收敛速度与难度负相关
        convergence_rate = 3.0 * (1.0 - difficulty * 0.7)  # 高难度收敛慢
        
        # 最终熵与难度正相关
        final_entropy = 0.5 + difficulty * 2.0  # [0.5, 2.5]
        
        # 指数衰减曲线
        entropy = final_entropy + (initial_entropy - final_entropy) * np.exp(-convergence_rate * layer_progress)
        
        # 添加随机波动
        noise = self.rng.normal(0, 0.1 * difficulty, len(layer_progress))
        entropy = np.clip(entropy + noise, 0.1, 10.0)
        
        return entropy
    
    def _base_stability_curve(self, difficulty: float, layer_progress: np.ndarray) -> np.ndarray:
        """
        基础稳定性曲线
        
        高难度：稳定性低，波动大
        低难度：稳定性高，快速收敛
        """
        # 基础稳定性
        base_stability = 0.5 + 0.4 * (1.0 - difficulty)  # [0.5, 0.9]
        
        # 随层数增加稳定性提高
        stability = base_stability + (1.0 - base_stability) * (1.0 - np.exp(-2.0 * layer_progress))
        
        # 高难度任务有更多波动
        noise = self.rng.normal(0, 0.1 * difficulty, len(layer_progress))
        stability = np.clip(stability + noise, 0.0, 1.0)
        
        return stability
    
    def _math_pattern(self, difficulty: float) -> Dict[str, np.ndarray]:
        """
        数学推理模式
        
        特征：
        - 熵值在关键推理步骤会突然升高
        - 稳定性在推理链中间较低
        - 存在明确的"顿悟"点
        """
        layer_progress = np.linspace(0, 1, self.num_layers)
        
        # 基础曲线
        entropy = self._base_entropy_curve(difficulty, layer_progress)
        stability = self._base_stability_curve(difficulty, layer_progress)
        
        # 数学特有：推理步骤突变
        num_steps = int(2 + difficulty * 4)  # 难度越高，推理步骤越多
        step_positions = self.rng.uniform(0.2, 0.8, num_steps)
        
        for pos in step_positions:
            idx = int(pos * self.num_layers)
            if idx < self.num_layers:
                # 在推理步骤处熵值突然升高
                entropy[idx:min(idx+3, self.num_layers)] += 0.5 * difficulty
                stability[idx:min(idx+2, self.num_layers)] -= 0.1 * difficulty
        
        return {
            "entropy": np.clip(entropy, 0.1, 10.0),
            "stability": np.clip(stability, 0.0, 1.0),
        }
    
    def _code_pattern(self, difficulty: float) -> Dict[str, np.ndarray]:
        """
        代码理解模式
        
        特征：
        - 熵值在语法边界处变化
        - 稳定性相对较高（结构化）
        - 嵌套深度影响认知负荷
        """
        layer_progress = np.linspace(0, 1, self.num_layers)
        
        entropy = self._base_entropy_curve(difficulty * 0.8, layer_progress)  # 代码相对结构化
        stability = self._base_stability_curve(difficulty * 0.7, layer_progress)
        
        # 代码特有：周期性模式（对应代码块结构）
        period = max(3, int((1.0 - difficulty) * 10))
        periodic_factor = 0.3 * np.sin(2 * np.pi * layer_progress * self.num_layers / period)
        entropy += periodic_factor * difficulty
        
        return {
            "entropy": np.clip(entropy, 0.1, 10.0),
            "stability": np.clip(stability, 0.0, 1.0),
        }
    
    def _logic_pattern(self, difficulty: float) -> Dict[str, np.ndarray]:
        """
        逻辑推理模式
        
        特征：
        - 熵值随推理深度增加
        - 需要更多层来收敛
        - 存在"死胡同"和"回溯"
        """
        layer_progress = np.linspace(0, 1, self.num_layers)
        
        entropy = self._base_entropy_curve(difficulty * 1.2, layer_progress)  # 逻辑推理更复杂
        stability = self._base_stability_curve(difficulty * 1.1, layer_progress)
        
        # 逻辑特有：可能存在回溯点
        if difficulty > 0.5 and self.rng.random() > 0.3:
            backtrack_pos = self.rng.uniform(0.3, 0.6)
            idx = int(backtrack_pos * self.num_layers)
            # 回溯：熵值突然升高，稳定性下降
            entropy[idx:min(idx+5, self.num_layers)] *= 1.3
            stability[idx:min(idx+3, self.num_layers)] *= 0.7
        
        return {
            "entropy": np.clip(entropy, 0.1, 10.0),
            "stability": np.clip(stability, 0.0, 1.0),
        }
    
    def _fact_pattern(self, difficulty: float) -> Dict[str, np.ndarray]:
        """
        事实检索模式
        
        特征：
        - 要么快速找到（低熵），要么找不到（高熵）
        - 稳定性呈阶跃变化
        - 较少的中间状态
        """
        layer_progress = np.linspace(0, 1, self.num_layers)
        
        # 事实检索：二分法特征
        if difficulty < 0.5:  # 简单事实：快速检索
            entropy = 3.0 * np.exp(-5.0 * layer_progress) + 0.3
            stability = 0.6 + 0.35 * (1.0 - np.exp(-4.0 * layer_progress))
        else:  # 复杂事实：可能OOD
            entropy = 2.0 + 2.0 * difficulty * (1.0 - np.exp(-1.0 * layer_progress))
            stability = 0.4 + 0.2 * layer_progress + self.rng.normal(0, 0.1, self.num_layers)
        
        return {
            "entropy": np.clip(entropy, 0.1, 10.0),
            "stability": np.clip(stability, 0.0, 1.0),
        }
    
    def _creative_pattern(self, difficulty: float) -> Dict[str, np.ndarray]:
        """
        创意生成模式
        
        特征：
        - 熵值整体较高（多种可能性）
        - 稳定性持续波动
        - 没有明确的"正确答案"
        """
        layer_progress = np.linspace(0, 1, self.num_layers)
        
        # 创意任务：高熵是正常的
        base_entropy = 2.5 + difficulty * 1.5
        entropy = base_entropy + 0.5 * np.sin(4 * np.pi * layer_progress)
        entropy += self.rng.normal(0, 0.3, self.num_layers)
        
        # 稳定性持续波动
        stability = 0.5 + 0.2 * np.cos(3 * np.pi * layer_progress)
        stability += self.rng.normal(0, 0.15, self.num_layers)
        
        return {
            "entropy": np.clip(entropy, 0.1, 10.0),
            "stability": np.clip(stability, 0.0, 1.0),
        }
    
    def _compute_optimal_exit(
        self,
        entropy: np.ndarray,
        stability: np.ndarray,
        difficulty: float,
        is_correct: bool,
    ) -> int:
        """
        计算最优退出层
        
        基于熵和稳定性的动态阈值
        """
        if not is_correct:
            return self.num_layers  # 错误预测不应该早退
        
        # 动态阈值：基于当前样本的统计量
        entropy_threshold = np.percentile(entropy, 30)  # 熵低于30分位数
        stability_threshold = np.percentile(stability, 70)  # 稳定性高于70分位数
        
        # 找到第一个满足条件的层
        for layer_idx in range(self.num_layers):
            if (entropy[layer_idx] <= entropy_threshold and 
                stability[layer_idx] >= stability_threshold):
                # 确保至少跑过一定比例的层
                min_layers = int(self.num_layers * 0.1 * (1 + difficulty))
                return max(layer_idx, min_layers)
        
        return self.num_layers
    
    def generate_sample(
        self,
        token_idx: int,
        task_type: TaskType,
        difficulty: float,
    ) -> SyntheticSample:
        """
        生成单个合成样本
        """
        # 获取任务特定模式
        pattern_fn = self.task_patterns[task_type]
        patterns = pattern_fn(difficulty)
        
        entropy = patterns["entropy"]
        stability = patterns["stability"]
        
        # 决定是否正确
        # 高难度任务更容易出错
        is_correct = bool(self.rng.random() > difficulty * 0.3)
        
        # OOD检测
        # 高熵 + 低稳定性 = 可能OOD
        avg_entropy = np.mean(entropy)
        avg_stability = np.mean(stability)
        is_ood = bool((avg_entropy > 4.0 and avg_stability < 0.5) or self.rng.random() < difficulty * 0.1)
        
        # 计算最优退出层
        optimal_exit = self._compute_optimal_exit(entropy, stability, difficulty, is_correct)
        
        # 构建8维特征序列
        layer_progress = np.linspace(0, 1, self.num_layers)
        features_per_layer = []
        
        prev_entropy = entropy[0]
        prev_norm = 1000.0 + self.rng.normal(0, 100)
        
        for layer_idx in range(self.num_layers):
            current_entropy = entropy[layer_idx]
            current_stability = stability[layer_idx]
            current_norm = prev_norm + self.rng.normal(0, 50)
            
            features = [
                float(current_entropy),                          # 0: entropy
                float(current_entropy - prev_entropy),           # 1: entropy_delta
                float(current_stability),                        # 2: stability
                float(current_norm / 1000.0),                    # 3: hidden_norm (normalized)
                float((current_norm - prev_norm) / 100.0),       # 4: norm_delta
                float(self.rng.normal(0, 0.1)),                  # 5: norm_acceleration
                float(0.0),                                      # 6: attention_entropy (reserved)
                float(layer_progress[layer_idx]),                # 7: layer_progress
            ]
            
            features_per_layer.append(features)
            prev_entropy = current_entropy
            prev_norm = current_norm
        
        # 认知负荷 = f(难度, 熵, 稳定性)
        cognitive_load = 0.4 * difficulty + 0.3 * (avg_entropy / 5.0) + 0.3 * (1.0 - avg_stability)
        cognitive_load = min(1.0, max(0.0, cognitive_load))
        
        return SyntheticSample(
            token_idx=token_idx,
            task_type=task_type.value,
            difficulty=difficulty,
            features_per_layer=features_per_layer,
            is_correct=is_correct,
            is_ood=is_ood,
            optimal_exit_layer=optimal_exit,
            final_entropy=float(entropy[-1]),
            cognitive_load=cognitive_load,
            metadata={
                "avg_entropy": float(avg_entropy),
                "avg_stability": float(avg_stability),
                "entropy_std": float(np.std(entropy)),
            },
        )
    
    def generate_dataset(
        self,
        num_samples_per_type: int = 500,
        difficulty_distribution: str = "uniform",
    ) -> List[SyntheticSample]:
        """
        生成完整数据集
        
        Args:
            num_samples_per_type: 每种任务类型的样本数
            difficulty_distribution: 难度分布 ("uniform", "beta", "bimodal")
        """
        samples = []
        token_idx = 0
        
        for task_type in TaskType:
            logger.info(f"Generating {num_samples_per_type} samples for {task_type.value}...")
            
            for _ in range(num_samples_per_type):
                # 生成难度
                if difficulty_distribution == "uniform":
                    difficulty = self.rng.uniform(0, 1)
                elif difficulty_distribution == "beta":
                    # Beta分布：更多简单和困难样本
                    difficulty = self.rng.beta(0.5, 0.5)
                elif difficulty_distribution == "bimodal":
                    # 双峰分布：要么简单要么困难
                    if self.rng.random() < 0.5:
                        difficulty = self.rng.beta(2, 5)  # 偏简单
                    else:
                        difficulty = self.rng.beta(5, 2)  # 偏困难
                else:
                    difficulty = self.rng.uniform(0, 1)
                
                sample = self.generate_sample(token_idx, task_type, difficulty)
                samples.append(sample)
                token_idx += 1
        
        # 打乱顺序
        self.rng.shuffle(samples)
        
        logger.info(f"Generated {len(samples)} total samples")
        return samples
    
    def save_dataset(
        self,
        samples: List[SyntheticSample],
        output_path: str,
    ):
        """保存数据集"""
        data = {
            "num_layers": self.num_layers,
            "num_features": self.num_features,
            "samples": [asdict(s) for s in samples],
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f)
        
        logger.info(f"Saved dataset to {output_path}")
    
    def load_and_merge(
        self,
        existing_path: str,
        new_samples: List[SyntheticSample],
    ) -> List[Dict]:
        """加载现有数据并合并新样本"""
        with open(existing_path, 'r') as f:
            existing = json.load(f)
        
        existing_samples = existing.get("samples", [])
        new_dicts = [asdict(s) for s in new_samples]
        
        # 合并
        merged = existing_samples + new_dicts
        
        logger.info(f"Merged {len(existing_samples)} existing + {len(new_samples)} new = {len(merged)} total")
        return merged


def augment_training_data(
    existing_data_path: str = "sedac_v8_training_data.json",
    output_path: str = "sedac_v9_augmented_data.json",
    samples_per_type: int = 500,
    seed: int = 42,
):
    """
    增强训练数据的便捷函数
    """
    augmentor = DataAugmentor(seed=seed)
    
    # 生成新样本
    new_samples = augmentor.generate_dataset(
        num_samples_per_type=samples_per_type,
        difficulty_distribution="beta",
    )
    
    # 加载现有数据并合并
    if Path(existing_data_path).exists():
        merged = augmentor.load_and_merge(existing_data_path, new_samples)
        data = {
            "num_layers": augmentor.num_layers,
            "num_features": augmentor.num_features,
            "samples": merged,
        }
    else:
        data = {
            "num_layers": augmentor.num_layers,
            "num_features": augmentor.num_features,
            "samples": [asdict(s) for s in new_samples],
        }
    
    # 保存
    with open(output_path, 'w') as f:
        json.dump(data, f)
    
    logger.info(f"Augmented data saved to {output_path}")
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    augment_training_data(samples_per_type=1000)
