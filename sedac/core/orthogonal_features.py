"""
SEDAC V7.4 - Orthogonal Features for Hallucination Detection

针对 "Confidently Wrong" 现象的多信号检测器

核心洞察:
- Stability 特征失效 (high-risk vs low-risk 差异仅 0.0093)
- 需要正交特征来捕捉 "稳定但错误" 的模式

实现的特征:
1. Entropy Trajectory - 熵值轨迹异常检测
2. Hidden Norm Trajectory - 隐状态范数变化
3. Representation Drift - 表示空间漂移速度
4. Convergence Pattern - 收敛模式分类
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class OrthogonalFeatures:
    """正交特征集合"""
    # Entropy-based
    entropy_mean: torch.Tensor          # 熵均值
    entropy_std: torch.Tensor           # 熵波动
    entropy_trend: torch.Tensor         # 熵趋势 (正=上升, 负=下降)
    entropy_plateau: torch.Tensor       # 熵平台期长度
    
    # Hidden norm-based
    norm_mean: torch.Tensor             # 范数均值
    norm_std: torch.Tensor              # 范数波动
    norm_acceleration: torch.Tensor     # 范数加速度 (二阶导)
    
    # Representation drift
    drift_velocity: torch.Tensor        # 表示漂移速度
    drift_acceleration: torch.Tensor    # 漂移加速度
    
    # Convergence pattern
    early_lock: torch.Tensor           # 早期锁定 (前1/3层稳定度)
    late_drift: torch.Tensor           # 晚期漂移 (后1/3层变化)
    
    # Stability (legacy)
    stability_mean: torch.Tensor
    stability_std: torch.Tensor


class OrthogonalFeatureExtractor:
    """
    正交特征提取器
    
    从 hidden_states 和 entropies 中提取多维特征，
    用于检测 "Confidently Wrong" 模式
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def extract(
        self,
        hidden_states: List[torch.Tensor],
        entropies: Optional[List[torch.Tensor]] = None
    ) -> OrthogonalFeatures:
        """
        提取正交特征
        
        Args:
            hidden_states: [num_layers] x [num_tokens, hidden_dim]
            entropies: [num_layers] x [num_tokens] (optional)
        
        Returns:
            OrthogonalFeatures 对象
        """
        num_layers = len(hidden_states)
        num_tokens = hidden_states[0].shape[0]
        
        # ========== 1. Stability Matrix ==========
        stability_list = []
        with torch.no_grad():
            for layer_idx in range(1, num_layers):
                h_prev = hidden_states[layer_idx - 1].float().to(self.device)
                h_curr = hidden_states[layer_idx].float().to(self.device)
                cos_sim = F.cosine_similarity(h_prev, h_curr, dim=1)
                stability = (cos_sim + 1.0) / 2.0
                stability_list.append(stability)
        
        stability_matrix = torch.stack(stability_list, dim=1)  # [num_tokens, num_layers-1]
        
        # ========== 2. Hidden Norm Matrix ==========
        norm_list = []
        for hs in hidden_states:
            norm = torch.norm(hs.float().to(self.device), dim=1)
            norm_list.append(norm)
        norm_matrix = torch.stack(norm_list, dim=1)  # [num_tokens, num_layers]
        
        # Normalize by first layer norm
        norm_matrix = norm_matrix / (norm_matrix[:, 0:1] + 1e-8)
        
        # ========== 3. Entropy Matrix ==========
        if entropies is not None and len(entropies) > 0:
            entropy_list = [e.float().to(self.device) for e in entropies]
            entropy_matrix = torch.stack(entropy_list, dim=1)  # [num_tokens, num_layers]
        else:
            entropy_matrix = None
        
        # ========== Extract Features ==========
        
        # --- Stability features ---
        stability_mean = stability_matrix.mean(dim=1)
        stability_std = stability_matrix.std(dim=1)
        
        # --- Entropy features ---
        if entropy_matrix is not None:
            entropy_mean = entropy_matrix.mean(dim=1)
            entropy_std = entropy_matrix.std(dim=1)
            
            # Entropy trend: linear regression slope
            x = torch.arange(num_layers, dtype=torch.float32, device=self.device)
            x_centered = x - x.mean()
            entropy_trend = (entropy_matrix * x_centered).sum(dim=1) / (x_centered ** 2).sum()
            
            # Entropy plateau: longest consecutive low-change region
            entropy_diff = torch.abs(entropy_matrix[:, 1:] - entropy_matrix[:, :-1])
            plateau_thresh = entropy_diff.mean(dim=1, keepdim=True) * 0.5
            is_plateau = entropy_diff < plateau_thresh
            entropy_plateau = self._longest_consecutive(is_plateau)
        else:
            entropy_mean = torch.zeros(num_tokens, device=self.device)
            entropy_std = torch.zeros(num_tokens, device=self.device)
            entropy_trend = torch.zeros(num_tokens, device=self.device)
            entropy_plateau = torch.zeros(num_tokens, device=self.device)
        
        # --- Hidden norm features ---
        norm_mean = norm_matrix.mean(dim=1)
        norm_std = norm_matrix.std(dim=1)
        
        # Norm acceleration (second derivative)
        norm_diff1 = norm_matrix[:, 1:] - norm_matrix[:, :-1]
        if norm_diff1.shape[1] > 1:
            norm_diff2 = norm_diff1[:, 1:] - norm_diff1[:, :-1]
            norm_acceleration = norm_diff2.abs().mean(dim=1)
        else:
            norm_acceleration = torch.zeros(num_tokens, device=self.device)
        
        # --- Representation drift ---
        # Drift = 1 - stability (how much the representation changes)
        drift_matrix = 1.0 - stability_matrix
        drift_velocity = drift_matrix.mean(dim=1)
        
        # Drift acceleration
        drift_diff = drift_matrix[:, 1:] - drift_matrix[:, :-1]
        drift_acceleration = drift_diff.abs().mean(dim=1) if drift_diff.shape[1] > 0 else torch.zeros(num_tokens, device=self.device)
        
        # --- Convergence pattern ---
        early_layers = max(1, num_layers // 3)
        late_start = num_layers - early_layers
        
        early_lock = stability_matrix[:, :early_layers].mean(dim=1)
        late_drift = drift_matrix[:, late_start-1:].mean(dim=1) if late_start > 1 else drift_matrix.mean(dim=1)
        
        return OrthogonalFeatures(
            entropy_mean=entropy_mean,
            entropy_std=entropy_std,
            entropy_trend=entropy_trend,
            entropy_plateau=entropy_plateau,
            norm_mean=norm_mean,
            norm_std=norm_std,
            norm_acceleration=norm_acceleration,
            drift_velocity=drift_velocity,
            drift_acceleration=drift_acceleration,
            early_lock=early_lock,
            late_drift=late_drift,
            stability_mean=stability_mean,
            stability_std=stability_std,
        )
    
    def _longest_consecutive(self, bool_matrix: torch.Tensor) -> torch.Tensor:
        """计算每行最长连续 True 的长度"""
        # Simple implementation: iterate through columns
        num_tokens = bool_matrix.shape[0]
        result = torch.zeros(num_tokens, device=self.device)
        current = torch.zeros(num_tokens, device=self.device)
        
        for col in range(bool_matrix.shape[1]):
            mask = bool_matrix[:, col]
            current = torch.where(mask, current + 1, torch.zeros_like(current))
            result = torch.maximum(result, current)
        
        return result


class HallucinationDetector:
    """
    幻觉检测器 - 基于正交特征的多信号融合
    
    核心思想:
    - "Confidently Wrong" 模式: 高稳定性 + 低熵 + 早期锁定
    - 正常收敛模式: 渐进稳定 + 熵下降 + 晚期收敛
    """
    
    def __init__(
        self,
        device: torch.device = None,
        # 特征权重
        w_entropy_plateau: float = 0.3,
        w_early_lock: float = 0.3,
        w_low_drift_accel: float = 0.2,
        w_flat_norm: float = 0.2,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = OrthogonalFeatureExtractor(device=self.device)
        
        self.w_entropy_plateau = w_entropy_plateau
        self.w_early_lock = w_early_lock
        self.w_low_drift_accel = w_low_drift_accel
        self.w_flat_norm = w_flat_norm
    
    def compute_hallucination_score(
        self,
        features: OrthogonalFeatures,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        计算幻觉分数 (0-1, 越高越可能是幻觉)
        
        "Confidently Wrong" 特征:
        1. entropy_plateau 高 -> 熵早早平台化，没有继续优化
        2. early_lock 高 -> 早期就锁定了表示
        3. drift_acceleration 低 -> 没有"探索"行为
        4. norm_std 低 -> 范数变化平缓
        """
        scores = []
        
        # 1. Entropy plateau (normalized by num_layers)
        if features.entropy_plateau.sum() > 0:
            plateau_score = features.entropy_plateau / features.entropy_plateau.max().clamp(min=1)
            scores.append(self.w_entropy_plateau * plateau_score)
        
        # 2. Early lock (high = suspicious)
        early_score = (features.early_lock - features.early_lock.min()) / (features.early_lock.max() - features.early_lock.min() + 1e-8)
        scores.append(self.w_early_lock * early_score)
        
        # 3. Low drift acceleration (inverted: low drift_accel = high score)
        drift_accel_norm = (features.drift_acceleration - features.drift_acceleration.min()) / (features.drift_acceleration.max() - features.drift_acceleration.min() + 1e-8)
        low_drift_score = 1.0 - drift_accel_norm
        scores.append(self.w_low_drift_accel * low_drift_score)
        
        # 4. Flat norm (low std = suspicious)
        norm_std_norm = (features.norm_std - features.norm_std.min()) / (features.norm_std.max() - features.norm_std.min() + 1e-8)
        flat_norm_score = 1.0 - norm_std_norm
        scores.append(self.w_flat_norm * flat_norm_score)
        
        # Combine
        hallucination_score = sum(scores)
        
        if normalize:
            hallucination_score = (hallucination_score - hallucination_score.min()) / (hallucination_score.max() - hallucination_score.min() + 1e-8)
        
        return hallucination_score
    
    def detect(
        self,
        hidden_states: List[torch.Tensor],
        entropies: Optional[List[torch.Tensor]] = None,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        检测幻觉
        
        Returns:
            (hallucination_mask, hallucination_score)
        """
        features = self.extractor.extract(hidden_states, entropies)
        score = self.compute_hallucination_score(features)
        mask = score > threshold
        return mask, score


def analyze_feature_separability(
    features: OrthogonalFeatures,
    high_risk_mask: torch.Tensor
) -> dict:
    """
    分析各特征对 high-risk 的区分度
    
    Returns:
        dict: {feature_name: (high_risk_mean, low_risk_mean, diff, t_statistic)}
    """
    results = {}
    
    hr = high_risk_mask
    lr = ~high_risk_mask
    
    feature_dict = {
        "entropy_mean": features.entropy_mean,
        "entropy_std": features.entropy_std,
        "entropy_trend": features.entropy_trend,
        "entropy_plateau": features.entropy_plateau,
        "norm_mean": features.norm_mean,
        "norm_std": features.norm_std,
        "norm_acceleration": features.norm_acceleration,
        "drift_velocity": features.drift_velocity,
        "drift_acceleration": features.drift_acceleration,
        "early_lock": features.early_lock,
        "late_drift": features.late_drift,
        "stability_mean": features.stability_mean,
        "stability_std": features.stability_std,
    }
    
    for name, feat in feature_dict.items():
        hr_vals = feat[hr]
        lr_vals = feat[lr]
        
        hr_mean = hr_vals.mean().item()
        lr_mean = lr_vals.mean().item()
        diff = hr_mean - lr_mean
        
        # Welch's t-statistic
        hr_std = hr_vals.std().item()
        lr_std = lr_vals.std().item()
        hr_n = hr_vals.shape[0]
        lr_n = lr_vals.shape[0]
        
        se = np.sqrt(hr_std**2 / hr_n + lr_std**2 / lr_n) if hr_n > 0 and lr_n > 0 else 1e-8
        t_stat = diff / (se + 1e-8)
        
        results[name] = {
            "high_risk_mean": hr_mean,
            "low_risk_mean": lr_mean,
            "diff": diff,
            "t_statistic": t_stat,
            "abs_t": abs(t_stat),
        }
    
    return results
