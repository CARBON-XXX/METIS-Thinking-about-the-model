"""
SEDAC V9.0 - 自适应训练器

目标：退出精度 95%+

核心改进：
1. 多任务学习：支持不同任务类型
2. 认知负荷预测：连续值而非离散标签
3. 自适应损失权重：根据样本难度动态调整
4. 更强的正则化：防止过拟合
5. 困难样本挖掘：重点训练边界样本
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from sedac.v8.intuition_network import IntuitionNetwork, IntuitionConfig

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveTrainingConfig:
    """训练配置"""
    # 数据
    data_path: str = "sedac_v9_augmented_data.json"
    val_split: float = 0.15
    
    # 训练
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # 学习率调度
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # 损失权重（初始值，会自适应调整）
    exit_weight: float = 1.0
    confidence_weight: float = 1.0
    cognitive_load_weight: float = 0.5
    
    # 正则化
    dropout: float = 0.2
    label_smoothing: float = 0.1
    
    # 困难样本挖掘
    hard_sample_ratio: float = 0.3
    
    # 早停
    patience: int = 15
    min_delta: float = 1e-4
    
    # 保存
    save_dir: str = "checkpoints"
    
    # 设备
    device: str = "auto"


class AdaptiveDataset(Dataset):
    """
    自适应数据集
    
    支持：
    - 多任务类型
    - 认知负荷标签
    - 样本权重（困难样本挖掘）
    """
    
    def __init__(
        self,
        samples: List[Dict],
        num_layers: int = 36,
        compute_weights: bool = True,
    ):
        self.samples = samples
        self.num_layers = num_layers
        
        # 展平数据
        self.flat_data = []
        for sample in samples:
            features_per_layer = sample["features_per_layer"]
            is_correct = sample.get("is_correct", True)
            is_ood = sample.get("is_ood", False)
            optimal_exit = sample.get("optimal_exit_layer", num_layers)
            final_entropy = sample.get("final_entropy", 1.0)
            cognitive_load = sample.get("cognitive_load", 0.5)
            difficulty = sample.get("difficulty", 0.5)
            task_type = sample.get("task_type", "unknown")
            
            for layer_idx, features in enumerate(features_per_layer):
                # 计算该层是否可以安全退出
                can_exit = layer_idx >= optimal_exit and is_correct and not is_ood
                
                # 计算该层的认知负荷（随层数递减）
                layer_progress = layer_idx / max(num_layers - 1, 1)
                layer_cognitive_load = cognitive_load * (1.0 - 0.5 * layer_progress)
                
                self.flat_data.append({
                    "features": features,
                    "layer_idx": layer_idx,
                    "is_correct": is_correct,
                    "is_ood": is_ood,
                    "optimal_exit_layer": optimal_exit,
                    "can_exit": can_exit,
                    "final_entropy": final_entropy,
                    "cognitive_load": layer_cognitive_load,
                    "difficulty": difficulty,
                    "task_type": task_type,
                })
        
        # 计算样本权重
        self.weights = None
        if compute_weights:
            self._compute_sample_weights()
        
        logger.info(f"Created dataset with {len(self.flat_data)} samples")
    
    def _compute_sample_weights(self):
        """
        计算样本权重
        
        困难样本（边界样本）获得更高权重
        """
        weights = []
        for item in self.flat_data:
            # 基础权重
            w = 1.0
            
            # 困难样本加权
            difficulty = item["difficulty"]
            w *= 1.0 + difficulty  # 难度越高权重越大
            
            # 边界层加权：接近最优退出层的样本更重要
            layer_idx = item["layer_idx"]
            optimal_exit = item["optimal_exit_layer"]
            distance_to_optimal = abs(layer_idx - optimal_exit)
            if distance_to_optimal < 5:
                w *= 2.0  # 边界样本加倍权重
            
            # 错误样本加权（防止假阳性）
            if not item["is_correct"]:
                w *= 1.5
            
            # OOD样本加权
            if item["is_ood"]:
                w *= 1.5
            
            weights.append(w)
        
        # 归一化
        total = sum(weights)
        self.weights = [w / total * len(weights) for w in weights]
    
    def __len__(self) -> int:
        return len(self.flat_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.flat_data[idx]
        
        return {
            "features": torch.tensor(item["features"], dtype=torch.float32),
            "layer_idx": torch.tensor(item["layer_idx"], dtype=torch.long),
            "is_correct": torch.tensor(float(item["is_correct"]), dtype=torch.float32),
            "is_ood": torch.tensor(float(item["is_ood"]), dtype=torch.float32),
            "can_exit": torch.tensor(float(item["can_exit"]), dtype=torch.float32),
            "optimal_exit_layer": torch.tensor(item["optimal_exit_layer"], dtype=torch.float32),
            "cognitive_load": torch.tensor(item["cognitive_load"], dtype=torch.float32),
            "difficulty": torch.tensor(item["difficulty"], dtype=torch.float32),
        }


class AdaptiveLoss(nn.Module):
    """
    自适应多任务损失
    
    组件：
    1. 退出决策损失：BCE with focal loss
    2. 置信度校准损失：ECE-aware
    3. 认知负荷回归损失：Huber loss
    4. OOD检测损失：BCE
    """
    
    def __init__(
        self,
        exit_weight: float = 1.0,
        confidence_weight: float = 1.0,
        cognitive_load_weight: float = 0.5,
        label_smoothing: float = 0.1,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.exit_weight = exit_weight
        self.confidence_weight = confidence_weight
        self.cognitive_load_weight = cognitive_load_weight
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        
        # 自适应权重（会在训练中调整）
        self.adaptive_weights = nn.Parameter(
            torch.ones(4), requires_grad=False
        )
    
    def focal_bce_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        """
        Focal BCE Loss
        
        对难分样本给予更高权重
        """
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** gamma
        return (focal_weight * bce).mean()
    
    def forward(
        self,
        signal,  # IntuitionSignal
        targets: Dict[str, torch.Tensor],
        layer_idx: int,
        total_layers: int = 36,
    ) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        """
        can_exit = targets["can_exit"]
        is_ood = targets["is_ood"]
        cognitive_load = targets["cognitive_load"]
        difficulty = targets["difficulty"]
        
        # 1. 退出决策损失（最重要）
        # p_confident高 → 应该退出（如果can_exit=True）
        exit_pred = signal.p_confident
        exit_loss = self.focal_bce_loss(exit_pred, can_exit, self.focal_gamma)
        
        # 2. 置信度校准损失
        # 置信度应该反映真实的退出安全性
        confidence_target = can_exit * (1.0 - difficulty * 0.3)  # 难度越高，置信度应该越低
        confidence_loss = F.mse_loss(signal.p_confident, confidence_target)
        
        # 3. 认知负荷回归损失
        # 使用Huber loss（对异常值鲁棒）
        cognitive_pred = 1.0 - signal.p_confident  # 简单近似：低置信度=高认知负荷
        cognitive_loss = F.smooth_l1_loss(cognitive_pred, cognitive_load)
        
        # 4. OOD检测损失
        ood_loss = F.binary_cross_entropy(signal.p_ood, is_ood)
        
        # 5. 幻觉检测损失
        # hallucination ≈ 低置信度 + 非OOD
        halluc_target = (1.0 - can_exit) * (1.0 - is_ood)
        halluc_loss = F.binary_cross_entropy(signal.p_hallucination, halluc_target)
        
        # 加权总损失
        total = (
            self.exit_weight * exit_loss +
            self.confidence_weight * confidence_loss +
            self.cognitive_load_weight * cognitive_loss +
            0.5 * ood_loss +
            0.5 * halluc_loss
        )
        
        return {
            "total": total,
            "exit": exit_loss,
            "confidence": confidence_loss,
            "cognitive_load": cognitive_loss,
            "ood": ood_loss,
            "hallucination": halluc_loss,
        }


class AdaptiveTrainer:
    """
    自适应训练器
    
    目标：退出精度 95%+
    """
    
    def __init__(
        self,
        config: AdaptiveTrainingConfig = None,
        intuition_config: IntuitionConfig = None,
    ):
        self.config = config or AdaptiveTrainingConfig()
        
        # 设备
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        logger.info(f"Using device: {self.device}")
        
        # 模型
        intuition_config = intuition_config or IntuitionConfig(dropout=self.config.dropout)
        self.model = IntuitionNetwork(intuition_config).to(self.device)
        
        # 损失函数
        self.loss_fn = AdaptiveLoss(
            exit_weight=self.config.exit_weight,
            confidence_weight=self.config.confidence_weight,
            cognitive_load_weight=self.config.cognitive_load_weight,
            label_smoothing=self.config.label_smoothing,
        )
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # 学习率调度
        self.scheduler = None  # 在train()中初始化
        
        # 训练状态
        self.best_val_loss = float('inf')
        self.best_exit_precision = 0.0
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": [], "exit_precision": [], "exit_recall": []}
    
    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """加载数据"""
        data_path = Path(self.config.data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found: {data_path}")
        
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        num_layers = raw_data.get("num_layers", 36)
        samples = raw_data.get("samples", [])
        
        logger.info(f"Loaded {len(samples)} tokens, {num_layers} layers")
        
        # 分割训练/验证
        val_size = int(len(samples) * self.config.val_split)
        train_samples = samples[val_size:]
        val_samples = samples[:val_size]
        
        train_dataset = AdaptiveDataset(train_samples, num_layers, compute_weights=True)
        val_dataset = AdaptiveDataset(val_samples, num_layers, compute_weights=False)
        
        # 使用加权采样器
        if train_dataset.weights:
            sampler = WeightedRandomSampler(
                weights=train_dataset.weights,
                num_samples=len(train_dataset),
                replacement=True,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                sampler=sampler,
                num_workers=0,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0,
            )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        loss_components = {"exit": 0.0, "confidence": 0.0, "cognitive_load": 0.0, "ood": 0.0}
        
        for batch in loader:
            features = batch["features"].to(self.device)
            layer_idx = batch["layer_idx"][0].item()
            
            # 前向传播
            signal = self.model(features, layer_idx)
            
            # 计算损失
            targets = {
                "can_exit": batch["can_exit"].to(self.device),
                "is_ood": batch["is_ood"].to(self.device),
                "cognitive_load": batch["cognitive_load"].to(self.device),
                "difficulty": batch["difficulty"].to(self.device),
            }
            
            losses = self.loss_fn(signal, targets, layer_idx)
            
            # 反向传播
            self.optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += losses["total"].item()
            for k in loss_components:
                if k in losses:
                    loss_components[k] += losses[k].item()
        
        n_batches = len(loader)
        return {
            "total": total_loss / n_batches,
            **{k: v / n_batches for k, v in loss_components.items()},
        }
    
    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        
        # 精确率和召回率统计
        true_positives = 0   # 正确预测退出且确实可以退出
        false_positives = 0  # 预测退出但不能退出
        false_negatives = 0  # 可以退出但没预测退出
        true_negatives = 0   # 不能退出且没预测退出
        
        total_samples = 0
        
        for batch in loader:
            features = batch["features"].to(self.device)
            layer_idx = batch["layer_idx"][0].item()
            
            signal = self.model(features, layer_idx)
            
            targets = {
                "can_exit": batch["can_exit"].to(self.device),
                "is_ood": batch["is_ood"].to(self.device),
                "cognitive_load": batch["cognitive_load"].to(self.device),
                "difficulty": batch["difficulty"].to(self.device),
            }
            
            losses = self.loss_fn(signal, targets, layer_idx)
            total_loss += losses["total"].item()
            
            # 计算退出决策统计
            exit_pred = (signal.p_confident > 0.5).float()
            can_exit = targets["can_exit"]
            
            true_positives += ((exit_pred == 1) & (can_exit == 1)).sum().item()
            false_positives += ((exit_pred == 1) & (can_exit == 0)).sum().item()
            false_negatives += ((exit_pred == 0) & (can_exit == 1)).sum().item()
            true_negatives += ((exit_pred == 0) & (can_exit == 0)).sum().item()
            
            total_samples += features.shape[0]
        
        n_batches = len(loader)
        
        # 计算指标
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        accuracy = (true_positives + true_negatives) / max(total_samples, 1)
        
        # 退出率
        exit_rate = (true_positives + false_positives) / max(total_samples, 1)
        
        return {
            "total": total_loss / n_batches,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "exit_rate": exit_rate,
        }
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """保存检查点"""
        Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "history": self.history,
            "best_exit_precision": self.best_exit_precision,
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
    
    def train(self) -> Dict[str, List]:
        """完整训练流程"""
        train_loader, val_loader = self.load_data()
        
        # 初始化学习率调度器
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=self.config.min_lr,
        )
        
        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        logger.info(f"Target: Exit Precision >= 95%")
        
        for epoch in range(self.config.epochs):
            # 训练
            train_metrics = self.train_epoch(train_loader)
            
            # 验证
            val_metrics = self.validate(val_loader)
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history["train_loss"].append(train_metrics["total"])
            self.history["val_loss"].append(val_metrics["total"])
            self.history["exit_precision"].append(val_metrics["precision"])
            self.history["exit_recall"].append(val_metrics["recall"])
            
            # 日志
            logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"Train Loss: {train_metrics['total']:.4f} | "
                f"Val Loss: {val_metrics['total']:.4f} | "
                f"Precision: {val_metrics['precision']*100:.2f}% | "
                f"Recall: {val_metrics['recall']*100:.2f}% | "
                f"F1: {val_metrics['f1']*100:.2f}% | "
                f"LR: {current_lr:.2e}"
            )
            
            # 检查是否达到目标
            if val_metrics["precision"] >= 0.95:
                logger.info(f"🎉 达到目标精度 {val_metrics['precision']*100:.2f}%!")
            
            # 保存最佳模型（基于precision）
            if val_metrics["precision"] > self.best_exit_precision:
                self.best_exit_precision = val_metrics["precision"]
                self.patience_counter = 0
                self.save_checkpoint(
                    f"{self.config.save_dir}/intuition_network_best_v9.pt",
                    is_best=True
                )
            else:
                self.patience_counter += 1
            
            # 早停
            if self.patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # 保存最终模型
        self.save_checkpoint(f"{self.config.save_dir}/intuition_network_final_v9.pt")
        
        logger.info(f"Training complete. Best Precision: {self.best_exit_precision*100:.2f}%")
        
        return self.history


def train_adaptive_model(
    data_path: str = "sedac_v9_augmented_data.json",
    epochs: int = 100,
    batch_size: int = 128,
):
    """便捷训练函数"""
    config = AdaptiveTrainingConfig(
        data_path=data_path,
        epochs=epochs,
        batch_size=batch_size,
    )
    
    trainer = AdaptiveTrainer(config)
    history = trainer.train()
    
    return history


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_adaptive_model()
