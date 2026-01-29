"""
SEDAC V9.0 - Intuition Network Trainer

训练直觉网络学习"认知注意力"：
- 什么时候该快（低熵 → Early Exit）
- 什么时候该慢（高熵 → Full Inference）
- 什么时候该求助（风险 → Intervention）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import logging

from sedac.v8.intuition_network import (
    IntuitionNetwork,
    IntuitionConfig,
    IntuitionLoss,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """训练配置"""
    # 数据
    data_path: str = "sedac_v8_training_data.json"
    val_split: float = 0.1
    
    # 训练参数
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # 早停
    patience: int = 10
    min_delta: float = 1e-4
    
    # 保存
    save_dir: str = "checkpoints"
    save_best: bool = True
    
    # 设备
    device: str = "auto"


class SEDACDataset(Dataset):
    """
    SEDAC训练数据集
    
    数据格式 (sedac_v8_training_data.json):
    {
        "num_layers": 36,
        "num_features": 8,
        "samples": [
            {
                "token_idx": int,
                "features_per_layer": [[8 features] * 36 layers],
                "is_correct": bool,
                "is_ood": bool,
                "optimal_exit_layer": int,
                "final_entropy": float,
                "metadata": {...}
            }
        ]
    }
    """
    
    def __init__(self, samples: List[Dict], num_layers: int = 36):
        self.samples = samples
        self.num_layers = num_layers
        
        # 展平数据: 每个(sample, layer)组合作为一个训练样本
        self.flat_data = []
        for sample in samples:
            features_per_layer = sample["features_per_layer"]
            is_correct = sample.get("is_correct", True)
            is_ood = sample.get("is_ood", False)
            optimal_exit = sample.get("optimal_exit_layer", num_layers)
            final_entropy = sample.get("final_entropy", 1.0)
            
            for layer_idx, features in enumerate(features_per_layer):
                self.flat_data.append({
                    "features": features,
                    "layer_idx": layer_idx,
                    "is_correct": is_correct,
                    "is_ood": is_ood,
                    "optimal_exit_layer": optimal_exit,
                    "final_entropy": final_entropy,
                    # 是否可以在此层安全退出
                    "can_exit_here": layer_idx >= optimal_exit and is_correct,
                })
        
        logger.info(f"Created dataset with {len(self.flat_data)} samples "
                   f"({len(samples)} tokens × {num_layers} layers)")
        
    def __len__(self) -> int:
        return len(self.flat_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.flat_data[idx]
        
        # 8维特征 (已经在数据中)
        features = torch.tensor(item["features"], dtype=torch.float32)
        
        # 标签
        return {
            "features": features,
            "layer_idx": torch.tensor(item["layer_idx"], dtype=torch.long),
            "is_correct": torch.tensor(float(item["is_correct"]), dtype=torch.float32),
            "is_ood": torch.tensor(float(item["is_ood"]), dtype=torch.float32),
            "optimal_exit_layer": torch.tensor(item["optimal_exit_layer"], dtype=torch.float32),
            "can_exit_here": torch.tensor(float(item["can_exit_here"]), dtype=torch.float32),
            "final_entropy": torch.tensor(item["final_entropy"], dtype=torch.float32),
        }


class IntuitionTrainer:
    """直觉网络训练器"""
    
    def __init__(
        self,
        config: TrainingConfig = None,
        intuition_config: IntuitionConfig = None,
    ):
        self.config = config or TrainingConfig()
        self.intuition_config = intuition_config or IntuitionConfig()
        
        # 设备
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        logger.info(f"Using device: {self.device}")
        
        # 模型
        self.model = IntuitionNetwork(self.intuition_config).to(self.device)
        self.loss_fn = IntuitionLoss(self.intuition_config)
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # 学习率调度
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 训练状态
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.history: List[Dict] = []
        
    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """加载数据"""
        data_path = Path(self.config.data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found: {data_path}")
        
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        # 解析数据结构
        num_layers = raw_data.get("num_layers", 36)
        samples = raw_data.get("samples", [])
        
        logger.info(f"Loaded {len(samples)} tokens, {num_layers} layers")
        
        # 分割训练/验证 (按token分割，不是按layer)
        val_size = int(len(samples) * self.config.val_split)
        train_samples = samples[val_size:]
        val_samples = samples[:val_size]
        
        train_dataset = SEDACDataset(train_samples, num_layers)
        val_dataset = SEDACDataset(val_samples, num_layers)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Windows兼容
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
        loss_components = {"speed": 0.0, "accuracy": 0.0, "calibration": 0.0, "ood": 0.0}
        
        for batch in loader:
            features = batch["features"].to(self.device)
            layer_indices = batch["layer_idx"].to(self.device)
            
            # 使用batch中的平均layer_idx（或第一个）
            layer_idx = layer_indices[0].item()
            
            # 前向传播
            signal = self.model(features, layer_idx)
            
            # 计算损失
            targets = {
                "is_correct": batch["is_correct"].to(self.device),
                "is_ood": batch["is_ood"].to(self.device),
                "optimal_exit_layer": batch["optimal_exit_layer"].to(self.device),
            }
            
            losses = self.loss_fn(signal, targets, layer_idx, total_layers=36)
            
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
        correct_exits = 0
        total_exits = 0
        total_samples = 0
        
        for batch in loader:
            features = batch["features"].to(self.device)
            layer_idx = batch["layer_idx"][0].item()
            
            signal = self.model(features, layer_idx)
            
            targets = {
                "is_correct": batch["is_correct"].to(self.device),
                "is_ood": batch["is_ood"].to(self.device),
                "optimal_exit_layer": batch["optimal_exit_layer"].to(self.device),
            }
            
            losses = self.loss_fn(signal, targets, layer_idx, total_layers=36)
            total_loss += losses["total"].item()
            
            # 计算早退精度: 模型预测EXIT时，是否真的可以安全退出
            decisions = self.model.get_decision(signal)
            exit_mask = decisions == 1  # EXIT
            can_exit = batch["can_exit_here"].to(self.device)
            
            # 正确退出 = 模型说EXIT且确实可以退出
            correct_exits += ((exit_mask) & (can_exit > 0.5)).sum().item()
            total_exits += exit_mask.sum().item()
            total_samples += features.shape[0]
        
        n_batches = len(loader)
        exit_precision = correct_exits / max(total_exits, 1)
        exit_rate = total_exits / max(total_samples, 1)
        
        return {
            "total": total_loss / n_batches,
            "exit_precision": exit_precision,
            "exit_rate": exit_rate,
        }
    
    def train(self) -> Dict[str, List]:
        """完整训练流程"""
        train_loader, val_loader = self.load_data()
        
        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        for epoch in range(self.config.epochs):
            # 训练
            train_metrics = self.train_epoch(train_loader)
            
            # 验证
            val_metrics = self.validate(val_loader)
            
            # 学习率调度
            self.scheduler.step(val_metrics["total"])
            
            # 记录
            self.history.append({
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "lr": self.optimizer.param_groups[0]["lr"],
            })
            
            # 日志
            logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"Train Loss: {train_metrics['total']:.4f} | "
                f"Val Loss: {val_metrics['total']:.4f} | "
                f"Exit Precision: {val_metrics['exit_precision']:.2%}"
            )
            
            # 早停检查
            if val_metrics["total"] < self.best_loss - self.config.min_delta:
                self.best_loss = val_metrics["total"]
                self.patience_counter = 0
                
                if self.config.save_best:
                    self.save_checkpoint("best")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # 保存最终模型
        self.save_checkpoint("final")
        
        return {
            "train_loss": [h["train"]["total"] for h in self.history],
            "val_loss": [h["val"]["total"] for h in self.history],
            "exit_precision": [h["val"]["exit_precision"] for h in self.history],
        }
    
    def save_checkpoint(self, name: str):
        """保存检查点"""
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.intuition_config,
            "best_loss": self.best_loss,
            "history": self.history,
        }
        
        path = save_dir / f"intuition_network_{name}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_loss = checkpoint.get("best_loss", float('inf'))
        self.history = checkpoint.get("history", [])
        logger.info(f"Loaded checkpoint: {path}")


def main():
    """训练入口"""
    config = TrainingConfig(
        data_path="sedac_v8_training_data.json",
        epochs=100,
        batch_size=32,
        learning_rate=1e-3,
    )
    
    trainer = IntuitionTrainer(config)
    history = trainer.train()
    
    print("\n训练完成!")
    print(f"最佳验证损失: {trainer.best_loss:.4f}")
    print(f"最终早退精度: {history['exit_precision'][-1]:.2%}")


if __name__ == "__main__":
    main()
