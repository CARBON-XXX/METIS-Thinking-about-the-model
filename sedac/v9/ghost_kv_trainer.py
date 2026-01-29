"""
SEDAC V9.0 - Ghost KV 训练器

训练TinyMLP预测KV Cache，实现5%计算量的KV生成

训练流程:
1. 冻结大模型
2. 只训练Ghost MLP
3. MSE损失对齐真实KV
4. 可选：余弦相似度损失保持方向
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import logging
import time
import os

logger = logging.getLogger(__name__)

from sedac.v9.ghost_kv import GhostKVManager, GhostKVConfig, create_ghost_kv_manager


@dataclass
class GhostKVTrainingConfig:
    """Ghost KV训练配置"""
    # 模型参数
    hidden_size: int = 4096
    num_heads: int = 32
    head_dim: int = 128
    num_layers: int = 32
    
    # 训练参数
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # 损失权重
    mse_weight: float = 1.0
    cosine_weight: float = 0.1
    
    # 保存
    save_dir: str = "./checkpoints/ghost_kv"
    save_every: int = 1000
    log_every: int = 100


class KVDataset(Dataset):
    """
    KV Cache训练数据集
    
    存储真实的hidden states和对应的KV Cache
    """
    
    def __init__(
        self,
        hidden_states: List[torch.Tensor],  # [batch, seq_len, hidden]
        keys: List[torch.Tensor],           # [batch, num_heads, seq_len, head_dim]
        values: List[torch.Tensor],
        layer_indices: List[int],
    ):
        self.hidden_states = hidden_states
        self.keys = keys
        self.values = values
        self.layer_indices = layer_indices
    
    def __len__(self) -> int:
        return len(self.hidden_states)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "hidden_states": self.hidden_states[idx],
            "key": self.keys[idx],
            "value": self.values[idx],
            "layer_idx": torch.tensor(self.layer_indices[idx]),
        }


class GhostKVTrainer:
    """
    Ghost KV训练器
    
    冻结大模型，训练TinyMLP预测KV
    """
    
    def __init__(
        self,
        config: GhostKVTrainingConfig,
        ghost_manager: Optional[GhostKVManager] = None,
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建或使用Ghost管理器
        if ghost_manager is None:
            self.ghost_manager = create_ghost_kv_manager(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                num_layers=config.num_layers,
                strategy="ghost",
            ).to(self.device)
        else:
            self.ghost_manager = ghost_manager.to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.ghost_manager.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # 学习率调度器
        self.scheduler = None  # 在train()中初始化
        
        # 训练状态
        self.global_step = 0
        self.best_loss = float('inf')
        self.losses: List[float] = []
        
        # 创建保存目录
        os.makedirs(config.save_dir, exist_ok=True)
    
    def compute_loss(
        self,
        pred_key: torch.Tensor,
        pred_value: torch.Tensor,
        target_key: torch.Tensor,
        target_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算损失
        
        包含:
        1. MSE损失（主要）
        2. 余弦相似度损失（保持方向）
        """
        cfg = self.config
        
        # MSE损失
        key_mse = F.mse_loss(pred_key, target_key)
        value_mse = F.mse_loss(pred_value, target_value)
        mse_loss = (key_mse + value_mse) * cfg.mse_weight
        
        # 余弦相似度损失
        key_cos = 1 - F.cosine_similarity(
            pred_key.flatten(2), target_key.flatten(2), dim=-1
        ).mean()
        value_cos = 1 - F.cosine_similarity(
            pred_value.flatten(2), target_value.flatten(2), dim=-1
        ).mean()
        cosine_loss = (key_cos + value_cos) * cfg.cosine_weight
        
        total_loss = mse_loss + cosine_loss
        
        metrics = {
            "total_loss": total_loss.item(),
            "mse_loss": mse_loss.item(),
            "cosine_loss": cosine_loss.item(),
            "key_mse": key_mse.item(),
            "value_mse": value_mse.item(),
        }
        
        return total_loss, metrics
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        训练一步
        """
        self.ghost_manager.train()
        self.optimizer.zero_grad()
        
        hidden_states = batch["hidden_states"].to(self.device)
        target_key = batch["key"].to(self.device)
        target_value = batch["value"].to(self.device)
        layer_idx = batch["layer_idx"][0].item()  # 假设batch内layer相同
        
        # 生成Ghost KV
        pred_key, pred_value = self.ghost_manager.generate_ghost_kv(
            hidden_states, layer_idx
        )
        
        # 计算损失
        loss, metrics = self.compute_loss(pred_key, pred_value, target_key, target_value)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.ghost_manager.parameters(),
            self.config.max_grad_norm
        )
        
        # 更新
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        
        self.global_step += 1
        self.losses.append(metrics["total_loss"])
        
        return metrics
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
    ):
        """
        训练循环
        """
        cfg = self.config
        total_steps = len(train_dataloader) * cfg.num_epochs
        
        # 初始化调度器
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=cfg.learning_rate,
            total_steps=total_steps,
            pct_start=cfg.warmup_steps / total_steps,
        )
        
        logger.info(f"Starting Ghost KV training")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Epochs: {cfg.num_epochs}")
        logger.info(f"  Batch size: {cfg.batch_size}")
        
        for epoch in range(cfg.num_epochs):
            epoch_losses = []
            epoch_start = time.time()
            
            for batch_idx, batch in enumerate(train_dataloader):
                metrics = self.train_step(batch)
                epoch_losses.append(metrics["total_loss"])
                
                # 日志
                if self.global_step % cfg.log_every == 0:
                    avg_loss = sum(self.losses[-cfg.log_every:]) / cfg.log_every
                    lr = self.optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"Step {self.global_step}: loss={avg_loss:.4f}, "
                        f"mse={metrics['mse_loss']:.4f}, cos={metrics['cosine_loss']:.4f}, "
                        f"lr={lr:.2e}"
                    )
                
                # 保存
                if self.global_step % cfg.save_every == 0:
                    self.save_checkpoint(f"step_{self.global_step}.pt")
            
            # Epoch结束
            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            epoch_time = time.time() - epoch_start
            
            logger.info(
                f"Epoch {epoch+1}/{cfg.num_epochs}: "
                f"loss={epoch_loss:.4f}, time={epoch_time:.1f}s"
            )
            
            # 验证
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                logger.info(f"Validation loss: {val_loss:.4f}")
                
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint("best.pt")
        
        # 保存最终模型
        self.save_checkpoint("final.pt")
        logger.info("Training complete!")
    
    def validate(self, dataloader: DataLoader) -> float:
        """验证"""
        self.ghost_manager.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                hidden_states = batch["hidden_states"].to(self.device)
                target_key = batch["key"].to(self.device)
                target_value = batch["value"].to(self.device)
                layer_idx = batch["layer_idx"][0].item()
                
                pred_key, pred_value = self.ghost_manager.generate_ghost_kv(
                    hidden_states, layer_idx
                )
                
                loss, _ = self.compute_loss(pred_key, pred_value, target_key, target_value)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        path = os.path.join(self.config.save_dir, filename)
        torch.save({
            "ghost_manager": self.ghost_manager.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "config": self.config,
        }, path)
        logger.info(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.ghost_manager.load_state_dict(checkpoint["ghost_manager"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint["scheduler"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]
        logger.info(f"Loaded checkpoint: {path}")


class KVDataCollector:
    """
    KV数据收集器
    
    从真实模型中收集KV Cache数据用于训练Ghost KV
    """
    
    def __init__(
        self,
        model: nn.Module,
        hidden_size: int = 4096,
        num_heads: int = 32,
        head_dim: int = 128,
    ):
        self.model = model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # 收集的数据
        self.hidden_states_list: List[torch.Tensor] = []
        self.keys_list: List[torch.Tensor] = []
        self.values_list: List[torch.Tensor] = []
        self.layer_indices_list: List[int] = []
        
        # 注册hook
        self.hooks = []
    
    def _make_hook(self, layer_idx: int):
        """创建hook函数"""
        def hook(module, input, output):
            # 尝试提取hidden states和KV
            if isinstance(input, tuple) and len(input) > 0:
                hidden = input[0]
                if hasattr(hidden, 'shape') and len(hidden.shape) == 3:
                    self.hidden_states_list.append(hidden.detach().cpu())
                    self.layer_indices_list.append(layer_idx)
                    
                    # 尝试提取KV
                    if isinstance(output, tuple) and len(output) > 1:
                        kv = output[1]
                        if kv is not None and isinstance(kv, tuple) and len(kv) == 2:
                            self.keys_list.append(kv[0].detach().cpu())
                            self.values_list.append(kv[1].detach().cpu())
        
        return hook
    
    def register_hooks(self):
        """注册hook到模型层"""
        # 查找层
        layers = None
        for attr in ['model.layers', 'layers', 'transformer.h', 'h']:
            try:
                layers = eval(f"self.model.{attr}")
                break
            except:
                pass
        
        if layers is None:
            logger.warning("Could not find transformer layers for hooking")
            return
        
        for idx, layer in enumerate(layers):
            hook = layer.register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)
        
        logger.info(f"Registered hooks on {len(self.hooks)} layers")
    
    def remove_hooks(self):
        """移除hook"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def collect(
        self,
        dataloader: DataLoader,
        max_samples: int = 10000,
    ) -> KVDataset:
        """
        收集KV数据
        """
        self.register_hooks()
        self.model.eval()
        
        try:
            with torch.no_grad():
                for batch in dataloader:
                    if len(self.hidden_states_list) >= max_samples:
                        break
                    
                    # 前向传播触发hook
                    if isinstance(batch, dict):
                        self.model(**batch)
                    else:
                        self.model(batch)
        finally:
            self.remove_hooks()
        
        logger.info(f"Collected {len(self.hidden_states_list)} samples")
        
        return KVDataset(
            self.hidden_states_list[:max_samples],
            self.keys_list[:max_samples],
            self.values_list[:max_samples],
            self.layer_indices_list[:max_samples],
        )
    
    def clear(self):
        """清除收集的数据"""
        self.hidden_states_list.clear()
        self.keys_list.clear()
        self.values_list.clear()
        self.layer_indices_list.clear()


def create_synthetic_kv_dataset(
    num_samples: int = 10000,
    hidden_size: int = 512,
    num_heads: int = 8,
    head_dim: int = 64,
    seq_len: int = 64,
    num_layers: int = 12,
) -> KVDataset:
    """
    创建合成KV数据集（用于测试）
    """
    hidden_states = []
    keys = []
    values = []
    layer_indices = []
    
    # 创建随机KV投影（模拟真实权重）
    wk = torch.randn(hidden_size, num_heads * head_dim) * 0.02
    wv = torch.randn(hidden_size, num_heads * head_dim) * 0.02
    
    for i in range(num_samples):
        layer_idx = i % num_layers
        
        # 随机hidden states
        hidden = torch.randn(1, seq_len, hidden_size)
        
        # 计算"真实"KV
        key = torch.matmul(hidden, wk).view(1, seq_len, num_heads, head_dim).transpose(1, 2)
        value = torch.matmul(hidden, wv).view(1, seq_len, num_heads, head_dim).transpose(1, 2)
        
        hidden_states.append(hidden.squeeze(0))
        keys.append(key.squeeze(0))
        values.append(value.squeeze(0))
        layer_indices.append(layer_idx)
    
    return KVDataset(hidden_states, keys, values, layer_indices)


def demo_ghost_kv_training():
    """演示Ghost KV训练"""
    print("=" * 70)
    print("SEDAC V9.0 Ghost KV Training Demo")
    print("=" * 70)
    
    # 配置
    config = GhostKVTrainingConfig(
        hidden_size=512,
        num_heads=8,
        head_dim=64,
        num_layers=12,
        batch_size=4,
        learning_rate=1e-4,
        num_epochs=2,
        log_every=10,
        save_dir="./cache/ghost_kv",
    )
    
    print(f"\n配置:")
    print(f"  Hidden Size: {config.hidden_size}")
    print(f"  Num Heads: {config.num_heads}")
    print(f"  Num Layers: {config.num_layers}")
    
    # 创建合成数据集
    print(f"\n创建合成数据集...")
    dataset = create_synthetic_kv_dataset(
        num_samples=200,
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        num_layers=config.num_layers,
    )
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    print(f"  样本数: {len(dataset)}")
    
    # 创建训练器
    trainer = GhostKVTrainer(config)
    
    print(f"\nGhost管理器参数量: {sum(p.numel() for p in trainer.ghost_manager.parameters()):,}")
    
    # 训练
    print(f"\n开始训练...")
    trainer.train(dataloader)
    
    # 测试
    print(f"\n测试Ghost KV生成:")
    trainer.ghost_manager.eval()
    
    sample = dataset[0]
    hidden = sample["hidden_states"].unsqueeze(0).to(trainer.device)
    target_key = sample["key"].unsqueeze(0).to(trainer.device)
    target_value = sample["value"].unsqueeze(0).to(trainer.device)
    
    with torch.no_grad():
        pred_key, pred_value = trainer.ghost_manager.generate_ghost_kv(hidden, 0)
    
    key_mse = F.mse_loss(pred_key, target_key).item()
    key_cos = F.cosine_similarity(pred_key.flatten(), target_key.flatten(), dim=0).item()
    
    print(f"  Key MSE: {key_mse:.6f}")
    print(f"  Key Cosine Similarity: {key_cos:.4f}")
    
    print("\n" + "=" * 70)
    print("Ghost KV Training: TinyMLP学会预测KV Cache")
    print("=" * 70)


if __name__ == "__main__":
    demo_ghost_kv_training()
