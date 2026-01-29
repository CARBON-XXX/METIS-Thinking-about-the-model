"""
SEDAC V9.0 Production Trainer

Ghost KV 生成器的生产级训练器
使用真实模型进行知识蒸馏
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
import logging
import time
import json
from pathlib import Path
from tqdm import tqdm

from .config import ProductionConfig, ModelConfig
from .engine import GhostKVGenerator
from .metrics import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    output_dir: str = "./ghost_kv_checkpoints"
    resume_from: Optional[str] = None
    
    fp16: bool = True
    bf16: bool = False
    
    similarity_target: float = 0.98
    mse_weight: float = 1.0
    cosine_weight: float = 0.5


@dataclass
class TrainingState:
    """训练状态"""
    global_step: int = 0
    epoch: int = 0
    best_similarity: float = 0.0
    train_loss: float = 0.0
    eval_loss: float = 0.0
    eval_similarity: float = 0.0


class KVDistillationDataset(Dataset):
    """
    KV Cache 蒸馏数据集
    
    从真实模型收集 KV Cache 用于训练 Ghost KV
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        texts: List[str],
        config: ProductionConfig,
        max_seq_len: int = 512,
        skip_layers: int = 4,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.max_seq_len = max_seq_len
        self.skip_layers = skip_layers
        
        self.samples: List[Dict[str, torch.Tensor]] = []
        self._collect_samples(texts)
    
    @torch.no_grad()
    def _collect_samples(self, texts: List[str]) -> None:
        """收集训练样本"""
        logger.info(f"Collecting {len(texts)} samples for KV distillation...")
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        for text in tqdm(texts, desc="Collecting KV samples"):
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_seq_len,
                truncation=True,
                padding="max_length",
            ).to(device)
            
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                use_cache=True,
            )
            
            hidden_states = outputs.hidden_states
            past_key_values = outputs.past_key_values
            
            for layer_idx in range(self.config.sedac.min_exit_layer, 
                                   len(hidden_states) - self.skip_layers - 1):
                input_hidden = hidden_states[layer_idx].cpu()
                
                target_kvs = []
                for skip_idx in range(self.skip_layers):
                    target_layer = layer_idx + skip_idx + 1
                    if target_layer < len(past_key_values):
                        k, v = past_key_values[target_layer]
                        target_kvs.append((k.cpu(), v.cpu()))
                
                if target_kvs:
                    self.samples.append({
                        "input_hidden": input_hidden.squeeze(0),
                        "target_kvs": target_kvs,
                        "layer_idx": layer_idx,
                    })
        
        logger.info(f"Collected {len(self.samples)} training samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def collate_kv_samples(samples: List[Dict]) -> Dict[str, Any]:
    """自定义 collate 函数"""
    input_hidden = torch.stack([s["input_hidden"] for s in samples])
    
    num_skip = len(samples[0]["target_kvs"])
    target_kvs = []
    
    for skip_idx in range(num_skip):
        k_batch = torch.stack([s["target_kvs"][skip_idx][0] for s in samples])
        v_batch = torch.stack([s["target_kvs"][skip_idx][1] for s in samples])
        target_kvs.append((k_batch, v_batch))
    
    return {
        "input_hidden": input_hidden,
        "target_kvs": target_kvs,
    }


class GhostKVTrainer:
    """
    Ghost KV 生产级训练器
    
    特性:
    - 真实模型蒸馏
    - 混合精度训练
    - 梯度累积
    - 检查点保存与恢复
    - 完整的指标监控
    """
    
    def __init__(
        self,
        ghost_kv: GhostKVGenerator,
        train_config: TrainingConfig,
        prod_config: ProductionConfig,
    ):
        self.ghost_kv = ghost_kv
        self.train_config = train_config
        self.prod_config = prod_config
        
        self.device = torch.device(prod_config.device)
        self.ghost_kv = self.ghost_kv.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.ghost_kv.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
        
        self.scaler = None
        if train_config.fp16 and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.state = TrainingState()
        self.metrics = MetricsCollector()
        
        self.output_dir = Path(train_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_loss(
        self,
        pred_kvs: List[Tuple[torch.Tensor, torch.Tensor]],
        target_kvs: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算蒸馏损失"""
        total_mse = 0.0
        total_cosine = 0.0
        
        for (pk, pv), (tk, tv) in zip(pred_kvs, target_kvs):
            tk = tk.to(pk.device, pk.dtype)
            tv = tv.to(pv.device, pv.dtype)
            
            k_mse = F.mse_loss(pk, tk)
            v_mse = F.mse_loss(pv, tv)
            
            k_cos = 1 - F.cosine_similarity(
                pk.flatten(2), tk.flatten(2), dim=-1
            ).mean()
            v_cos = 1 - F.cosine_similarity(
                pv.flatten(2), tv.flatten(2), dim=-1
            ).mean()
            
            total_mse += k_mse + v_mse
            total_cosine += k_cos + v_cos
        
        num_layers = len(pred_kvs)
        mse_loss = total_mse / num_layers
        cosine_loss = total_cosine / num_layers
        
        loss = (
            self.train_config.mse_weight * mse_loss +
            self.train_config.cosine_weight * cosine_loss
        )
        
        return loss, {
            "mse_loss": mse_loss.item(),
            "cosine_loss": cosine_loss.item(),
            "total_loss": loss.item(),
        }
    
    def compute_similarity(
        self,
        pred_kvs: List[Tuple[torch.Tensor, torch.Tensor]],
        target_kvs: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> float:
        """计算输出相似度"""
        similarities = []
        
        for (pk, pv), (tk, tv) in zip(pred_kvs, target_kvs):
            tk = tk.to(pk.device, pk.dtype)
            tv = tv.to(pv.device, pv.dtype)
            
            k_sim = F.cosine_similarity(pk.flatten(), tk.flatten(), dim=0).item()
            v_sim = F.cosine_similarity(pv.flatten(), tv.flatten(), dim=0).item()
            similarities.extend([k_sim, v_sim])
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """单步训练"""
        self.ghost_kv.train()
        
        input_hidden = batch["input_hidden"].to(self.device)
        target_kvs = batch["target_kvs"]
        
        num_skip = len(target_kvs)
        
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                pred_kvs = self.ghost_kv(input_hidden, num_skip)
                loss, metrics = self.compute_loss(pred_kvs, target_kvs)
            
            scaled_loss = loss / self.train_config.gradient_accumulation_steps
            self.scaler.scale(scaled_loss).backward()
        else:
            pred_kvs = self.ghost_kv(input_hidden, num_skip)
            loss, metrics = self.compute_loss(pred_kvs, target_kvs)
            
            scaled_loss = loss / self.train_config.gradient_accumulation_steps
            scaled_loss.backward()
        
        return metrics
    
    def optimizer_step(self) -> None:
        """优化器步骤"""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.ghost_kv.parameters(),
                self.train_config.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(
                self.ghost_kv.parameters(),
                self.train_config.max_grad_norm
            )
            self.optimizer.step()
        
        self.optimizer.zero_grad()
    
    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """评估"""
        self.ghost_kv.eval()
        
        total_loss = 0.0
        total_similarity = 0.0
        num_batches = 0
        
        for batch in eval_loader:
            input_hidden = batch["input_hidden"].to(self.device)
            target_kvs = batch["target_kvs"]
            num_skip = len(target_kvs)
            
            pred_kvs = self.ghost_kv(input_hidden, num_skip)
            _, metrics = self.compute_loss(pred_kvs, target_kvs)
            similarity = self.compute_similarity(pred_kvs, target_kvs)
            
            total_loss += metrics["total_loss"]
            total_similarity += similarity
            num_batches += 1
        
        return {
            "eval_loss": total_loss / num_batches,
            "eval_similarity": total_similarity / num_batches,
        }
    
    def save_checkpoint(self, name: str = "checkpoint") -> None:
        """保存检查点"""
        path = self.output_dir / f"{name}.pt"
        
        checkpoint = {
            "model_state_dict": self.ghost_kv.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "state": {
                "global_step": self.state.global_step,
                "epoch": self.state.epoch,
                "best_similarity": self.state.best_similarity,
            },
            "train_config": self.train_config,
        }
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.ghost_kv.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        state_dict = checkpoint.get("state", {})
        self.state.global_step = state_dict.get("global_step", 0)
        self.state.epoch = state_dict.get("epoch", 0)
        self.state.best_similarity = state_dict.get("best_similarity", 0.0)
        
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        logger.info(f"Checkpoint loaded from {path}")
    
    def train(
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """
        训练主循环
        
        Returns:
            训练结果统计
        """
        config = self.train_config
        total_steps = len(train_loader) * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return max(0.1, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        logger.info("=" * 60)
        logger.info("Starting Ghost KV Training")
        logger.info(f"Total epochs: {config.num_epochs}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Warmup steps: {warmup_steps}")
        logger.info(f"Target similarity: {config.similarity_target}")
        logger.info("=" * 60)
        
        train_start = time.time()
        
        for epoch in range(config.num_epochs):
            self.state.epoch = epoch
            epoch_loss = 0.0
            epoch_steps = 0
            
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
            
            for batch_idx, batch in enumerate(progress):
                metrics = self.train_step(batch)
                epoch_loss += metrics["total_loss"]
                epoch_steps += 1
                
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    self.optimizer_step()
                    scheduler.step()
                    self.state.global_step += 1
                
                if self.state.global_step % config.logging_steps == 0:
                    avg_loss = epoch_loss / epoch_steps
                    progress.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    })
                
                if eval_loader and self.state.global_step % config.eval_steps == 0:
                    eval_metrics = self.evaluate(eval_loader)
                    self.state.eval_loss = eval_metrics["eval_loss"]
                    self.state.eval_similarity = eval_metrics["eval_similarity"]
                    
                    logger.info(
                        f"Step {self.state.global_step}: "
                        f"eval_loss={eval_metrics['eval_loss']:.4f}, "
                        f"similarity={eval_metrics['eval_similarity']:.4f}"
                    )
                    
                    if eval_metrics["eval_similarity"] > self.state.best_similarity:
                        self.state.best_similarity = eval_metrics["eval_similarity"]
                        self.save_checkpoint("best")
                        logger.info(f"New best similarity: {self.state.best_similarity:.4f}")
                
                if self.state.global_step % config.save_steps == 0:
                    self.save_checkpoint(f"step_{self.state.global_step}")
            
            self.state.train_loss = epoch_loss / epoch_steps
            logger.info(f"Epoch {epoch+1} completed. Avg loss: {self.state.train_loss:.4f}")
        
        train_time = time.time() - train_start
        
        self.save_checkpoint("final")
        
        results = {
            "total_steps": self.state.global_step,
            "train_time_seconds": train_time,
            "final_train_loss": self.state.train_loss,
            "best_similarity": self.state.best_similarity,
            "target_achieved": self.state.best_similarity >= config.similarity_target,
        }
        
        logger.info("=" * 60)
        logger.info("Training Complete")
        logger.info(f"Best similarity: {self.state.best_similarity:.4f}")
        logger.info(f"Target: {config.similarity_target}")
        logger.info(f"Status: {'✅ ACHIEVED' if results['target_achieved'] else '⚠️ NOT ACHIEVED'}")
        logger.info("=" * 60)
        
        return results


def train_ghost_kv_from_model(
    teacher_model: nn.Module,
    tokenizer,
    train_texts: List[str],
    eval_texts: Optional[List[str]] = None,
    prod_config: Optional[ProductionConfig] = None,
    train_config: Optional[TrainingConfig] = None,
) -> Tuple[GhostKVGenerator, Dict[str, Any]]:
    """
    便捷函数：从真实模型训练 Ghost KV
    
    Args:
        teacher_model: 教师模型
        tokenizer: tokenizer
        train_texts: 训练文本
        eval_texts: 评估文本
        prod_config: 生产配置
        train_config: 训练配置
    
    Returns:
        训练好的 Ghost KV 和训练结果
    """
    if prod_config is None:
        prod_config = ProductionConfig()
    
    if train_config is None:
        train_config = TrainingConfig()
    
    ghost_kv = GhostKVGenerator(prod_config)
    
    train_dataset = KVDistillationDataset(
        teacher_model, tokenizer, train_texts, prod_config
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=collate_kv_samples,
    )
    
    eval_loader = None
    if eval_texts:
        eval_dataset = KVDistillationDataset(
            teacher_model, tokenizer, eval_texts, prod_config
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=train_config.batch_size,
            collate_fn=collate_kv_samples,
        )
    
    trainer = GhostKVTrainer(ghost_kv, train_config, prod_config)
    results = trainer.train(train_loader, eval_loader)
    
    return ghost_kv, results
