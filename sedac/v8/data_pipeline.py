"""
SEDAC V8.0 - Training Data Pipeline

生成训练数据的管道:
1. 从真实推理中收集 hidden states + entropies
2. 标注 ground truth (is_correct, is_ood, optimal_exit_layer)
3. 生成训练样本

训练样本格式:
{
    "features": [8-dim tensor per layer],
    "targets": {
        "is_correct": bool,
        "is_ood": bool, 
        "optimal_exit_layer": int,
    }
}
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm


@dataclass
class TrainingSample:
    """单个训练样本"""
    token_idx: int
    features_per_layer: List[torch.Tensor]  # [num_layers, 8]
    is_correct: bool
    is_ood: bool
    optimal_exit_layer: int
    final_entropy: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingDataset:
    """训练数据集"""
    samples: List[TrainingSample]
    num_layers: int
    num_features: int = 8
    
    def __len__(self):
        return len(self.samples)
    
    def get_batch(
        self, 
        indices: List[int],
        layer_idx: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """获取一个 batch"""
        batch_features = []
        batch_is_correct = []
        batch_is_ood = []
        batch_optimal_layer = []
        
        for idx in indices:
            sample = self.samples[idx]
            batch_features.append(sample.features_per_layer[layer_idx])
            batch_is_correct.append(float(sample.is_correct))
            batch_is_ood.append(float(sample.is_ood))
            batch_optimal_layer.append(sample.optimal_exit_layer)
        
        features = torch.stack(batch_features).to(device)
        targets = {
            "is_correct": torch.tensor(batch_is_correct, device=device),
            "is_ood": torch.tensor(batch_is_ood, device=device),
            "optimal_exit_layer": torch.tensor(batch_optimal_layer, device=device),
        }
        
        return features, targets
    
    def save(self, path: str):
        """保存数据集"""
        data = {
            "num_layers": self.num_layers,
            "num_features": self.num_features,
            "samples": [
                {
                    "token_idx": s.token_idx,
                    "features_per_layer": [f.tolist() for f in s.features_per_layer],
                    "is_correct": s.is_correct,
                    "is_ood": s.is_ood,
                    "optimal_exit_layer": s.optimal_exit_layer,
                    "final_entropy": s.final_entropy,
                    "metadata": s.metadata,
                }
                for s in self.samples
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path: str, device: torch.device = None) -> "TrainingDataset":
        """加载数据集"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        device = device or torch.device("cpu")
        
        samples = []
        for s in data["samples"]:
            samples.append(TrainingSample(
                token_idx=s["token_idx"],
                features_per_layer=[
                    torch.tensor(f, device=device) for f in s["features_per_layer"]
                ],
                is_correct=s["is_correct"],
                is_ood=s["is_ood"],
                optimal_exit_layer=s["optimal_exit_layer"],
                final_entropy=s["final_entropy"],
                metadata=s.get("metadata", {}),
            ))
        
        return cls(
            samples=samples,
            num_layers=data["num_layers"],
            num_features=data.get("num_features", 8),
        )


class DataPipeline:
    """
    训练数据生成管道
    
    从 hidden states 和 entropies 生成训练样本
    """
    
    def __init__(
        self,
        device: torch.device = None,
        high_risk_percentile: float = 75,
        ood_percentile: float = 95,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.high_risk_percentile = high_risk_percentile
        self.ood_percentile = ood_percentile
    
    def generate_from_files(
        self,
        data_dir: str,
        output_path: str = None
    ) -> TrainingDataset:
        """
        从文件生成训练数据
        
        Args:
            data_dir: 包含 hidden_states_layer*.pt 和 entropies_layer*.pt 的目录
            output_path: 保存路径 (可选)
        
        Returns:
            TrainingDataset
        """
        data_dir = Path(data_dir)
        
        # 加载数据
        layer_files = sorted(
            data_dir.glob("hidden_states_layer*.pt"),
            key=lambda x: int(x.stem.split("layer")[-1])
        )
        num_layers = len(layer_files)
        
        print(f"Loading data from {data_dir}...")
        print(f"  Found {num_layers} layers")
        
        hidden_states = []
        for f in layer_files:
            hs = torch.load(f, map_location=self.device)
            hidden_states.append(hs)
        
        entropies = []
        for layer_idx in range(num_layers):
            entropy_file = data_dir / f"entropies_layer{layer_idx}.pt"
            if entropy_file.exists():
                ent = torch.load(entropy_file, map_location=self.device)
                entropies.append(ent)
            else:
                entropies.append(None)
        
        num_tokens = hidden_states[0].shape[0]
        print(f"  Tokens: {num_tokens}")
        
        # 计算阈值
        final_entropies = entropies[-1].cpu().numpy() if entropies[-1] is not None else np.zeros(num_tokens)
        high_risk_threshold = float(np.percentile(final_entropies, self.high_risk_percentile))
        ood_threshold = float(np.percentile(final_entropies, self.ood_percentile))
        
        print(f"  High-risk threshold (P{self.high_risk_percentile}): {high_risk_threshold:.4f}")
        print(f"  OOD threshold (P{self.ood_percentile}): {ood_threshold:.4f}")
        
        # 生成样本
        dataset = self._generate_samples(
            hidden_states,
            entropies,
            high_risk_threshold,
            ood_threshold,
        )
        
        # 保存
        if output_path:
            dataset.save(output_path)
            print(f"  Saved to {output_path}")
        
        return dataset
    
    def _generate_samples(
        self,
        hidden_states: List[torch.Tensor],
        entropies: List[Optional[torch.Tensor]],
        high_risk_threshold: float,
        ood_threshold: float,
    ) -> TrainingDataset:
        """生成训练样本"""
        num_layers = len(hidden_states)
        num_tokens = hidden_states[0].shape[0]
        
        samples = []
        
        print(f"Generating samples...")
        
        for token_idx in tqdm(range(num_tokens), desc="Tokens"):
            # 提取每层特征
            features_per_layer = []
            prev_hidden = None
            prev_entropy = None
            history_norms = []
            
            for layer_idx in range(num_layers):
                features = torch.zeros(8, device=self.device)
                
                hidden = hidden_states[layer_idx][token_idx]
                entropy = entropies[layer_idx][token_idx] if entropies[layer_idx] is not None else None
                
                # 0. Entropy
                if entropy is not None:
                    features[0] = entropy
                
                # 1. Entropy delta
                if entropy is not None and prev_entropy is not None:
                    features[1] = entropy - prev_entropy
                
                # 2. Stability
                if prev_hidden is not None:
                    cos_sim = F.cosine_similarity(
                        prev_hidden.float().unsqueeze(0),
                        hidden.float().unsqueeze(0),
                        dim=1
                    )
                    features[2] = (cos_sim.item() + 1.0) / 2.0
                else:
                    features[2] = 1.0
                
                # 3. Hidden norm
                hidden_norm = torch.norm(hidden.float()).item()
                features[3] = hidden_norm / 1000.0
                
                # 4. Norm delta
                if len(history_norms) > 0:
                    features[4] = (hidden_norm - history_norms[-1]) / 100.0
                
                # 5. Norm acceleration
                if len(history_norms) >= 2:
                    prev_delta = history_norms[-1] - history_norms[-2]
                    curr_delta = hidden_norm - history_norms[-1]
                    features[5] = (curr_delta - prev_delta) / 100.0
                
                # 6, 7. Reserved
                features[6] = 0.0
                features[7] = layer_idx / num_layers  # 层进度
                
                features_per_layer.append(features.clone())
                
                # 更新历史
                prev_hidden = hidden
                if entropy is not None:
                    prev_entropy = entropy
                history_norms.append(hidden_norm)
            
            # 计算 labels
            final_entropy = entropies[-1][token_idx].item() if entropies[-1] is not None else 0
            is_correct = final_entropy < high_risk_threshold
            is_ood = final_entropy > ood_threshold
            
            # 计算最优退出层 (第一个稳定且正确的层)
            optimal_exit_layer = num_layers
            if is_correct:
                for layer_idx in range(num_layers // 3, num_layers):
                    if features_per_layer[layer_idx][2] > 0.95:  # stability > 0.95
                        optimal_exit_layer = layer_idx
                        break
            
            samples.append(TrainingSample(
                token_idx=token_idx,
                features_per_layer=features_per_layer,
                is_correct=is_correct,
                is_ood=is_ood,
                optimal_exit_layer=optimal_exit_layer,
                final_entropy=final_entropy,
            ))
        
        # 统计
        correct_count = sum(1 for s in samples if s.is_correct)
        ood_count = sum(1 for s in samples if s.is_ood)
        avg_optimal_layer = sum(s.optimal_exit_layer for s in samples) / len(samples)
        
        print(f"\nDataset statistics:")
        print(f"  Total samples: {len(samples)}")
        print(f"  Correct: {correct_count} ({correct_count/len(samples)*100:.1f}%)")
        print(f"  OOD: {ood_count} ({ood_count/len(samples)*100:.1f}%)")
        print(f"  Avg optimal exit layer: {avg_optimal_layer:.1f}")
        
        return TrainingDataset(
            samples=samples,
            num_layers=num_layers,
        )


class Trainer:
    """
    Intuition Network 训练器
    """
    
    def __init__(
        self,
        network,
        loss_fn,
        optimizer,
        device: torch.device = None,
    ):
        self.network = network
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.network = self.network.to(self.device)
    
    def train_epoch(
        self,
        dataset: TrainingDataset,
        batch_size: int = 64,
        num_epochs: int = 1,
    ) -> Dict[str, float]:
        """训练一个 epoch"""
        self.network.train()
        
        num_samples = len(dataset)
        indices = list(range(num_samples))
        
        total_losses = {"total": 0, "speed": 0, "accuracy": 0, "calibration": 0}
        num_batches = 0
        
        for epoch in range(num_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, num_samples, batch_size):
                batch_indices = indices[start:start + batch_size]
                
                # 随机选择一个层
                layer_idx = np.random.randint(dataset.num_layers // 3, dataset.num_layers)
                
                features, targets = dataset.get_batch(batch_indices, layer_idx, self.device)
                
                # 前向传播
                signal = self.network(features, layer_idx)
                
                # 计算损失
                losses = self.loss_fn(signal, targets, layer_idx, dataset.num_layers)
                
                # 反向传播
                self.optimizer.zero_grad()
                losses["total"].backward()
                self.optimizer.step()
                
                # 累计损失
                for k, v in losses.items():
                    total_losses[k] += v.item()
                num_batches += 1
        
        # 平均损失
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses
    
    def evaluate(
        self,
        dataset: TrainingDataset,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """评估"""
        self.network.eval()
        
        num_samples = len(dataset)
        indices = list(range(num_samples))
        
        correct_exits = 0
        wrong_exits = 0
        total_exit_layer = 0
        exit_count = 0
        
        with torch.no_grad():
            for start in range(0, num_samples, batch_size):
                batch_indices = indices[start:start + batch_size]
                
                for idx in batch_indices:
                    sample = dataset.samples[idx]
                    
                    # 逐层评估
                    for layer_idx in range(dataset.num_layers):
                        features = sample.features_per_layer[layer_idx].unsqueeze(0).to(self.device)
                        signal = self.network(features, layer_idx)
                        
                        # 检查是否退出
                        if signal.p_confident.item() > self.network.config.confident_threshold:
                            exit_count += 1
                            total_exit_layer += layer_idx
                            
                            if sample.is_correct:
                                correct_exits += 1
                            else:
                                wrong_exits += 1
                            break
        
        results = {
            "exit_rate": exit_count / num_samples if num_samples > 0 else 0,
            "avg_exit_layer": total_exit_layer / exit_count if exit_count > 0 else dataset.num_layers,
            "exit_precision": correct_exits / exit_count if exit_count > 0 else 0,
            "correct_exits": correct_exits,
            "wrong_exits": wrong_exits,
        }
        
        return results


def generate_training_data():
    """生成训练数据的便捷函数"""
    pipeline = DataPipeline()
    dataset = pipeline.generate_from_files(
        data_dir="sedac_data_v7_full",
        output_path="sedac_v8_training_data.json"
    )
    return dataset


if __name__ == "__main__":
    generate_training_data()
