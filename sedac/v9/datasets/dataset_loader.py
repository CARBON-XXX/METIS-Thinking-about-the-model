"""
SEDAC V9.0 - HuggingFace数据集加载器

支持的数据集:
- 数学: NuminaMath-CoT, Orca-Math
- 代码: CodeFeedback-Filtered-Instruction
- 指令: Alpaca, UltraChat
"""

from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Iterator
import logging
import json
import os

logger = logging.getLogger(__name__)

# HuggingFace datasets
try:
    from datasets import load_dataset, Dataset as HFDataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("datasets library not available. Install with: pip install datasets")


@dataclass
class DatasetConfig:
    """数据集配置"""
    name: str
    hf_path: str
    subset: Optional[str] = None
    split: str = "train"
    text_field: str = "text"
    max_samples: int = 10000
    cache_dir: str = "./cache/datasets"


# 预定义数据集配置
DATASET_CONFIGS = {
    # 数学
    "numina_math": DatasetConfig(
        name="NuminaMath-CoT",
        hf_path="AI-MO/NuminaMath-CoT",
        text_field="solution",
        max_samples=50000,
    ),
    "orca_math": DatasetConfig(
        name="Orca-Math",
        hf_path="microsoft/orca-math-word-problems-200k",
        text_field="question",
        max_samples=50000,
    ),
    # 代码
    "code_feedback": DatasetConfig(
        name="CodeFeedback",
        hf_path="m-a-p/CodeFeedback-Filtered-Instruction",
        text_field="query",
        max_samples=50000,
    ),
    # 指令
    "alpaca": DatasetConfig(
        name="Alpaca",
        hf_path="tatsu-lab/alpaca",
        text_field="text",
        max_samples=50000,
    ),
    "ultrachat": DatasetConfig(
        name="UltraChat",
        hf_path="stingning/ultrachat",
        text_field="data",
        max_samples=50000,
    ),
}


class SEDACDataset(Dataset):
    """
    SEDAC训练数据集
    
    自动估算难度，支持多种数据源
    """
    
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: Any = None,
        max_length: int = 2048,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        if self.tokenizer is not None:
            # Tokenize
            encoded = self.tokenizer(
                sample["text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "difficulty": torch.tensor(sample.get("difficulty", 0.5)),
                "category": sample.get("category", "unknown"),
            }
        
        return sample


class DatasetLoader:
    """
    数据集加载器
    
    支持从HuggingFace加载多种数据集
    """
    
    def __init__(self, cache_dir: str = "./cache/datasets"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        if not HF_AVAILABLE:
            raise RuntimeError("datasets library required. Install with: pip install datasets")
    
    def load_dataset(
        self,
        dataset_name: str,
        max_samples: Optional[int] = None,
        streaming: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        加载单个数据集
        """
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
        
        config = DATASET_CONFIGS[dataset_name]
        max_samples = max_samples or config.max_samples
        
        logger.info(f"Loading {config.name} from {config.hf_path}...")
        
        try:
            # 加载数据集
            if config.subset:
                ds = load_dataset(
                    config.hf_path,
                    config.subset,
                    split=config.split,
                    cache_dir=self.cache_dir,
                    streaming=streaming,
                )
            else:
                ds = load_dataset(
                    config.hf_path,
                    split=config.split,
                    cache_dir=self.cache_dir,
                    streaming=streaming,
                )
            
            # 转换为样本列表
            samples = []
            count = 0
            
            for item in ds:
                if count >= max_samples:
                    break
                
                # 提取文本
                text = self._extract_text(item, config.text_field)
                if text:
                    # 估算难度
                    difficulty = self._estimate_difficulty(text, dataset_name)
                    
                    samples.append({
                        "text": text,
                        "difficulty": difficulty,
                        "category": dataset_name,
                        "source": config.name,
                    })
                    count += 1
            
            logger.info(f"Loaded {len(samples)} samples from {config.name}")
            return samples
            
        except Exception as e:
            logger.error(f"Failed to load {config.name}: {e}")
            return []
    
    def _extract_text(self, item: Dict, field: str) -> Optional[str]:
        """提取文本字段"""
        if field in item:
            value = item[field]
            if isinstance(value, str):
                return value
            elif isinstance(value, list):
                # UltraChat格式
                return " ".join(str(v) for v in value if v)
            elif isinstance(value, dict):
                return json.dumps(value)
        
        # 尝试常见字段
        for f in ["text", "content", "instruction", "input", "output", "question", "answer"]:
            if f in item and item[f]:
                return str(item[f])
        
        return None
    
    def _estimate_difficulty(self, text: str, category: str) -> float:
        """
        估算文本难度
        
        基于多种启发式特征
        """
        difficulty = 0.5
        
        # 长度因子
        length = len(text)
        if length > 2000:
            difficulty += 0.1
        elif length > 1000:
            difficulty += 0.05
        elif length < 100:
            difficulty -= 0.1
        
        # 数学特征
        if category in ["numina_math", "orca_math"]:
            # 数学符号密度
            math_symbols = sum(1 for c in text if c in "∫∑∏√∞≠≤≥±×÷")
            if math_symbols > 5:
                difficulty += 0.15
            
            # LaTeX公式
            if "\\frac" in text or "\\int" in text or "\\sum" in text:
                difficulty += 0.1
        
        # 代码特征
        if category == "code_feedback":
            # 代码块
            if "```" in text or "def " in text or "class " in text:
                difficulty += 0.1
            
            # 复杂度指示词
            if any(w in text.lower() for w in ["algorithm", "optimize", "complexity", "recursive"]):
                difficulty += 0.1
        
        # 推理特征
        if any(w in text.lower() for w in ["therefore", "because", "since", "thus", "hence"]):
            difficulty += 0.05
        
        # 多步骤
        steps = text.count("Step ") + text.count("step ")
        if steps > 3:
            difficulty += 0.1
        
        return min(max(difficulty, 0.0), 1.0)
    
    def load_all_datasets(
        self,
        max_samples_per_dataset: int = 10000,
    ) -> List[Dict[str, Any]]:
        """
        加载所有数据集
        """
        all_samples = []
        
        for dataset_name in DATASET_CONFIGS:
            samples = self.load_dataset(dataset_name, max_samples_per_dataset)
            all_samples.extend(samples)
        
        logger.info(f"Total samples loaded: {len(all_samples)}")
        return all_samples
    
    def create_dataloader(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: Any = None,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 4,
    ) -> DataLoader:
        """
        创建DataLoader
        """
        dataset = SEDACDataset(samples, tokenizer)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )


def download_all_datasets(cache_dir: str = "./cache/datasets"):
    """
    下载所有数据集（预热缓存）
    """
    if not HF_AVAILABLE:
        print("请先安装datasets: pip install datasets")
        return
    
    loader = DatasetLoader(cache_dir)
    
    print("=" * 60)
    print("SEDAC V9.0 - 下载训练数据集")
    print("=" * 60)
    
    for name, config in DATASET_CONFIGS.items():
        print(f"\n下载: {config.name}")
        print(f"  路径: {config.hf_path}")
        
        try:
            samples = loader.load_dataset(name, max_samples=1000)
            print(f"  ✅ 成功加载 {len(samples)} 样本")
            
            # 显示示例
            if samples:
                sample = samples[0]
                text_preview = sample["text"][:100] + "..." if len(sample["text"]) > 100 else sample["text"]
                print(f"  示例: {text_preview}")
                print(f"  难度: {sample['difficulty']:.2f}")
        except Exception as e:
            print(f"  ❌ 失败: {e}")
    
    print("\n" + "=" * 60)
    print("数据集下载完成")
    print("=" * 60)


if __name__ == "__main__":
    download_all_datasets()
