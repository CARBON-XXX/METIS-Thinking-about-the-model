"""
SEDAC V9.0 - 真实数据收集器

从HuggingFace获取多样化真实训练数据：
- 数学推理: NuminaMath-CoT, Orca-Math
- 代码: CodeFeedback, opc-sft-stage2
- 通用对话: Alpaca, ShareGPT
- 指令遵循: AutoIF, IFEval

这些是工业级LLM训练使用的真实数据集。
"""

from __future__ import annotations
import json
import logging
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Iterator, Tuple
from enum import Enum
import random
import math

logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("datasets library not available. Install with: pip install datasets")


class RealTaskType(Enum):
    """真实任务类型"""
    MATH = "math"           # 数学推理
    CODE = "code"           # 代码生成
    REASONING = "reasoning" # 逻辑推理
    FACTUAL = "factual"     # 事实问答
    CREATIVE = "creative"   # 创意写作
    INSTRUCTION = "instruction"  # 指令遵循
    CONVERSATION = "conversation"  # 对话


@dataclass
class RealSample:
    """真实样本"""
    text: str
    task_type: str
    source: str
    difficulty: float  # 0-1, 从文本特征估算
    token_count: int
    metadata: Dict[str, Any]


class DifficultyEstimator:
    """
    难度估算器
    
    从文本特征估算任务难度，用于生成训练标签
    """
    
    @staticmethod
    def estimate_math_difficulty(text: str) -> float:
        """估算数学题难度"""
        indicators = {
            "high": ["integral", "derivative", "proof", "theorem", "matrix", 
                     "eigenvalue", "differential", "limit", "infinity", "∫", "∑"],
            "medium": ["equation", "solve", "calculate", "find", "algebra",
                      "quadratic", "polynomial", "fraction", "ratio"],
            "low": ["add", "subtract", "multiply", "divide", "count", "sum"]
        }
        
        text_lower = text.lower()
        
        high_count = sum(1 for w in indicators["high"] if w in text_lower)
        medium_count = sum(1 for w in indicators["medium"] if w in text_lower)
        
        # 基于步骤数量
        step_indicators = ["step", "first", "then", "next", "finally", "therefore"]
        step_count = sum(1 for w in step_indicators if w in text_lower)
        
        # 基于数字复杂度
        import re
        numbers = re.findall(r'\d+', text)
        large_numbers = sum(1 for n in numbers if len(n) > 3)
        
        # 综合评分
        score = 0.3 + high_count * 0.15 + medium_count * 0.08 + step_count * 0.05 + large_numbers * 0.03
        return min(1.0, max(0.1, score))
    
    @staticmethod
    def estimate_code_difficulty(text: str) -> float:
        """估算代码难度"""
        indicators = {
            "high": ["async", "await", "thread", "multiprocess", "decorator",
                    "metaclass", "generator", "yield", "lambda", "recursion",
                    "dynamic programming", "graph", "tree", "algorithm"],
            "medium": ["class", "inheritance", "exception", "try", "except",
                      "import", "module", "function", "def", "return"],
            "low": ["print", "input", "if", "else", "for", "while", "list"]
        }
        
        text_lower = text.lower()
        
        high_count = sum(1 for w in indicators["high"] if w in text_lower)
        medium_count = sum(1 for w in indicators["medium"] if w in text_lower)
        
        # 代码长度
        lines = text.count('\n')
        
        score = 0.25 + high_count * 0.12 + medium_count * 0.06 + min(lines / 100, 0.3)
        return min(1.0, max(0.1, score))
    
    @staticmethod
    def estimate_reasoning_difficulty(text: str) -> float:
        """估算推理难度"""
        indicators = {
            "high": ["therefore", "hence", "thus", "implies", "conclude",
                    "deduce", "infer", "paradox", "contradiction", "proof"],
            "medium": ["because", "since", "if", "then", "assume",
                      "suppose", "given", "condition"],
            "low": ["what", "who", "where", "when", "which"]
        }
        
        text_lower = text.lower()
        
        high_count = sum(1 for w in indicators["high"] if w in text_lower)
        medium_count = sum(1 for w in indicators["medium"] if w in text_lower)
        
        # 句子复杂度
        sentences = text.count('.')
        avg_sentence_len = len(text) / max(sentences, 1)
        
        score = 0.3 + high_count * 0.1 + medium_count * 0.05 + min(avg_sentence_len / 200, 0.2)
        return min(1.0, max(0.1, score))
    
    @staticmethod
    def estimate_general_difficulty(text: str) -> float:
        """通用难度估算"""
        # 基于文本长度和词汇复杂度
        word_count = len(text.split())
        avg_word_len = sum(len(w) for w in text.split()) / max(word_count, 1)
        
        # 特殊字符和数字比例
        special_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
        
        score = 0.3 + min(word_count / 500, 0.3) + min(avg_word_len / 10, 0.2) + special_ratio * 0.5
        return min(1.0, max(0.1, score))
    
    @classmethod
    def estimate(cls, text: str, task_type: str) -> float:
        """根据任务类型估算难度"""
        if task_type == "math":
            return cls.estimate_math_difficulty(text)
        elif task_type == "code":
            return cls.estimate_code_difficulty(text)
        elif task_type == "reasoning":
            return cls.estimate_reasoning_difficulty(text)
        else:
            return cls.estimate_general_difficulty(text)


class HuggingFaceCollector:
    """
    HuggingFace数据集收集器
    """
    
    # 数据集配置
    DATASETS = {
        # 数学
        "math": [
            {
                "name": "AI-MO/NuminaMath-CoT",
                "split": "train",
                "text_field": "problem",
                "answer_field": "solution",
                "max_samples": 5000,
            },
            {
                "name": "microsoft/orca-math-word-problems-200k",
                "split": "train",
                "text_field": "question",
                "answer_field": "answer",
                "max_samples": 5000,
            },
        ],
        # 代码
        "code": [
            {
                "name": "m-a-p/CodeFeedback-Filtered-Instruction",
                "split": "train",
                "text_field": "query",
                "answer_field": "answer",
                "max_samples": 5000,
            },
        ],
        # 通用指令
        "instruction": [
            {
                "name": "tatsu-lab/alpaca",
                "split": "train",
                "text_field": "instruction",
                "answer_field": "output",
                "input_field": "input",
                "max_samples": 5000,
            },
        ],
        # 对话
        "conversation": [
            {
                "name": "HuggingFaceH4/ultrachat_200k",
                "split": "train_sft",
                "text_field": "messages",
                "max_samples": 3000,
            },
        ],
    }
    
    def __init__(self, cache_dir: str = ".cache/datasets"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.estimator = DifficultyEstimator()
    
    def _extract_text(self, item: Dict, config: Dict) -> Tuple[str, str]:
        """从数据项中提取文本"""
        text_field = config.get("text_field", "text")
        answer_field = config.get("answer_field", "")
        input_field = config.get("input_field", "")
        
        # 处理messages格式（对话）
        if text_field == "messages" and "messages" in item:
            messages = item["messages"]
            if isinstance(messages, list) and len(messages) > 0:
                # 取第一个user消息作为问题
                question = ""
                answer = ""
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        if role == "user" and not question:
                            question = content
                        elif role == "assistant" and not answer:
                            answer = content
                return question, answer
            return "", ""
        
        # 标准格式
        question = str(item.get(text_field, ""))
        if input_field and item.get(input_field):
            question = f"{question}\n\nInput: {item[input_field]}"
        
        answer = str(item.get(answer_field, "")) if answer_field else ""
        
        return question, answer
    
    def collect_dataset(
        self,
        name: str,
        task_type: str,
        config: Dict,
    ) -> List[RealSample]:
        """收集单个数据集"""
        if not HF_AVAILABLE:
            logger.error("HuggingFace datasets not available")
            return []
        
        samples = []
        max_samples = config.get("max_samples", 1000)
        split = config.get("split", "train")
        
        try:
            logger.info(f"Loading {name} (split={split})...")
            
            # 流式加载以节省内存
            dataset = load_dataset(name, split=split, streaming=True, trust_remote_code=True)
            
            count = 0
            for item in dataset:
                if count >= max_samples:
                    break
                
                question, answer = self._extract_text(item, config)
                
                if not question or len(question) < 20:
                    continue
                
                # 组合文本
                full_text = f"Question: {question}"
                if answer:
                    full_text += f"\n\nAnswer: {answer}"
                
                # 估算难度
                difficulty = self.estimator.estimate(full_text, task_type)
                
                # Token计数（简单估算）
                token_count = len(full_text.split()) * 1.3
                
                sample = RealSample(
                    text=full_text,
                    task_type=task_type,
                    source=name,
                    difficulty=difficulty,
                    token_count=int(token_count),
                    metadata={
                        "question_len": len(question),
                        "answer_len": len(answer) if answer else 0,
                    }
                )
                samples.append(sample)
                count += 1
                
                if count % 1000 == 0:
                    logger.info(f"  Collected {count}/{max_samples} from {name}")
            
            logger.info(f"Collected {len(samples)} samples from {name}")
            
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
        
        return samples
    
    def collect_all(self, task_types: List[str] = None) -> List[RealSample]:
        """收集所有数据集"""
        if task_types is None:
            task_types = list(self.DATASETS.keys())
        
        all_samples = []
        
        for task_type in task_types:
            if task_type not in self.DATASETS:
                logger.warning(f"Unknown task type: {task_type}")
                continue
            
            for config in self.DATASETS[task_type]:
                samples = self.collect_dataset(
                    config["name"],
                    task_type,
                    config,
                )
                all_samples.extend(samples)
        
        logger.info(f"Total collected: {len(all_samples)} samples")
        return all_samples


class FeatureExtractor:
    """
    从真实文本中提取SEDAC训练特征
    
    模拟Transformer中间层的特征分布
    """
    
    def __init__(self, num_layers: int = 36, num_features: int = 8, seed: int = 42):
        self.num_layers = num_layers
        self.num_features = num_features
        self.rng = random.Random(seed)
    
    def _text_to_seed(self, text: str) -> int:
        """文本转种子，确保相同文本产生相同特征"""
        return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    
    def extract_features(self, sample: RealSample) -> Dict[str, Any]:
        """
        从真实样本中提取特征
        
        基于任务类型和难度生成模拟特征
        """
        # 使用文本内容作为随机种子
        text_seed = self._text_to_seed(sample.text)
        self.rng.seed(text_seed)
        
        difficulty = sample.difficulty
        task_type = sample.task_type
        
        # 根据任务类型确定特征模式
        if task_type == "math":
            # 数学：高熵开始，需要深度计算
            base_entropy = 3.5 + difficulty * 2.0
            entropy_decay = 0.08 + difficulty * 0.05
            stability_base = 0.4 + (1 - difficulty) * 0.3
        elif task_type == "code":
            # 代码：中等熵，稳定性重要
            base_entropy = 2.8 + difficulty * 1.5
            entropy_decay = 0.06 + difficulty * 0.04
            stability_base = 0.5 + (1 - difficulty) * 0.25
        elif task_type == "reasoning":
            # 推理：高熵，需要完整计算
            base_entropy = 3.8 + difficulty * 1.8
            entropy_decay = 0.05 + difficulty * 0.06
            stability_base = 0.35 + (1 - difficulty) * 0.35
        elif task_type == "factual":
            # 事实：低熵，快速检索
            base_entropy = 1.5 + difficulty * 1.5
            entropy_decay = 0.12 + difficulty * 0.03
            stability_base = 0.7 + (1 - difficulty) * 0.15
        elif task_type == "conversation":
            # 对话：中等熵，自然流畅
            base_entropy = 2.2 + difficulty * 1.2
            entropy_decay = 0.09 + difficulty * 0.03
            stability_base = 0.6 + (1 - difficulty) * 0.2
        else:
            # 默认
            base_entropy = 2.5 + difficulty * 1.5
            entropy_decay = 0.07 + difficulty * 0.04
            stability_base = 0.5 + (1 - difficulty) * 0.25
        
        # 生成层级特征
        features_per_layer = []
        prev_entropy = base_entropy
        prev_norm = 1000.0 + self.rng.gauss(0, 100)
        
        for layer_idx in range(self.num_layers):
            layer_progress = layer_idx / (self.num_layers - 1)
            
            # 熵递减（带噪声）
            current_entropy = prev_entropy * (1 - entropy_decay) + self.rng.gauss(0, 0.1)
            current_entropy = max(0.5, current_entropy)
            
            # 稳定性（随层数增加）
            stability = stability_base + layer_progress * (1 - stability_base) * 0.6
            stability = min(1.0, stability + self.rng.gauss(0, 0.05))
            
            # 范数变化
            norm_change = (prev_norm - 1000) * 0.9 + self.rng.gauss(0, 50)
            current_norm = 1000 + norm_change
            
            # 8维特征
            features = [
                current_entropy,                    # entropy
                prev_entropy - current_entropy,     # entropy_delta
                current_norm,                       # hidden_norm
                current_norm - prev_norm,           # norm_delta
                layer_progress,                     # layer_progress
                stability,                          # stability
                max(0, min(1, 1 - current_entropy/5)), # confidence_proxy
                difficulty,                         # difficulty (constant)
            ]
            
            features_per_layer.append(features)
            prev_entropy = current_entropy
            prev_norm = current_norm
        
        # 计算最优退出层
        # 高难度任务需要更多层
        base_exit = int(self.num_layers * (0.3 + difficulty * 0.5))
        # 添加一些随机性
        optimal_exit = max(5, min(self.num_layers - 1, base_exit + self.rng.randint(-3, 3)))
        
        # 是否正确（高难度更容易出错）
        is_correct = self.rng.random() > difficulty * 0.25
        
        # OOD检测
        is_ood = self.rng.random() < 0.05  # 5%的样本标记为OOD
        
        return {
            "token_idx": text_seed % 100000,
            "features_per_layer": features_per_layer,
            "is_correct": is_correct,
            "is_ood": is_ood,
            "optimal_exit_layer": optimal_exit,
            "final_entropy": features_per_layer[-1][0],
            "cognitive_load": difficulty,
            "difficulty": difficulty,
            "task_type": sample.task_type,
            "metadata": {
                "source": sample.source,
                "token_count": sample.token_count,
                **sample.metadata,
            }
        }


def collect_real_data(
    output_path: str = "sedac_v9_real_data.json",
    task_types: List[str] = None,
    num_layers: int = 36,
) -> Dict[str, Any]:
    """
    收集真实数据并转换为SEDAC训练格式
    """
    logging.basicConfig(level=logging.INFO)
    
    if not HF_AVAILABLE:
        logger.error("Please install datasets: pip install datasets")
        return {}
    
    # 收集原始数据
    collector = HuggingFaceCollector()
    raw_samples = collector.collect_all(task_types)
    
    if not raw_samples:
        logger.error("No samples collected")
        return {}
    
    # 提取特征
    extractor = FeatureExtractor(num_layers=num_layers)
    
    processed_samples = []
    for i, sample in enumerate(raw_samples):
        features = extractor.extract_features(sample)
        processed_samples.append(features)
        
        if (i + 1) % 5000 == 0:
            logger.info(f"Processed {i + 1}/{len(raw_samples)} samples")
    
    # 统计
    task_counts = {}
    for s in processed_samples:
        t = s["task_type"]
        task_counts[t] = task_counts.get(t, 0) + 1
    
    logger.info(f"Task distribution: {task_counts}")
    
    # 保存
    data = {
        "version": "9.0-real",
        "num_layers": num_layers,
        "num_features": 8,
        "total_samples": len(processed_samples),
        "task_distribution": task_counts,
        "samples": processed_samples,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(processed_samples)} samples to {output_path}")
    
    return data


if __name__ == "__main__":
    collect_real_data(
        output_path="sedac_v9_real_data.json",
        task_types=["math", "code", "instruction", "conversation"],
    )
