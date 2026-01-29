"""
SEDAC V9.0 - Cognitive Attention Engine

核心引擎：实现时间维度的稀疏性计算

与DeepSeek-VL2的对偶关系:
    DeepSeek-VL2: 空间稀疏 → "看哪里" (Spatial Attention)
    SEDAC V9.0:   时间稀疏 → "想多深" (Computational Attention)

统一理念: 在信息的荒原中，只开采高密度的矿脉
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum, auto
from pathlib import Path
import logging

from sedac.v8.intuition_network import (
    IntuitionNetwork,
    IntuitionConfig,
    IntuitionSignal,
    FeatureExtractor,
)
from sedac.v8.metacognition import (
    MetacognitionModule,
    Decision,
    InterventionType,
    MetacognitiveState,
    InterventionResult,
)

logger = logging.getLogger(__name__)


class AttentionMode(Enum):
    """认知注意力模式"""
    REFLEX = auto()      # 反射模式：极低熵，几乎不需要思考
    INTUITION = auto()   # 直觉模式：低熵，快速判断
    DELIBERATE = auto()  # 审慎模式：中熵，需要思考
    ANALYSIS = auto()    # 分析模式：高熵，深度推理
    UNCERTAINTY = auto() # 不确定模式：极高熵，可能需要外部帮助


@dataclass
class AttentionState:
    """认知注意力状态"""
    mode: AttentionMode
    entropy_level: float          # 当前熵水平 [0, 1]
    confidence: float             # 置信度 [0, 1]
    recommended_depth: float      # 推荐计算深度 [0, 1]
    should_exit: bool             # 是否应该退出
    should_intervene: bool        # 是否需要干预
    intervention_type: Optional[InterventionType] = None
    reasoning: str = ""


@dataclass 
class EngineConfig:
    """引擎配置"""
    # 模式阈值
    reflex_threshold: float = 0.95      # 极高置信 → 反射
    intuition_threshold: float = 0.80   # 高置信 → 直觉
    deliberate_threshold: float = 0.60  # 中置信 → 审慎
    analysis_threshold: float = 0.40    # 低置信 → 分析
    # 低于analysis_threshold → 不确定
    
    # 退出控制
    min_layer_ratio: float = 0.2        # 最小层比例
    max_speedup: float = 4.0            # 最大加速比
    
    # 干预阈值
    hallucination_threshold: float = 0.5
    ood_threshold: float = 0.6
    
    # 模型路径
    checkpoint_path: Optional[str] = None


class CognitiveAttentionEngine:
    """
    认知注意力引擎 (The Entropy Engine)
    
    SEDAC V9.0 的核心组件
    
    职责:
    1. 实时评估token的"认知复杂度"（熵）
    2. 动态分配计算深度（层数）
    3. 触发必要的干预机制
    
    工作原理:
    ```
    输入(hidden_state) 
        → 特征提取(entropy, stability, ...)
        → 直觉网络(IntuitionNetwork)
        → 认知模式判定(AttentionMode)
        → 输出决策(退出/继续/干预)
    ```
    
    使用示例:
    ```python
    engine = CognitiveAttentionEngine()
    
    for layer_idx, layer in enumerate(model.layers):
        hidden = layer(hidden)
        
        state = engine.step(hidden, layer_idx, total_layers)
        
        if state.should_exit:
            break
        elif state.should_intervene:
            # 处理干预...
    ```
    """
    
    def __init__(
        self,
        config: EngineConfig = None,
        intuition_config: IntuitionConfig = None,
        device: torch.device = None,
    ):
        self.config = config or EngineConfig()
        self.intuition_config = intuition_config or IntuitionConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 核心组件
        self.intuition = IntuitionNetwork(self.intuition_config).to(self.device)
        self.metacognition = MetacognitionModule(
            confident_threshold=self.config.intuition_threshold,
            hallucination_threshold=self.config.hallucination_threshold,
            ood_threshold=self.config.ood_threshold,
            min_layer_ratio=self.config.min_layer_ratio,
        )
        self.feature_extractor = FeatureExtractor(device=self.device)
        
        # 加载训练好的模型（如果有）
        if self.config.checkpoint_path:
            self.load_checkpoint(self.config.checkpoint_path)
        
        # 状态追踪
        self.history: List[AttentionState] = []
        self.stats = {
            "total_tokens": 0,
            "exits_by_mode": {mode.name: 0 for mode in AttentionMode},
            "interventions": 0,
            "avg_exit_layer": 0.0,
        }
    
    def reset(self):
        """重置状态"""
        self.feature_extractor.reset()
        self.history.clear()
        self.metacognition.reset_stats()
    
    def load_checkpoint(self, path: str):
        """加载训练好的直觉网络"""
        # PyTorch 2.6+ 需要 weights_only=False 或添加安全全局类
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.intuition.load_state_dict(checkpoint["model_state_dict"])
        self.intuition.eval()
        logger.info(f"Loaded intuition network from {path}")
    
    def _determine_mode(self, confidence: float) -> AttentionMode:
        """根据置信度确定认知模式"""
        if confidence >= self.config.reflex_threshold:
            return AttentionMode.REFLEX
        elif confidence >= self.config.intuition_threshold:
            return AttentionMode.INTUITION
        elif confidence >= self.config.deliberate_threshold:
            return AttentionMode.DELIBERATE
        elif confidence >= self.config.analysis_threshold:
            return AttentionMode.ANALYSIS
        else:
            return AttentionMode.UNCERTAINTY
    
    def _calculate_recommended_depth(
        self, 
        mode: AttentionMode, 
        layer_idx: int, 
        total_layers: int
    ) -> float:
        """计算推荐的计算深度"""
        current_progress = layer_idx / total_layers
        
        depth_map = {
            AttentionMode.REFLEX: 0.2,       # 只需要20%的层
            AttentionMode.INTUITION: 0.4,    # 40%
            AttentionMode.DELIBERATE: 0.6,   # 60%
            AttentionMode.ANALYSIS: 0.8,     # 80%
            AttentionMode.UNCERTAINTY: 1.0,  # 完整推理
        }
        
        target_depth = depth_map[mode]
        
        # 如果已经超过推荐深度，返回当前进度
        return max(current_progress, target_depth)
    
    def step(
        self,
        hidden: torch.Tensor,
        layer_idx: int,
        total_layers: int,
        entropy: Optional[torch.Tensor] = None,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> AttentionState:
        """
        单步推理 - 核心决策函数
        
        Args:
            hidden: 当前层的hidden state [batch_size, hidden_dim]
            layer_idx: 当前层索引
            total_layers: 总层数
            entropy: 可选的预计算熵
            attention_weights: 可选的注意力权重
        
        Returns:
            AttentionState: 包含决策和推理信息
        """
        # 1. 特征提取
        features = self.feature_extractor.extract(
            hidden, entropy, attention_weights
        )
        
        # 2. 直觉信号
        with torch.no_grad():
            signal = self.intuition(features, layer_idx)
        
        # 3. 获取概率值
        p_confident = signal.p_confident.mean().item()
        p_hallucination = signal.p_hallucination.mean().item()
        p_ood = signal.p_ood.mean().item()
        
        # 4. 确定认知模式
        mode = self._determine_mode(p_confident)
        
        # 5. 计算推荐深度
        recommended_depth = self._calculate_recommended_depth(
            mode, layer_idx, total_layers
        )
        
        # 6. 当前进度
        current_progress = (layer_idx + 1) / total_layers
        min_progress = self.config.min_layer_ratio
        
        # 7. 决策逻辑
        should_exit = False
        should_intervene = False
        intervention_type = None
        reasoning_parts = []
        
        # 检查是否达到最小层数
        if current_progress < min_progress:
            reasoning_parts.append(f"Below min layer ({current_progress:.1%} < {min_progress:.1%})")
        else:
            # 检查干预需求
            if p_hallucination > self.config.hallucination_threshold:
                should_intervene = True
                intervention_type = InterventionType.SPECULATIVE_DECODE
                reasoning_parts.append(f"High hallucination risk ({p_hallucination:.2f})")
                
            elif p_ood > self.config.ood_threshold:
                should_intervene = True
                intervention_type = InterventionType.RAG_RETRIEVAL
                reasoning_parts.append(f"OOD detected ({p_ood:.2f})")
                
            # 检查是否可以退出
            elif current_progress >= recommended_depth:
                if mode in [AttentionMode.REFLEX, AttentionMode.INTUITION]:
                    should_exit = True
                    reasoning_parts.append(f"Mode={mode.name}, confidence={p_confident:.2f}")
        
        # 8. 构建状态
        state = AttentionState(
            mode=mode,
            entropy_level=1.0 - p_confident,  # 熵 ≈ 1 - 置信度
            confidence=p_confident,
            recommended_depth=recommended_depth,
            should_exit=should_exit,
            should_intervene=should_intervene,
            intervention_type=intervention_type,
            reasoning=" | ".join(reasoning_parts) if reasoning_parts else f"Mode={mode.name}",
        )
        
        # 9. 记录历史
        self.history.append(state)
        
        # 10. 更新统计
        if should_exit:
            self.stats["exits_by_mode"][mode.name] += 1
            self.stats["total_tokens"] += 1
            # 更新平均退出层
            n = self.stats["total_tokens"]
            old_avg = self.stats["avg_exit_layer"]
            self.stats["avg_exit_layer"] = old_avg + (layer_idx - old_avg) / n
        
        if should_intervene:
            self.stats["interventions"] += 1
        
        return state
    
    def get_speedup(self, total_layers: int) -> float:
        """计算当前加速比"""
        if self.stats["total_tokens"] == 0:
            return 1.0
        avg_exit = self.stats["avg_exit_layer"]
        if avg_exit == 0:
            return 1.0
        return total_layers / avg_exit
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "mode_distribution": {
                k: v / max(self.stats["total_tokens"], 1)
                for k, v in self.stats["exits_by_mode"].items()
            },
        }
    
    def intervene(
        self,
        intervention_type: InterventionType,
        hidden: torch.Tensor,
        context: Dict[str, Any] = None,
    ) -> InterventionResult:
        """执行干预"""
        return self.metacognition.execute_intervention(
            intervention_type, hidden, context
        )


def create_engine(
    checkpoint_path: Optional[str] = None,
    device: str = "auto",
    **kwargs
) -> CognitiveAttentionEngine:
    """
    工厂函数：创建认知注意力引擎
    
    Args:
        checkpoint_path: 训练好的模型路径
        device: 设备 ("auto", "cuda", "cpu")
        **kwargs: 其他EngineConfig参数
    
    Returns:
        CognitiveAttentionEngine实例
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = EngineConfig(
        checkpoint_path=checkpoint_path,
        **kwargs
    )
    
    return CognitiveAttentionEngine(
        config=config,
        device=torch.device(device),
    )
