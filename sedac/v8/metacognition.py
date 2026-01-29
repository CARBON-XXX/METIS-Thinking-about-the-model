"""
SEDAC V8.0 - Metacognition Module (元认知模块)

实现多路决策树:
- CONFIDENT → Early Exit
- UNCERTAIN → Full Inference  
- HALLUCINATION_RISK → Intervention

干预手段:
- SPECULATIVE_DECODE: 多路验证
- CHAIN_OF_THOUGHT: 注入 "let me think step by step"
- RAG_RETRIEVAL: 调用外部知识库
- CODE_INTERPRETER: 调用代码执行器
"""

import torch
from dataclasses import dataclass
from typing import Optional, List, Callable, Dict, Any
from enum import Enum, auto


class Decision(Enum):
    """元认知决策类型"""
    CONTINUE = auto()           # 继续计算下一层
    EARLY_EXIT = auto()         # 提前退出，输出当前结果
    FULL_INFERENCE = auto()     # 强制跑完所有层
    INTERVENE = auto()          # 触发干预机制


class InterventionType(Enum):
    """干预类型"""
    NONE = auto()
    SPECULATIVE_DECODE = auto()   # 多路验证
    CHAIN_OF_THOUGHT = auto()     # 思维链提示
    RAG_RETRIEVAL = auto()        # 检索增强
    CODE_INTERPRETER = auto()     # 代码执行
    HUMAN_IN_LOOP = auto()        # 人工介入


@dataclass
class MetacognitiveState:
    """元认知状态"""
    decision: Decision
    intervention: InterventionType = InterventionType.NONE
    confidence: float = 0.0
    hallucination_risk: float = 0.0
    ood_risk: float = 0.0
    layer_idx: int = 0
    reasoning: str = ""


@dataclass
class InterventionResult:
    """干预结果"""
    success: bool
    new_output: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class InterventionHandler:
    """
    干预处理器基类
    
    子类需要实现 handle() 方法
    """
    
    def __init__(self, name: str):
        self.name = name
    
    def handle(
        self,
        hidden_state: torch.Tensor,
        context: Dict[str, Any]
    ) -> InterventionResult:
        raise NotImplementedError


class SpeculativeDecodeHandler(InterventionHandler):
    """
    推测解码干预
    
    生成多个候选，然后验证
    """
    
    def __init__(self, num_candidates: int = 3):
        super().__init__("speculative_decode")
        self.num_candidates = num_candidates
    
    def handle(
        self,
        hidden_state: torch.Tensor,
        context: Dict[str, Any]
    ) -> InterventionResult:
        # 占位实现 - 实际需要与模型集成
        return InterventionResult(
            success=True,
            metadata={
                "type": "speculative_decode",
                "num_candidates": self.num_candidates,
                "action": "VERIFY_BEFORE_OUTPUT",
            }
        )


class ChainOfThoughtHandler(InterventionHandler):
    """
    思维链干预
    
    注入 "Wait, let me think step by step" 提示
    """
    
    def __init__(self, prompt: str = "Wait, let me think step by step."):
        super().__init__("chain_of_thought")
        self.prompt = prompt
    
    def handle(
        self,
        hidden_state: torch.Tensor,
        context: Dict[str, Any]
    ) -> InterventionResult:
        return InterventionResult(
            success=True,
            metadata={
                "type": "chain_of_thought",
                "inject_prompt": self.prompt,
                "action": "PREPEND_TO_OUTPUT",
            }
        )


class RAGRetrievalHandler(InterventionHandler):
    """
    RAG 检索干预
    
    调用外部知识库获取相关信息
    """
    
    def __init__(self, retriever: Optional[Callable] = None):
        super().__init__("rag_retrieval")
        self.retriever = retriever
    
    def handle(
        self,
        hidden_state: torch.Tensor,
        context: Dict[str, Any]
    ) -> InterventionResult:
        # 如果有 retriever，调用它
        if self.retriever is not None:
            query = context.get("query", "")
            retrieved = self.retriever(query)
            return InterventionResult(
                success=True,
                metadata={
                    "type": "rag_retrieval",
                    "retrieved_docs": retrieved,
                    "action": "AUGMENT_CONTEXT",
                }
            )
        
        return InterventionResult(
            success=False,
            metadata={"error": "No retriever configured"}
        )


class MetacognitionModule:
    """
    元认知模块 (The Prefrontal Cortex)
    
    SEDAC V8.0 的决策中枢
    
    职责:
    1. 整合直觉信号
    2. 做出元认知决策
    3. 触发并管理干预
    
    工作流:
    ```
    直觉信号 → 元认知判断 → 决策
                              ├→ EARLY_EXIT (快速输出)
                              ├→ CONTINUE (继续计算)
                              ├→ FULL_INFERENCE (跑完全程)
                              └→ INTERVENE (触发干预)
                                    ├→ Speculative Decode
                                    ├→ Chain of Thought
                                    ├→ RAG Retrieval
                                    └→ Code Interpreter
    ```
    """
    
    def __init__(
        self,
        confident_threshold: float = 0.7,
        hallucination_threshold: float = 0.5,
        ood_threshold: float = 0.6,
        min_layer_ratio: float = 0.2,
    ):
        self.confident_threshold = confident_threshold
        self.hallucination_threshold = hallucination_threshold
        self.ood_threshold = ood_threshold
        self.min_layer_ratio = min_layer_ratio
        
        # 干预处理器
        self.handlers: Dict[InterventionType, InterventionHandler] = {
            InterventionType.SPECULATIVE_DECODE: SpeculativeDecodeHandler(),
            InterventionType.CHAIN_OF_THOUGHT: ChainOfThoughtHandler(),
            InterventionType.RAG_RETRIEVAL: RAGRetrievalHandler(),
        }
        
        # 统计
        self.stats = {
            "total_decisions": 0,
            "early_exits": 0,
            "full_inferences": 0,
            "interventions": 0,
            "intervention_types": {},
        }
    
    def register_handler(
        self, 
        intervention_type: InterventionType,
        handler: InterventionHandler
    ):
        """注册干预处理器"""
        self.handlers[intervention_type] = handler
    
    def decide(
        self,
        p_confident: float,
        p_hallucination: float,
        p_ood: float,
        layer_idx: int,
        total_layers: int,
    ) -> MetacognitiveState:
        """
        做出元认知决策
        
        决策逻辑 (优先级从高到低):
        1. 幻觉风险高 → INTERVENE (speculative decode)
        2. 超纲风险高 → INTERVENE (RAG retrieval)
        3. 层数太少 → CONTINUE
        4. 置信度高 → EARLY_EXIT
        5. 置信度低 → CONTINUE (可能需要 FULL_INFERENCE)
        """
        self.stats["total_decisions"] += 1
        
        min_layer = int(total_layers * self.min_layer_ratio)
        reasoning_parts = []
        
        # 1. 检查幻觉风险
        if p_hallucination > self.hallucination_threshold:
            self.stats["interventions"] += 1
            self.stats["intervention_types"]["hallucination"] = \
                self.stats["intervention_types"].get("hallucination", 0) + 1
            
            reasoning_parts.append(
                f"High hallucination risk ({p_hallucination:.2f} > {self.hallucination_threshold})"
            )
            
            return MetacognitiveState(
                decision=Decision.INTERVENE,
                intervention=InterventionType.SPECULATIVE_DECODE,
                confidence=p_confident,
                hallucination_risk=p_hallucination,
                ood_risk=p_ood,
                layer_idx=layer_idx,
                reasoning=" | ".join(reasoning_parts),
            )
        
        # 2. 检查 OOD 风险
        if p_ood > self.ood_threshold:
            self.stats["interventions"] += 1
            self.stats["intervention_types"]["ood"] = \
                self.stats["intervention_types"].get("ood", 0) + 1
            
            reasoning_parts.append(
                f"High OOD risk ({p_ood:.2f} > {self.ood_threshold})"
            )
            
            return MetacognitiveState(
                decision=Decision.INTERVENE,
                intervention=InterventionType.RAG_RETRIEVAL,
                confidence=p_confident,
                hallucination_risk=p_hallucination,
                ood_risk=p_ood,
                layer_idx=layer_idx,
                reasoning=" | ".join(reasoning_parts),
            )
        
        # 3. 检查最小层数
        if layer_idx < min_layer:
            reasoning_parts.append(
                f"Below min layer ({layer_idx} < {min_layer})"
            )
            
            return MetacognitiveState(
                decision=Decision.CONTINUE,
                intervention=InterventionType.NONE,
                confidence=p_confident,
                hallucination_risk=p_hallucination,
                ood_risk=p_ood,
                layer_idx=layer_idx,
                reasoning=" | ".join(reasoning_parts),
            )
        
        # 4. 检查置信度
        if p_confident > self.confident_threshold:
            self.stats["early_exits"] += 1
            reasoning_parts.append(
                f"High confidence ({p_confident:.2f} > {self.confident_threshold})"
            )
            
            return MetacognitiveState(
                decision=Decision.EARLY_EXIT,
                intervention=InterventionType.NONE,
                confidence=p_confident,
                hallucination_risk=p_hallucination,
                ood_risk=p_ood,
                layer_idx=layer_idx,
                reasoning=" | ".join(reasoning_parts),
            )
        
        # 5. 默认继续
        reasoning_parts.append(
            f"Low confidence ({p_confident:.2f} <= {self.confident_threshold}), continue"
        )
        
        return MetacognitiveState(
            decision=Decision.CONTINUE,
            intervention=InterventionType.NONE,
            confidence=p_confident,
            hallucination_risk=p_hallucination,
            ood_risk=p_ood,
            layer_idx=layer_idx,
            reasoning=" | ".join(reasoning_parts),
        )
    
    def execute_intervention(
        self,
        intervention_type: InterventionType,
        hidden_state: torch.Tensor,
        context: Dict[str, Any] = None
    ) -> InterventionResult:
        """执行干预"""
        if context is None:
            context = {}
        
        handler = self.handlers.get(intervention_type)
        if handler is None:
            return InterventionResult(
                success=False,
                metadata={"error": f"No handler for {intervention_type}"}
            )
        
        return handler.handle(hidden_state, context)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = self.stats["total_decisions"]
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            "early_exit_rate": self.stats["early_exits"] / total,
            "intervention_rate": self.stats["interventions"] / total,
        }
    
    def reset_stats(self):
        """重置统计"""
        self.stats = {
            "total_decisions": 0,
            "early_exits": 0,
            "full_inferences": 0,
            "interventions": 0,
            "intervention_types": {},
        }


class SEDACv8:
    """
    SEDAC V8.0 - The Intuition Layer
    
    完整的元认知推理系统
    
    使用示例:
    ```python
    sedac = SEDACv8()
    
    for layer_idx, layer in enumerate(model.layers):
        hidden = layer(hidden)
        entropy = compute_entropy(hidden)
        
        state = sedac.step(hidden, entropy, layer_idx, len(model.layers))
        
        if state.decision == Decision.EARLY_EXIT:
            break
        elif state.decision == Decision.INTERVENE:
            result = sedac.intervene(state.intervention, hidden, context)
            # 处理干预结果
    ```
    """
    
    def __init__(
        self,
        intuition_network = None,
        metacognition: MetacognitionModule = None,
        device: torch.device = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 延迟导入避免循环依赖
        from sedac.v8.intuition_network import (
            IntuitionNetwork, IntuitionConfig, FeatureExtractor
        )
        
        self.intuition = intuition_network or IntuitionNetwork(IntuitionConfig())
        self.intuition = self.intuition.to(self.device)
        
        self.metacognition = metacognition or MetacognitionModule()
        self.feature_extractor = FeatureExtractor(device=self.device)
    
    def reset(self):
        """重置状态"""
        self.feature_extractor.reset()
        self.metacognition.reset_stats()
    
    def step(
        self,
        hidden: torch.Tensor,
        entropy: Optional[torch.Tensor] = None,
        layer_idx: int = 0,
        total_layers: int = 32,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> MetacognitiveState:
        """
        单步推理
        
        Returns:
            MetacognitiveState 包含决策和干预类型
        """
        # 提取特征
        features = self.feature_extractor.extract(
            hidden, entropy, attention_weights
        )
        
        # 获取直觉信号
        with torch.no_grad():
            signal = self.intuition(features, layer_idx)
        
        # 元认知决策
        state = self.metacognition.decide(
            p_confident=signal.p_confident.mean().item(),
            p_hallucination=signal.p_hallucination.mean().item(),
            p_ood=signal.p_ood.mean().item(),
            layer_idx=layer_idx,
            total_layers=total_layers,
        )
        
        return state
    
    def intervene(
        self,
        intervention_type: InterventionType,
        hidden: torch.Tensor,
        context: Dict[str, Any] = None
    ) -> InterventionResult:
        """执行干预"""
        return self.metacognition.execute_intervention(
            intervention_type, hidden, context
        )
    
    def get_stats(self) -> Dict[str, Any]:
        return self.metacognition.get_stats()
