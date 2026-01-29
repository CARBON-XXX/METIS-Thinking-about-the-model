"""
SEDAC-O1: 自适应思考时间 (Adaptive Computation Time)

对标: OpenAI o1 / DeepSeek-R1

核心理念:
- V9.0 SEDAC 是为了"省算力"（做减法）
- SEDAC-O1 是为了"增智慧"（做加法）

当检测到极高熵（极度困惑）时：
1. 不仅不跳层，反而动态插入额外的"思考Token"
2. 循环调用计算模块，直到熵降低到可接受水平
3. 实现System 2深度推理

这是通向AGI的核心 —— 让模型在难问题上自动展开思维链
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
from enum import Enum, auto
import logging
import math

logger = logging.getLogger(__name__)


class ThinkingMode(Enum):
    """思考模式"""
    FAST = auto()      # System 1: 快速直觉响应
    SLOW = auto()      # System 2: 深度推理
    ADAPTIVE = auto()  # 自适应切换


@dataclass
class ThinkingState:
    """思考状态"""
    mode: ThinkingMode
    thinking_depth: int           # 当前思考深度
    max_thinking_depth: int       # 最大允许深度
    accumulated_entropy: float    # 累积熵
    entropy_trajectory: List[float] = field(default_factory=list)
    thinking_tokens: List[str] = field(default_factory=list)
    confidence_trajectory: List[float] = field(default_factory=list)
    should_continue_thinking: bool = True
    reasoning_complete: bool = False


@dataclass
class ThinkingConfig:
    """思考配置"""
    # 熵阈值
    high_entropy_threshold: float = 4.5      # 触发深度思考
    low_entropy_threshold: float = 2.0       # 可以停止思考
    
    # 思考深度
    max_thinking_steps: int = 8              # 最大思考步数
    min_thinking_steps: int = 1              # 最小思考步数
    
    # 自适应参数
    entropy_reduction_target: float = 0.3    # 每步熵降低目标
    confidence_threshold: float = 0.8        # 停止思考的置信度
    
    # Token预算
    max_thinking_tokens: int = 512           # 最大思考Token数
    
    # 学习参数
    adaptive_threshold: bool = True          # 是否自适应调整阈值


class ThinkingTokenGenerator:
    """
    思考Token生成器
    
    在高熵时生成"思考提示"引导模型深入推理
    """
    
    # 预定义的思考提示模板
    THINKING_PROMPTS = {
        "decompose": [
            "Let me break this down step by step.",
            "First, I need to identify the key components.",
            "Let's analyze this systematically.",
        ],
        "verify": [
            "Let me verify this reasoning.",
            "I should double-check this conclusion.",
            "Wait, let me reconsider.",
        ],
        "explore": [
            "What if I approach this differently?",
            "Another way to think about this is...",
            "Consider the alternative perspective:",
        ],
        "synthesize": [
            "Putting it all together...",
            "Based on the above analysis...",
            "Therefore, the conclusion is...",
        ],
    }
    
    def __init__(self, config: ThinkingConfig = None):
        self.config = config or ThinkingConfig()
        self.step_count = 0
    
    def generate_prompt(
        self,
        entropy: float,
        thinking_depth: int,
        entropy_trend: str = "stable",  # "decreasing", "increasing", "stable"
    ) -> str:
        """
        根据当前状态生成思考提示
        """
        if thinking_depth == 0:
            # 开始思考
            prompts = self.THINKING_PROMPTS["decompose"]
        elif entropy_trend == "increasing":
            # 熵增加，需要验证
            prompts = self.THINKING_PROMPTS["verify"]
        elif thinking_depth >= self.config.max_thinking_steps - 2:
            # 接近结束，需要综合
            prompts = self.THINKING_PROMPTS["synthesize"]
        else:
            # 继续探索
            prompts = self.THINKING_PROMPTS["explore"]
        
        # 循环选择
        prompt = prompts[self.step_count % len(prompts)]
        self.step_count += 1
        
        return prompt
    
    def reset(self):
        """重置状态"""
        self.step_count = 0


class EntropyMonitor:
    """
    熵监控器
    
    跟踪熵的变化趋势，决定何时启动/停止深度思考
    """
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history: List[float] = []
        
        # 在线统计量
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # 用于计算方差
    
    def update(self, entropy: float):
        """更新熵观测"""
        self.history.append(entropy)
        if len(self.history) > self.window_size * 2:
            self.history.pop(0)
        
        # Welford在线更新
        self.n += 1
        delta = entropy - self.mean
        self.mean += delta / self.n
        delta2 = entropy - self.mean
        self.M2 += delta * delta2
    
    @property
    def std(self) -> float:
        """标准差"""
        if self.n < 2:
            return 1.0
        return math.sqrt(self.M2 / (self.n - 1))
    
    def get_trend(self) -> str:
        """获取熵趋势"""
        if len(self.history) < 3:
            return "stable"
        
        recent = self.history[-3:]
        if recent[-1] < recent[0] * 0.9:
            return "decreasing"
        elif recent[-1] > recent[0] * 1.1:
            return "increasing"
        else:
            return "stable"
    
    def get_percentile(self, entropy: float) -> float:
        """获取熵的百分位数"""
        if self.n < 10:
            return 0.5
        
        z_score = (entropy - self.mean) / (self.std + 1e-6)
        # 近似CDF
        percentile = 0.5 * (1 + math.erf(z_score / math.sqrt(2)))
        return percentile
    
    def should_trigger_thinking(self, entropy: float, threshold_percentile: float = 0.8) -> bool:
        """是否应该触发深度思考"""
        return self.get_percentile(entropy) > threshold_percentile
    
    def reset(self):
        """重置"""
        self.history.clear()


class AdaptiveComputationController:
    """
    自适应计算控制器
    
    核心组件，决定何时以及如何进行深度思考
    """
    
    def __init__(self, config: ThinkingConfig = None):
        self.config = config or ThinkingConfig()
        self.entropy_monitor = EntropyMonitor()
        self.token_generator = ThinkingTokenGenerator(config)
        
        # 当前思考状态
        self.current_state: Optional[ThinkingState] = None
        
        # 统计
        self.total_tokens = 0
        self.thinking_tokens = 0
        self.thinking_sessions = 0
    
    def should_start_thinking(self, entropy: float, confidence: float) -> bool:
        """
        决定是否启动深度思考
        """
        # 更新监控器
        self.entropy_monitor.update(entropy)
        
        # 条件1: 熵超过高阈值
        if entropy > self.config.high_entropy_threshold:
            return True
        
        # 条件2: 熵处于历史高位（自适应）
        if self.config.adaptive_threshold:
            if self.entropy_monitor.should_trigger_thinking(entropy, 0.85):
                return True
        
        # 条件3: 置信度极低
        if confidence < 0.2:
            return True
        
        return False
    
    def should_continue_thinking(self, state: ThinkingState) -> bool:
        """
        决定是否继续思考
        """
        # 已达到最大深度
        if state.thinking_depth >= state.max_thinking_depth:
            return False
        
        # 熵已经降低到可接受水平
        if state.entropy_trajectory and state.entropy_trajectory[-1] < self.config.low_entropy_threshold:
            return False
        
        # 置信度已经足够高
        if state.confidence_trajectory and state.confidence_trajectory[-1] > self.config.confidence_threshold:
            return False
        
        # 熵在持续增加（思考无效）
        if len(state.entropy_trajectory) >= 3:
            if all(state.entropy_trajectory[i] < state.entropy_trajectory[i+1] 
                   for i in range(-3, -1)):
                return False
        
        return True
    
    def start_thinking(self, entropy: float, confidence: float) -> ThinkingState:
        """
        启动深度思考
        """
        self.thinking_sessions += 1
        
        # 根据初始熵估计需要的思考深度
        entropy_excess = entropy - self.config.low_entropy_threshold
        estimated_steps = min(
            self.config.max_thinking_steps,
            max(self.config.min_thinking_steps, int(entropy_excess / self.config.entropy_reduction_target))
        )
        
        state = ThinkingState(
            mode=ThinkingMode.SLOW,
            thinking_depth=0,
            max_thinking_depth=estimated_steps,
            accumulated_entropy=entropy,
            entropy_trajectory=[entropy],
            confidence_trajectory=[confidence],
            should_continue_thinking=True,
            reasoning_complete=False,
        )
        
        self.current_state = state
        self.token_generator.reset()
        
        logger.debug(f"Started thinking session: entropy={entropy:.2f}, estimated_steps={estimated_steps}")
        
        return state
    
    def step(
        self,
        entropy: float,
        confidence: float,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[ThinkingState, Optional[str]]:
        """
        执行一步思考
        
        Returns:
            (updated_state, thinking_prompt or None)
        """
        if self.current_state is None:
            # 检查是否需要启动思考
            if self.should_start_thinking(entropy, confidence):
                state = self.start_thinking(entropy, confidence)
                prompt = self.token_generator.generate_prompt(
                    entropy, 
                    state.thinking_depth,
                    self.entropy_monitor.get_trend()
                )
                return state, prompt
            else:
                # 快速模式，不需要思考
                return ThinkingState(
                    mode=ThinkingMode.FAST,
                    thinking_depth=0,
                    max_thinking_depth=0,
                    accumulated_entropy=entropy,
                    should_continue_thinking=False,
                    reasoning_complete=True,
                ), None
        
        # 已在思考中
        state = self.current_state
        state.thinking_depth += 1
        state.entropy_trajectory.append(entropy)
        state.confidence_trajectory.append(confidence)
        state.accumulated_entropy += entropy
        
        self.thinking_tokens += 1
        
        # 检查是否继续
        if not self.should_continue_thinking(state):
            state.should_continue_thinking = False
            state.reasoning_complete = True
            self.current_state = None
            return state, None
        
        # 生成下一个思考提示
        prompt = self.token_generator.generate_prompt(
            entropy,
            state.thinking_depth,
            self.entropy_monitor.get_trend()
        )
        state.thinking_tokens.append(prompt)
        
        return state, prompt
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_tokens": self.total_tokens,
            "thinking_tokens": self.thinking_tokens,
            "thinking_sessions": self.thinking_sessions,
            "thinking_ratio": self.thinking_tokens / max(self.total_tokens, 1),
            "avg_thinking_depth": self.thinking_tokens / max(self.thinking_sessions, 1),
        }
    
    def reset(self):
        """重置状态"""
        self.current_state = None
        self.entropy_monitor.reset()
        self.token_generator.reset()


class SEDACO1Engine:
    """
    SEDAC-O1 引擎
    
    结合V9.0的早退能力和O1的深度思考能力
    """
    
    def __init__(
        self,
        config: ThinkingConfig = None,
        device: torch.device = None,
    ):
        self.config = config or ThinkingConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.controller = AdaptiveComputationController(config)
        
        # 模式统计
        self.fast_count = 0
        self.slow_count = 0
    
    def process(
        self,
        entropy: float,
        confidence: float,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[ThinkingMode, Optional[str], Dict[str, Any]]:
        """
        处理单个Token
        
        Returns:
            (mode, thinking_prompt, metadata)
        """
        self.controller.total_tokens += 1
        
        state, prompt = self.controller.step(entropy, confidence, hidden_states)
        
        if state.mode == ThinkingMode.FAST:
            self.fast_count += 1
        else:
            self.slow_count += 1
        
        metadata = {
            "thinking_depth": state.thinking_depth,
            "accumulated_entropy": state.accumulated_entropy,
            "should_continue": state.should_continue_thinking,
            "reasoning_complete": state.reasoning_complete,
        }
        
        return state.mode, prompt, metadata
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计"""
        total = self.fast_count + self.slow_count
        return {
            "fast_mode_ratio": self.fast_count / max(total, 1),
            "slow_mode_ratio": self.slow_count / max(total, 1),
            **self.controller.get_statistics(),
        }
    
    def reset(self):
        """重置"""
        self.controller.reset()
        self.fast_count = 0
        self.slow_count = 0


class ThinkingTokenEmbedder(nn.Module):
    """
    思考Token嵌入器
    
    将思考提示转换为可以注入模型的嵌入
    """
    
    def __init__(
        self,
        hidden_size: int = 4096,
        num_thinking_types: int = 4,  # decompose, verify, explore, synthesize
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 思考类型嵌入
        self.type_embeddings = nn.Embedding(num_thinking_types, hidden_size)
        
        # 深度位置嵌入
        self.depth_projection = nn.Linear(1, hidden_size)
        
        # 融合层
        self.fusion = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(
        self,
        thinking_type: int,
        thinking_depth: int,
        max_depth: int = 8,
    ) -> torch.Tensor:
        """
        生成思考Token嵌入
        
        Returns:
            embedding: [1, hidden_size]
        """
        # 类型嵌入
        type_idx = torch.tensor([thinking_type], device=self.type_embeddings.weight.device)
        type_emb = self.type_embeddings(type_idx)
        
        # 深度嵌入
        depth_normalized = torch.tensor([[thinking_depth / max_depth]], 
                                        device=self.type_embeddings.weight.device,
                                        dtype=torch.float32)
        depth_emb = self.depth_projection(depth_normalized)
        
        # 融合
        combined = torch.cat([type_emb, depth_emb], dim=-1)
        output = self.fusion(combined)
        
        return output


def create_sedac_o1_engine(config: ThinkingConfig = None) -> SEDACO1Engine:
    """创建SEDAC-O1引擎"""
    return SEDACO1Engine(config)


def demo_sedac_o1():
    """演示SEDAC-O1"""
    import random
    
    print("=" * 60)
    print("SEDAC-O1 Demo: Adaptive Computation Time")
    print("=" * 60)
    
    engine = create_sedac_o1_engine()
    
    # 模拟不同难度的问题
    scenarios = [
        ("简单事实问答", [(1.5, 0.9), (1.2, 0.95)]),  # 低熵，高置信
        ("中等推理", [(3.0, 0.6), (2.5, 0.7), (2.0, 0.8)]),  # 中熵
        ("复杂数学证明", [(5.0, 0.2), (4.5, 0.3), (4.0, 0.4), (3.5, 0.5), (3.0, 0.7), (2.5, 0.85)]),  # 高熵，需要深度思考
    ]
    
    for scenario_name, entropy_confidence_pairs in scenarios:
        print(f"\n{'='*40}")
        print(f"场景: {scenario_name}")
        print(f"{'='*40}")
        
        engine.reset()
        
        for i, (entropy, confidence) in enumerate(entropy_confidence_pairs):
            mode, prompt, metadata = engine.process(entropy, confidence)
            
            mode_str = "🏃 FAST" if mode == ThinkingMode.FAST else "🤔 SLOW"
            print(f"  Step {i+1}: entropy={entropy:.2f}, conf={confidence:.2f} → {mode_str}")
            
            if prompt:
                print(f"    💭 {prompt}")
            
            if metadata["reasoning_complete"]:
                print(f"    ✅ 推理完成 (depth={metadata['thinking_depth']})")
        
        stats = engine.get_statistics()
        print(f"\n  统计: Fast={stats['fast_mode_ratio']*100:.1f}%, Slow={stats['slow_mode_ratio']*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("SEDAC-O1: 简单问题快速回答，复杂问题深度思考")
    print("=" * 60)


if __name__ == "__main__":
    demo_sedac_o1()
