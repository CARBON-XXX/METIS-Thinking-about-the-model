"""
SEDAC V9.0 Demo - The Entropy Engine

演示认知注意力引擎的工作流程：
- 时间稀疏性计算（想多深）
- 认知模式切换（反射/直觉/审慎/分析/不确定）
- 干预机制触发
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from sedac.v9 import (
    CognitiveAttentionEngine,
    AttentionMode,
    EngineConfig,
    create_engine,
)


def demo_attention_modes():
    """演示认知模式切换"""
    print("=" * 70)
    print("Demo 1: Cognitive Attention Modes (认知模式)")
    print("=" * 70)
    print("""
    与DeepSeek-VL2的对偶关系:
    - DeepSeek: 空间稀疏 → "看哪里" (Spatial Attention)
    - SEDAC:    时间稀疏 → "想多深" (Computational Attention)
    
    认知模式层级:
    - REFLEX (反射):     极高置信 → 几乎不需要思考
    - INTUITION (直觉):  高置信 → 快速判断
    - DELIBERATE (审慎): 中置信 → 需要思考  
    - ANALYSIS (分析):   低置信 → 深度推理
    - UNCERTAINTY (不确定): 极低置信 → 可能需要外部帮助
    """)
    
    config = EngineConfig(
        reflex_threshold=0.95,
        intuition_threshold=0.80,
        deliberate_threshold=0.60,
        analysis_threshold=0.40,
        min_layer_ratio=0.2,
    )
    
    engine = CognitiveAttentionEngine(config=config)
    
    # 模拟不同置信度的场景
    scenarios = [
        ("极高置信 (1+1=?)", 0.98, AttentionMode.REFLEX),
        ("高置信 (常识问答)", 0.85, AttentionMode.INTUITION),
        ("中等置信 (复杂推理)", 0.65, AttentionMode.DELIBERATE),
        ("低置信 (专业问题)", 0.45, AttentionMode.ANALYSIS),
        ("极低置信 (超纲问题)", 0.25, AttentionMode.UNCERTAINTY),
    ]
    
    print(f"{'场景':<25} | {'置信度':>8} | {'期望模式':<15} | {'实际模式':<15}")
    print("-" * 80)
    
    for name, confidence, expected_mode in scenarios:
        actual_mode = engine._determine_mode(confidence)
        match = "✓" if actual_mode == expected_mode else "✗"
        print(f"{name:<25} | {confidence:>8.2f} | {expected_mode.name:<15} | {actual_mode.name:<15} {match}")


def demo_engine_simulation():
    """演示引擎模拟推理"""
    print("\n" + "=" * 70)
    print("Demo 2: Engine Simulation (引擎模拟)")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 尝试加载训练好的模型
    checkpoint_path = "checkpoints/intuition_network_best.pt"
    if Path(checkpoint_path).exists():
        print(f"加载训练好的模型: {checkpoint_path}")
        engine = create_engine(checkpoint_path=checkpoint_path, device=str(device))
    else:
        print("未找到训练模型，使用随机初始化")
        engine = create_engine(device=str(device))
    engine.reset()
    
    num_layers = 36
    batch_size = 1
    hidden_dim = 2048
    
    print(f"\n模拟 {num_layers} 层推理...")
    print()
    
    # 场景1: Easy Token (高置信，应该早退)
    print("场景1: Easy Token (高置信)")
    print("-" * 50)
    
    engine.reset()
    for layer_idx in range(num_layers):
        # 模拟高置信场景：稳定的hidden state
        hidden = torch.randn(batch_size, hidden_dim, device=device) * 0.1
        if layer_idx > 0:
            hidden = prev_hidden * 0.95 + hidden * 0.05  # 高度稳定
        prev_hidden = hidden.clone()
        
        # 模拟低熵
        entropy = torch.tensor([0.5], device=device)  # 低熵
        
        state = engine.step(hidden, layer_idx, num_layers, entropy)
        
        if state.should_exit:
            print(f"  Layer {layer_idx:2d}: EXIT | Mode={state.mode.name} | Conf={state.confidence:.2f}")
            break
        elif layer_idx % 5 == 0:
            print(f"  Layer {layer_idx:2d}: CONTINUE | Mode={state.mode.name} | Conf={state.confidence:.2f}")
    
    # 场景2: Hard Token (低置信，应该跑完)
    print("\n场景2: Hard Token (低置信)")
    print("-" * 50)
    
    engine.reset()
    exit_layer = num_layers
    for layer_idx in range(num_layers):
        # 模拟低置信场景：变化剧烈的hidden state
        hidden = torch.randn(batch_size, hidden_dim, device=device)
        
        # 模拟高熵
        entropy = torch.tensor([3.5], device=device)  # 高熵
        
        state = engine.step(hidden, layer_idx, num_layers, entropy)
        
        if state.should_exit:
            exit_layer = layer_idx
            print(f"  Layer {layer_idx:2d}: EXIT | Mode={state.mode.name}")
            break
        elif state.should_intervene:
            print(f"  Layer {layer_idx:2d}: INTERVENE → {state.intervention_type.name}")
        elif layer_idx % 10 == 0:
            print(f"  Layer {layer_idx:2d}: CONTINUE | Mode={state.mode.name} | Conf={state.confidence:.2f}")
    
    if exit_layer == num_layers:
        print(f"  完整推理完成 ({num_layers} layers)")


def demo_with_real_data():
    """使用真实数据演示"""
    print("\n" + "=" * 70)
    print("Demo 3: Real Data Test (真实数据测试)")
    print("=" * 70)
    
    data_dir = Path("sedac_data_v7_full")
    if not data_dir.exists():
        print(f"数据目录不存在: {data_dir}")
        print("跳过真实数据测试...")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    layer_files = sorted(data_dir.glob("hidden_states_layer*.pt"),
                        key=lambda x: int(x.stem.split("layer")[-1]))
    num_layers = len(layer_files)
    
    if num_layers == 0:
        print("未找到hidden states文件")
        return
    
    hidden_states = []
    for f in layer_files:
        hs = torch.load(f, map_location=device)
        hidden_states.append(hs)
    
    entropies = []
    for layer_idx in range(num_layers):
        entropy_file = data_dir / f"entropies_layer{layer_idx}.pt"
        if entropy_file.exists():
            ent = torch.load(entropy_file, map_location=device)
            entropies.append(ent)
    
    num_tokens = hidden_states[0].shape[0]
    print(f"加载 {num_tokens} tokens, {num_layers} layers")
    
    # 高风险标记
    if entropies:
        final_entropies = entropies[-1].cpu().numpy()
        high_risk_threshold = float(np.percentile(final_entropies, 75))
        high_risk_mask = entropies[-1] > high_risk_threshold
        print(f"高风险tokens: {high_risk_mask.sum().item()} / {num_tokens}")
    else:
        high_risk_mask = torch.zeros(num_tokens, dtype=torch.bool)
    
    # 测试
    test_tokens = min(100, num_tokens)
    engine = create_engine(device=str(device))
    
    exit_layers = []
    mode_counts = {mode.name: 0 for mode in AttentionMode}
    correct_exits = 0
    wrong_exits = 0
    
    for token_idx in range(test_tokens):
        engine.reset()
        
        exit_layer = num_layers
        exit_mode = None
        
        for layer_idx in range(num_layers):
            hidden = hidden_states[layer_idx][token_idx:token_idx+1]
            entropy = entropies[layer_idx][token_idx:token_idx+1] if layer_idx < len(entropies) else None
            
            state = engine.step(hidden, layer_idx, num_layers, entropy)
            
            if state.should_exit:
                exit_layer = layer_idx
                exit_mode = state.mode
                
                if high_risk_mask[token_idx]:
                    wrong_exits += 1
                else:
                    correct_exits += 1
                break
        
        exit_layers.append(exit_layer)
        if exit_mode:
            mode_counts[exit_mode.name] += 1
    
    # 统计
    avg_exit = sum(exit_layers) / len(exit_layers)
    speedup = num_layers / avg_exit if avg_exit > 0 else 1.0
    exited_count = sum(1 for l in exit_layers if l < num_layers)
    
    print(f"\n结果 (前 {test_tokens} 个 token):")
    print(f"  平均退出层: {avg_exit:.1f} / {num_layers}")
    print(f"  加速比: {speedup:.2f}x")
    print(f"  退出率: {exited_count}/{test_tokens} ({exited_count/test_tokens*100:.1f}%)")
    print(f"  正确退出: {correct_exits}")
    print(f"  错误退出 (高风险): {wrong_exits}")
    
    if exited_count > 0:
        precision = correct_exits / exited_count
        print(f"  退出精度: {precision*100:.1f}%")
    
    print(f"\n模式分布:")
    for mode, count in mode_counts.items():
        if count > 0:
            print(f"  {mode}: {count} ({count/test_tokens*100:.1f}%)")


def demo_comparison():
    """V8 vs V9 对比"""
    print("\n" + "=" * 70)
    print("Demo 4: V8 vs V9 Comparison (版本对比)")
    print("=" * 70)
    print("""
    V8.0 (Intuition Network):
    - 三元输出: p_confident, p_hallucination, p_ood
    - 二元决策: EXIT / CONTINUE / INTERVENE
    
    V9.0 (Cognitive Attention Engine):
    - 五级认知模式: REFLEX → INTUITION → DELIBERATE → ANALYSIS → UNCERTAINTY
    - 推荐计算深度: 动态建议需要多少层
    - 更精细的控制粒度
    
    核心区别:
    - V8: "这个token我确定吗？" (二分法)
    - V9: "这个token需要多少思考？" (连续谱)
    """)


def main():
    print("=" * 70)
    print("SEDAC V9.0 - The Entropy Engine")
    print("Cognitive Attention Engine: 认知注意力引擎")
    print("=" * 70)
    print("""
    核心理念: 在信息的荒原中，只开采高密度的矿脉
    
    - 不是"算得更快"，而是"算得更少"
    - 不是"加速器"，而是"认知协处理器"
    """)
    
    demo_attention_modes()
    demo_engine_simulation()
    demo_with_real_data()
    demo_comparison()
    
    print("\n" + "=" * 70)
    print("V9.0 演示完成!")
    print("=" * 70)
    print("""
下一步:
1. 训练直觉网络 (python -m sedac.v9.trainer)
2. 收集更多训练数据
3. 集成到推理框架
""")


if __name__ == "__main__":
    main()
