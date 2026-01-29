"""
SEDAC Quick Start - 零配置使用示例

用户拿到手不用做任何设置，直接用。
"""

import torch
from pathlib import Path

# 一行导入
from sedac import auto


def demo_basic():
    """最简单的使用方式"""
    print("=" * 60)
    print("Demo 1: 最简单的使用方式")
    print("=" * 60)
    
    # 一行创建
    monitor = auto()
    
    # 模拟数据
    batch_size = 8
    hidden_dim = 2048
    num_layers = 36
    
    print(f"Batch size: {batch_size}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Num layers: {num_layers}")
    print()
    
    # 模拟前向传播
    monitor.reset(batch_size)
    
    for layer_idx in range(num_layers):
        # 模拟隐藏状态 (实际使用时这是 layer(hidden) 的输出)
        hidden = torch.randn(batch_size, hidden_dim, device=monitor.device)
        
        # 核心: 一行判断是否退出
        exit_mask = monitor.step(hidden, layer_idx, num_layers)
        
        if exit_mask.any():
            print(f"Layer {layer_idx}: {exit_mask.sum().item()}/{batch_size} tokens want to exit")
        
        if exit_mask.all():
            print(f"All tokens exited at layer {layer_idx}")
            break
    
    # 查看统计
    stats = monitor.get_stats()
    print(f"\nStats: {stats}")


def demo_modes():
    """不同模式对比"""
    print("\n" + "=" * 60)
    print("Demo 2: 不同模式对比")
    print("=" * 60)
    
    modes = ["conservative", "balanced", "aggressive", "maximum_speed"]
    
    for mode in modes:
        monitor = auto(mode)
        config = monitor._config  # 需要先 step 一次才有 config
        
        # 触发自动初始化
        hidden = torch.randn(1, 2048, device=monitor.device)
        monitor.step(hidden, 0, 36)
        
        config = monitor.get_config()
        print(f"\n{mode.upper()}:")
        print(f"  tau = {config.tau:.3f}")
        print(f"  k = {config.k}")
        print(f"  min_layer_ratio = {config.min_layer_ratio:.2f}")
        print(f"  min_layers_before_exit = {config.min_layers_before_exit}")


def demo_with_real_data():
    """使用真实数据测试"""
    print("\n" + "=" * 60)
    print("Demo 3: 使用真实数据")
    print("=" * 60)
    
    data_dir = Path("sedac_data_v7_full")
    if not data_dir.exists():
        print(f"Data dir not found: {data_dir}")
        return
    
    # 加载数据
    layer_files = sorted(data_dir.glob("hidden_states_layer*.pt"))
    num_layers = len(layer_files)
    
    hidden_states = []
    for f in layer_files:
        hs = torch.load(f, map_location="cuda" if torch.cuda.is_available() else "cpu")
        hidden_states.append(hs)
    
    num_tokens = hidden_states[0].shape[0]
    print(f"Loaded {num_tokens} tokens, {num_layers} layers")
    
    # 测试不同模式
    for mode in ["balanced", "aggressive"]:
        monitor = auto(mode)
        
        exit_layers = []
        
        for token_idx in range(min(100, num_tokens)):  # 测试前 100 个 token
            monitor.reset(1)
            
            for layer_idx in range(num_layers):
                hidden = hidden_states[layer_idx][token_idx:token_idx+1]
                exit_mask = monitor.step(hidden, layer_idx, num_layers)
                
                if exit_mask.item():
                    exit_layers.append(layer_idx)
                    break
            else:
                exit_layers.append(num_layers)
        
        avg_exit = sum(exit_layers) / len(exit_layers)
        speedup = num_layers / avg_exit
        exit_rate = sum(1 for l in exit_layers if l < num_layers) / len(exit_layers)
        
        print(f"\n{mode.upper()}:")
        print(f"  Avg exit layer: {avg_exit:.1f} / {num_layers}")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Exit rate: {exit_rate*100:.1f}%")


if __name__ == "__main__":
    demo_basic()
    demo_modes()
    demo_with_real_data()
    
    print("\n" + "=" * 60)
    print("SEDAC 零配置启动成功!")
    print("=" * 60)
