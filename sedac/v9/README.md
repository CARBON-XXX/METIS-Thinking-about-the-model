# SEDAC V9.0 - Speculative Early-exit with Dynamic Adaptive Control

## 概述

SEDAC V9.0 是一个高性能的 LLM 早退出推理引擎，通过动态判断 token 的计算复杂度，在保持输出质量的同时大幅减少计算量。

### 核心特性

- **CUDA 加速决策**: 融合熵决策内核，延迟 <1ms
- **普适性接口**: 支持任意 HuggingFace 模型 (LLaMA, Qwen, Mistral 等)
- **Ghost KV 生成器**: 轻量级 MLP 预测跳过层的 KV Cache
- **Attention Sink 保护**: 保护关键 token 不被提前退出

## 性能基准

| 场景 | 延迟 | 状态 |
|------|------|------|
| 32 tokens | 0.019ms | ✅ |
| 256 tokens | 0.218ms | ✅ |
| 1024 tokens | 0.775ms | ✅ |
| 4096 tokens | 3.27ms | - |

## 安装

### 1. 编译 CUDA 扩展

```bash
cd sedac/v9/cuda_ext
python setup_v2.py build_ext --inplace
```

### 2. 验证安装

```bash
python tests/test_cuda_kernels.py
```

## 快速开始

```python
from transformers import AutoModelForCausalLM
from sedac.v9.core import create_sedac_engine

# 加载模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# 创建 SEDAC 引擎
engine = create_sedac_engine(
    model,
    exit_threshold=0.7,    # 退出阈值
    min_exit_layer=4,      # 最小退出层
    anchor_interval=4,     # 锚点间隔
    verbose=True,
)

# 推理
result = engine.forward(input_ids, attention_mask=attention_mask)
print(f"Skip ratio: {result['skip_ratio']*100:.1f}%")
```

## 项目结构

```
sedac/v9/
├── core/
│   ├── __init__.py          # 模块导出
│   ├── sedac_engine.py      # 主引擎 (普适性接口)
│   └── ghost_kv.py          # Ghost KV 生成器
├── cuda_ext/
│   ├── sedac_kernels_v2.cu  # 优化 CUDA 内核
│   ├── sedac_ops_v2.cpp     # PyTorch 绑定
│   ├── setup_v2.py          # 编译脚本
│   └── *.pyd                # 编译产物
└── tests/
    ├── test_cuda_kernels.py # CUDA 单元测试
    └── test_qwen_integration.py
```

## CUDA 内核优化

### V2 优化策略

1. **Warp Shuffle Reduction**: 使用 `__shfl_down_sync` 替代共享内存规约
2. **向量化加载**: 使用 `float4` 批量加载数据
3. **动态 Block Size**: 根据 vocab_size 自动选择最优配置
4. **融合计算**: 2-pass 替代 3-pass (Max + Fused SumExp/Entropy)
5. **FP16 支持**: 半精度计算路径

### API

```python
import sedac_cuda_v2

# 融合熵决策
entropy, confidence, decision, load = sedac_cuda_v2.fused_entropy_decision_v2(
    logits,        # [N, vocab]
    hidden,        # [N, hidden]
    prev_hidden,   # [N, hidden]
    mean_entropy,  # float
    std_entropy,   # float
    layer_progress,# float (0~1)
    threshold,     # float
)

# Token 路由分割
active_h, active_i, exit_h, exit_i = sedac_cuda_v2.token_router_split_v2(
    hidden,        # [N, hidden]
    decision,      # [N] bool
)

# Token 路由合并
output = sedac_cuda_v2.token_router_merge_v2(
    active_hidden, active_indices,
    exit_hidden, exit_indices,
    total_size
)

# 批量处理
entropy, conf, dec, load = sedac_cuda_v2.batched_entropy_decision(
    logits,        # [batch, seq, vocab]
    hidden,        # [batch, seq, hidden]
    prev_hidden,   # [batch, seq, hidden]
    ...
)
```

## Ghost KV 训练

```python
from sedac.v9.core import create_ghost_kv_for_model, GhostKVTrainer

# 创建 Ghost KV 生成器
ghost_kv = create_ghost_kv_for_model(model)

# 训练器
trainer = GhostKVTrainer(ghost_kv, model, ghost_kv.config)

# 训练
for epoch in range(10):
    losses = trainer.train_epoch(dataloader, skip_start_layer=4, num_skip_layers=2)
    print(f"Epoch {epoch}: loss={losses['loss']:.4f}")

# 保存
trainer.save("ghost_kv.pt")
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `exit_threshold` | 0.7 | 退出阈值，越低越激进 |
| `min_exit_layer` | 4 | 最小退出层 |
| `anchor_interval` | 4 | 锚点层间隔 |
| `protect_first_n` | 2 | 保护前 N 层 |
| `protect_last_n` | 1 | 保护后 N 层 |
| `attention_sink_tokens` | 4 | 注意力汇聚 token 数 |

## 系统要求

- CUDA 12.0+
- PyTorch 2.0+
- RTX 3000/4000 系列 GPU
- Windows/Linux

## License

MIT
