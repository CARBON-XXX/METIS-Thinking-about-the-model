# SEDAC V9.0 Production Module

生产级语义熵引导动态注意力核心引擎

## 概述

SEDAC (Semantic Entropy-guided Dynamic Attention Core) V9.0 Production 是一个符合 NVIDIA TensorRT / Triton Inference Server 标准的生产级推理加速框架。

### 核心特性

- **高性能 CUDA 内核**: 优化的熵计算和决策内核
- **自适应阈值控制**: 动态调整早退阈值
- **Ghost KV 生成**: 预测跳过层的 KV Cache
- **O1 深度推理**: 高熵问题的自适应计算
- **完整指标监控**: Prometheus 兼容的性能监控

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 基本使用

```python
from sedac.v9.production import create_pipeline, GenerationConfig

# 创建推理管线
pipeline = create_pipeline("Qwen/Qwen2.5-7B-Instruct")

# 生成
result = pipeline("What is AI?", GenerationConfig(max_new_tokens=256))
print(result.generated_text)
print(f"TPS: {result.tokens_per_second:.1f}")
print(f"Skip Ratio: {result.skip_ratio*100:.1f}%")
```

### 启动 API 服务器

```bash
python -m sedac.v9.production.run serve --model Qwen/Qwen2.5-7B-Instruct --port 8000
```

### 运行基准测试

```bash
python -m sedac.v9.production.run benchmark --model Qwen/Qwen2.5-7B-Instruct
```

### 训练 Ghost KV

```bash
python -m sedac.v9.production.run train --model Qwen/Qwen2.5-7B-Instruct --epochs 3
```

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    SEDACInferencePipeline                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │   Tokenizer │  │    Model     │  │  ProductionSEDAC │    │
│  │             │  │  (Qwen/LLaMA)│  │     Engine       │    │
│  └─────────────┘  └──────────────┘  └─────────────────┘    │
│                                              │               │
│                         ┌────────────────────┼───────┐      │
│                         ↓                    ↓       ↓      │
│                  EntropyComputer    GhostKVGen  O1Controller│
│                         │                    │       │      │
│                         └────────────────────┴───────┘      │
│                                     │                        │
│                              MetricsCollector                │
└─────────────────────────────────────────────────────────────┘
```

## API 参考

### ProductionConfig

```python
from sedac.v9.production import ProductionConfig

config = ProductionConfig(
    device="cuda:0",
    precision=PrecisionMode.FP16,
)
```

### SEDACInferencePipeline

```python
from sedac.v9.production import SEDACInferencePipeline, GenerationConfig

pipeline = SEDACInferencePipeline("model_name")
pipeline.load()

# 单次推理
result = pipeline("prompt", GenerationConfig())

# 批量推理
results = pipeline.batch_generate(["p1", "p2", "p3"])

# 流式生成
for chunk in pipeline.stream_generate("prompt"):
    print(chunk, end="")
```

### ProductionSEDACEngine

```python
from sedac.v9.production import ProductionSEDACEngine, ProductionConfig

config = ProductionConfig()
engine = ProductionSEDACEngine(config)
engine.attach_model(model)

output = engine.forward_with_sedac(input_ids)
```

## 性能指标

### 期望性能 (Qwen2.5-7B, RTX 4090)

| Batch Size | Baseline TPS | SEDAC TPS | Speedup | Skip Ratio |
|------------|--------------|-----------|---------|------------|
| 1          | ~50          | ~75       | 1.5x    | ~30%       |
| 4          | ~180         | ~290      | 1.6x    | ~35%       |
| 16         | ~600         | ~1000     | 1.7x    | ~40%       |

### 延迟目标

- 熵计算: < 0.1ms
- 早退决策: < 0.05ms
- Ghost KV 生成: < 0.2ms

## 部署

### Docker

```bash
docker build -t sedac:v9 -f Dockerfile .
docker run --gpus all -p 8000:8000 sedac:v9
```

### API 端点

- `POST /generate` - 生成文本
- `POST /batch_generate` - 批量生成
- `GET /health` - 健康检查
- `GET /metrics` - 性能指标
- `GET /metrics/prometheus` - Prometheus 格式

## 配置

详见 `config.yaml` 配置文件。

## 测试

```bash
# 单元测试
python -m sedac.v9.production.tests

# 集成测试
python -m sedac.v9.production.integration_test --model Qwen/Qwen2.5-0.5B-Instruct
```

## 许可证

MIT License
