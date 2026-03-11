# METIS DPO 训练排障与技术总结报告

## 1. 核心目标
在 NVIDIA DGX 节点（单卡 128GB 统一内存）上，针对 `Qwen2.5-7B-Instruct` 模型进行 Direct Preference Optimization (DPO) 训练。目标是通过扩展数据集（1000 prompts）和优化超参数，提升模型的认知对齐水平，同时避免因显存溢出（OOM）导致的死机。

---

## 2. 遇到的所有问题清单 (Issue Tracker)

| 编号 | 现象 / 错误 | 根本原因 (Root Cause) | 解决方案 (Solution) |
| :--- | :--- | :--- | :--- |
| **#1** | **HuggingFace 无法连接** <br> 网络超时导致模型/分词器下载失败 | 国内直连 HuggingFace 服务器的网络不稳定。 | 在所有执行脚本头部加入环境变量强制使用镜像站：`export HF_ENDPOINT=https://hf-mirror.com` |
| **#2** | **"Memory-constrained mode" 被误触发** <br> 128GB 的 DGX 仍以 Batch=1 龟速运行 | `trainer_phase.py` 中的显存检测逻辑使用 `free_gb` (剩余显存) 判断。由于基础模型已加载，剩余显存低于 80GB，触发了限制模式。 | 修复检测逻辑，改为通过 `total_gb` (总物理显存) 判断环境等级，解除限制。 |
| **#3** | **DGX 第一次硬死机** <br> Batch Size=8, 2048上下文下直接死锁 | DPO 需要同时进行策略模型和参考模型的前向传播。如果不开启梯度检查点 (Gradient Checkpointing)，激增的激活值 (Activations) 会瞬间打爆 128GB 显存。 | 强制开启 `gradient_checkpointing = True`，用计算换空间。 |
| **#4** | **DGX 第二次硬死机** <br> 尝试将 Batch Size 拉升至 16 加速时再次死机 | DPO 双倍前向传播的特性下，Batch 16 + 2048 Context + 7B 模型的极限激活值依然超过了 128GB 的物理边界。 | 下调 `dpo_max_length` 至 1536，`dpo_batch_size` 稳在 8，`grad_accum` 设为 4。显存压制在 60-70GB 的安全甜点区。 |
| **#5** | **评估基准退化 (Alignment Tax)** <br> 之前跑完的模型出现基础能力下降 | DPO 的 KL 散度惩罚过小，且训练数据量不足（之前只有 144 条有效）。 | 数据集规模拉升至 1000 组，`dpo_beta` 从 0.1 提高到 0.25 以保护原始校准度。 |
| **#6** | **终端打分阶段无进度可见** <br> 日志被吞，无法确认 Reward Margin 收敛情况 | `transformers.Trainer` 的 `report_to="none"` 且标准输出未捕获各项奖励拆解。 | 注入 `MetricsJsonCallback`，劫持 `on_log` 事件并将数据吐到 JSON；同时手搓 Flask + Chart.js 实时监控大屏面板。 |

---

## 3. 核心技术原理解析 (Technical Deep Dive)

### 3.1 DPO 训练的显存机制 (Why 128GB is not enough?)
在标准的 SFT (Supervised Fine-Tuning) 中，显存占用主要由**模型权重**、**梯度**、**优化器状态 (AdamW)** 和**激活值 (Activations)** 组成。
但在 DPO (Direct Preference Optimization) 中，显存占用极其特殊：
1. **双模型常驻**：内存中必须同时存放正在训练的 Policy Model (带梯度) 和冻结的 Reference Model (无梯度)。
2. **2x 前向传播**：对于每一对数据 (Chosen 和 Rejected)，Policy 和 Reference 都要各跑一次前向传播。等效于在一次 step 中处理的 token 量翻了数倍。
3. **激活显存爆炸**：Transformer 的 Self-Attention 复杂度是 $O(L^2)$（$L$ 为 Context Length）。在 $L=2048$、Batch=16 且双模型的情况下，前向传播产生的激活值会呈指数级爆炸，瞬间从 30GB 飙升突破 120GB，导致 Linux 内核直接触发 OOM-Killer 甚至内核死锁（Hard Freeze）。

### 3.2 显存防御与压榨策略 (Memory Mitigation)
- **Gradient Checkpointing (GC)**：
  这是一种必须开启的核心技术。它在反向传播时重新计算某些层的激活值，而不是把前向传播的所有激活值都存在显存里。**代价是多花费 ~20% 的计算时间，但能节省 70% 的激活显存。**
- **Batch Size 与 Context Length 平衡**：
  将长度从 2048 砍到 1536（绝大多数对话已足够），并把 Micro Batch Size 锁在 8。这样单步峰值显存被死死按在 64GB 左右。然后通过 `gradient_accumulation_steps=4`，在逻辑上维持 Global Batch Size = 32（8x4），保证梯度下降的方向稳定。

### 3.3 实时监控架构 (Real-time Telemetry)
DPO 训练不同于 SFT 只看 Loss，**必须监控 Rewards Margin（偏好间隔）和 Accuracies（准确率）**。
- **采集层**：继承 `TrainerCallback`，每次 `logging_steps` 时拦截 `logs` 字典，追加写入物理文件 `training_metrics.json`。
- **服务层**：使用 Python Flask 构建轻量级 Web API，提供 `/api/metrics` 接口暴露时间序列数据。
- **渲染层**：前端轮询拉取数据，使用 Chart.js 将 `rewards/chosen`、`rewards/rejected` 分离度可视化。当 Margin 从 0 逐步爬升并趋于平缓时，意味着模型已经学到了偏好规则。

---

## 4. 当前最优生产级配置 (Production Config)
针对 128GB GPU 训练 7B 模型 DPO 的安全基线配置：
```python
dpo_epochs = 3
dpo_learning_rate = 1e-6
dpo_batch_size = 8                 # 物理微批次
dpo_gradient_accumulation = 4      # 有效批次 = 32
dpo_beta = 0.25                    # KL 散度约束
dpo_max_length = 1536              # 安全上下文截断
gradient_checkpointing = True      # OOM 防御盾
lora_r = 64
lora_alpha = 128
```

---

