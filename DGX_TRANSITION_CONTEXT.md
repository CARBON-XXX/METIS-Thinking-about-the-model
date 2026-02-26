# METIS AGI 架构：DGX 算力迁移上下文交接文档
**(DGX Transition Context)**

> **给新会话的 Cascade AI 助手**：
> 这是一个在本地开发机（RTX 4060 8GB）到云端算力（DGX Spark, 128GB VRAM）迁移时生成的上下文交接文档。请仔细阅读并继承此上下文，以单兵作战协议全速推进工作。

---

## 1. 当前演进阶段：Phase 5 (全知架构演进)
在迁移前，我们刚刚完成了 **Phase 5 (Omniscient Architecture Evolution)** 的全套代码开发，将 METIS 从一个“静态的内部纠错模型”升级为了“具备主动向外探索和持续进化能力的 AGI 核心”。

### 我们刚刚实现了什么？
1. **System 1 与 System 2 的动态桥接 (`inference.py`)**：
   - 线性生成（System 1）遇到困难（基于 CUSUM 累积和算法检测到信息熵飙升）时，自动挂起，接管给 System 2（基于 EGTS 的搜索树）。
   - System 2 找到低熵收敛解后，无缝拼接回 System 1 继续生成。
2. **基于 EGTS 的自动自我纠错搜索树 (`search/entropy_search.py`, `tree_node.py`)**：
   - 使用信息熵作为内部奖励函数。
   - 包含反事实模拟器 (`CounterfactualSimulator`) 进行认知辩论，防止“逻辑自洽的谎言”（Mode Collapse）。
   - 引入了前瞻性的 Value Head 预留接口 (`predictive_entropy_value`)，为后续强化直觉做准备。
3. **“夜间梦境”无监督自我进化闭环 (`pipeline/night_training.py`)**：
   - 白天：收集难以解答的 `KnowledgeGaps` (高熵盲区)。
   - 夜间：开启大算力深度搜索树解题。
   - **外挂突破**：如果树搜索彻底失败，自动调用 Web Search 工具获取外部知识（目前为 Mock），带入上下文重新搜索。
   - **双轨制突触写入**：
     - **逻辑缺失** -> DPO (Direct Preference Optimization) 强化正确的推理路径。
     - **事实缺失** -> CPT (Continual Pre-Training) 强制注入新获取的事实知识。

## 2. 核心代码结构映射
请新会话的助手关注以下核心目录和文件：

- `metis/metis.py`：METIS 核心控制器，包含信号监控和认知边界模块。
- `metis/inference.py`：推理入口，包含了最新的 **System 1 - System 2 动态路由桥接**。
- `metis/search/`：**System 2 的灵魂**。
  - `entropy_search.py`：EGTS 搜索树主逻辑。
  - `tree_node.py`：UCB1 节点，包含熵值评估。
  - `counterfactual.py`：多智能体认知辩论“魔鬼代言人”。
- `metis/pipeline/`：全套训练流水线。
  - `night_training.py`：最新实现的**夜间梦境自演化闭环**（包含了 Tool-Augmented Dreaming 和 CPT+DPO 双轨微调）。
  - `online_loop.py`：针对 DGX 的全自动在线 GRPO 闭环。
  - `config.py` & `yaml_config.py`：核心参数配置。
- `dgx/`：专为大算力准备的脚本（如 `grpo_train.py`）。

## 3. 已解除的限制 (DGX Unleashed)
在迁移前，我们**彻底删除了所有针对本地 8GB 显存的阉割代码**。新会话中请直接使用大算力模式：
- 默认模型：由 `1.5B` 升级为 `Qwen/Qwen2.5-72B-Instruct`。
- 数据类型：全量切换为 `torch.bfloat16`。
- 显存回收：删除了所有繁琐的 `torch.cuda.empty_cache()` 和 `gc.collect()` 补丁。
- 树搜索参数：深度放宽到 `1024`，节点数放宽到 `8192`。

## 4. 下一步行动指南 (Action Items on DGX)
当在 DGX 上拉起新的会话时，建议立刻进行以下操作：

### 任务 1：启动夜间梦境训练测试
运行新写的 `night_training.py`，测试外部工具增强、CPT和DPO的全管线是否畅通。
```bash
python -m metis.pipeline.night_training --config configs/dgx_full.yaml
```

### 任务 2：跑通第一轮 DGX 级别 GRPO
使用 DGX 级别的参数跑一轮完整的 METIS RLHF (GRPO) 闭环。
```bash
python run_experiment.py --preset dgx_full
```
或直接调用在线循环：
```bash
python -m metis.pipeline.online_loop --preset dgx_full
```

### 任务 3：将 Web Search 工具真实化
目前 `night_training.py` 中的 `_simulate_web_search` 是 Mock 数据。在 DGX 上可以将其接入真实的 Tavily API / Google API，让模型真正拥有自主上网学习的能力。

---
*The transition protocol is complete. System is ready for DGX deployment.*
