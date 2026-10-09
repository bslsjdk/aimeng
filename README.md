# Aimeng：GGUF 基础模型的动态神经元调度训练

本仓库训练轻量级神经元组调度器，并评估任务质量、推理成本和内存。目标基础模型固定为 **Ornith-1.5-9B-GGUF Q4_K_M**，使用 GGUF 路径而非 Apple 专用 MLX 执行路径。

## 已固定的模型

- Hugging Face 仓库：`ornith-ai/Ornith-1.5-9B-GGUF`
- 文件：`Ornith-1.5-9B-Q4_K_M.gguf`
- 公开页面列出的文件大小约 5.78 GB。
- 模型不会提交进 Git；通过脚本下载到被忽略的本地 `models/` 目录。

## 下载并检查

在有足够磁盘空间和网络的训练环境运行：

```bash
python -m pip install -r requirements.txt
python scripts/fetch_model.py
python scripts/inspect_gguf.py --output runs/gguf_manifest.json
```

下载脚本会检查至少 8 GiB 可用磁盘空间，并使用 Hugging Face Hub 缓存。也可以设置 `AIMENG_MODEL_GGUF` 指向已有的本地 GGUF 文件，再运行检查脚本。

## 重要训练边界

GGUF 是量化推理格式，不是常规梯度训练检查点。我们会把该 Q4_K_M GGUF 用作冻结教师/推理基线，收集可复现的任务质量、延迟、资源数据，并训练独立的轻量路由器。**不会声称直接对 Q4 GGUF 做常规反向传播。**

要让路由器真正选择 FFN 神经元组并节省计算，需要执行后端暴露相应内部结构并真正跳过结构化矩阵计算。仅训练一个外部分类器或在逻辑上屏蔽通道，不足以证明节省了计算或内存。

## 阶段路线

1. 锁定模型身份、文件 SHA-256 和 GGUF 张量清单。
2. 建立完整模型的任务质量、首 token 延迟、token/s、峰值 RSS/PSS 基线。
3. 先用后端真实支持的上下文、批大小、最大生成 token、线程和并发做 small/medium/full 配对预算实验。
4. 通过客观验证器整理遥测，训练轻量预算控制器并在独立 test 集上与 full-budget 基线比较。
5. 只有证明后端确实跳过了结构化矩阵计算后，才研究通道组/层级动态跳算。
6. 实施候选池动态扩容、质量门控和安全回退。
7. 在目标 Android 设备上验证实际收益，再独立集成到运行时仓库。

## 4 GiB 运行时约束

整个 Android 软件运行时峰值不得超过 4096 MiB。训练机内存不等于手机运行时内存；GGUF 文件大小、mmap 虚拟地址空间也都不能代替实际峰值测量。必须考虑权重驻留页、KV cache、执行缓冲区和后端/NPU 分配，并留出安全余量。

## 当前状态

训练仓库已建立模型检查、下载和实验规范。模型下载是否完成、路由器是否训练成功、内存是否下降，都必须以实际运行记录为准；仓库文件本身不能证明这些结果。


## 实验执行脚本（研究原型）

- `scripts/validate_telemetry.py`：检查 JSONL 字段、重复 run ID、预算配对和任务拆分泄漏。
- `scripts/collect_budget_traces.py`：调用兼容的 `llama-cli` 对同一任务跑 small/medium/full 三档；默认不请求 GPU offload，只有确认 CUDA 构建后才显式传 `--gpu-layers 99`。
- `scripts/train_budget_controller.py`：从客观 pass/fail 结果中选择最低可靠预算，训练独立的小型 PyTorch 分类器；不更新 9B GGUF 权重。
- `scripts/evaluate_budget_controller.py`：在独立 `test` 任务上报告预算标签准确率、质量通过率、平均延迟和可用 PSS，并与 full-budget 基线比较。
- `scripts/predict_budget.py`：对单个任务特征做离线预算预测，不负责执行推理或安全回退。
- `scripts/online_adaptation.py`：根据独立验证结果和整应用 PSS，输出任务内下一步动作；未知反馈、内存数据缺失、超时和 OOM 都不会触发盲目升档。它不更新主模型权重，也不直接执行推理。
- `scripts/experience_learning.py`：保存独立验证通过/失败的策略经验，按任务类别学习 `direct/decompose/verify` 提示策略；可通过采集器的 `--strategy-memory` 接入 9B 推理实验。它学习的是策略选择，不修改 GGUF 权重。
- `tests/test_pipeline.py` + GitHub Actions：覆盖基础验证器、训练目标、重复预算质量门槛和测试报告计算。

详细步骤见 [采集与训练操作手册](docs/COLLECT_AND_TRAIN.md)。Colab Notebook 默认不编译后端、不启动耗时采集；需手动启用并检查真实后端日志。当前脚本和测试尚未在实际 Colab/GPU/目标手机上完成端到端验证，不能把代码提交视为模型训练成功。


## 9B 行为层自我学习实验

在已有 9B 模型上启用持久化策略记忆，不需要从零训练基础模型：

```bash
python scripts/collect_budget_traces.py \\
  --model models/Ornith-1.5-9B-Q4_K_M.gguf \\
  --tasks examples/tasks.example.jsonl \\
  --output runs/strategy-traces.jsonl \\
  --strategy-memory runs/verified-experiences.jsonl
```

策略记忆只接收带独立验证器的 pass/fail 结果，并以 full-budget 试验更新经验；第一次没有足够历史时使用 direct 策略。该功能目前属于实验性行为层学习，必须用独立任务集验证是否真的提升质量或速度。Android 整应用 4096 MiB 限制仍然有效，当前采集器的子进程内存数据不能证明手机端满足该限制。

## 在线学习机制

新增 [在线学习闭环设计](docs/ONLINE_LEARNING_LOOP.md)：区分任务内临时适应与任务后持久巩固，使用独立验证器产生误差信号，并将质量和 4096 MiB Android 运行时限制设为不可被奖励抵消的硬门槛。该文档是设计规范。`scripts/online_adaptation.py` 已实现首个可单测的任务内决策器，但尚未接入真实推理执行器，也不代表在线学习已在 Colab 或手机端完成端到端验证。
