# AIMENG：训练优先的 AI 学习框架

**当前优先级：先训练并验证 AI 的能力，再决定手机端部署方案。** 本仓库负责训练数据、任务轨迹、策略学习、知识获取、学生模型实验与独立评测；独立仓库 [bslsjdk/ai](https://github.com/bslsjdk/ai) 的 Android 运行软件放到模型能力达标后再适配。模型权重格式由未来的训练产物与目标运行后端共同决定，AIMENG 不把 GGUF、MLX 或任何单一格式作为系统前提。

现有 GGUF/llama.cpp 脚本保留为**历史研究基线**，用于可复现的预算控制与策略实验，不代表最终模型架构，也不代表已经完成 9B 参数训练。当前仓库里训练轻量控制器与保存行为经验的能力，和训练一个新学生语言模型是不同阶段。执行顺序见 [训练优先路线图](docs/TRAINING_FIRST_ROADMAP.md)。

## 旧基线模型（不是最终格式决策）

现有一批可复现实验以 `ornith-ai/Ornith-1.5-9B-GGUF` 的 `Ornith-1.5-9B-Q4_K_M.gguf` 为教师基线。保留它是为了延续既有采集器与预算控制实验，不意味着最终训练目标或 Android 部署必须使用 GGUF。模型文件不会提交进 Git；旧实验脚本仍可能要求兼容的 `llama-cli`。

## 下载并检查

在有足够磁盘空间和网络的训练环境运行：

```bash
python -m pip install -r requirements.txt
python scripts/fetch_model.py
python scripts/inspect_gguf.py --output runs/gguf_manifest.json
```

下载脚本会检查至少 8 GiB 可用磁盘空间，并使用 Hugging Face Hub 缓存。也可以设置 `AIMENG_MODEL_GGUF` 指向已有的本地 GGUF 文件，再运行检查脚本。

## 重要训练边界

旧实验中的 GGUF 是量化推理工件，不是常规梯度训练检查点。现有脚本用它收集任务质量、延迟和资源数据，并训练独立的轻量控制器。**这不等于已经训练了新的通用语言模型。** 若要训练学生模型，需要另外选择可训练的模型检查点、数据与训练方法；最终部署格式以后再决定。

要让路由器真正选择 FFN 神经元组并节省计算，需要执行后端暴露相应内部结构并真正跳过结构化矩阵计算。仅训练一个外部分类器或在逻辑上屏蔽通道，不足以证明节省了计算或内存。

## 阶段路线

当前主路线以 [训练优先路线图](docs/TRAINING_FIRST_ROADMAP.md) 为准：先盘点资源、建立高质量数据与验证集，跑通可量化的行为学习实验，再独立决定是否训练学生模型/蒸馏。旧 GGUF 预算实验属于辅助基线；动态跳算与 Android 运行时集成均后置，不能抢在模型能力和训练数据验证之前。

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


## 旧基线：9B 行为层自我学习实验

此实验展示如何在冻结的旧基线模型上启用持久化策略记忆，不需要从零训练基础模型；它是行为策略实验，不是最终学生模型训练：

```bash
python scripts/collect_budget_traces.py \\
  --model models/Ornith-1.5-9B-Q4_K_M.gguf \\
  --tasks examples/tasks.example.jsonl \\
  --output runs/strategy-traces.jsonl \\
  --strategy-memory runs/verified-experiences.jsonl
```

策略记忆只接收带独立验证器的 pass/fail 结果，并以 full-budget 试验更新经验；第一次没有足够历史时使用 direct 策略。该功能目前属于实验性行为层学习，必须用独立任务集验证是否真的提升质量或速度。Android 整应用 4096 MiB 限制仍然有效，当前采集器的子进程内存数据不能证明手机端满足该限制。

## 在线学习机制（研究基线）

新增 [在线学习闭环设计](docs/ONLINE_LEARNING_LOOP.md)：区分任务内临时适应与任务后持久巩固，使用独立验证器产生误差信号，并将质量和 4096 MiB Android 运行时限制设为不可被奖励抵消的硬门槛。该文档是设计规范。`scripts/online_adaptation.py` 已实现首个可单测的任务内决策器，但尚未接入真实推理执行器，也不代表在线学习已在 Colab 或手机端完成端到端验证。


## 有界动态工作集与 9B 知识导入（新实验分支）

- 设计约束：[神经科学启发的有界运行时](docs/NEUROINSPIRED_BOUNDED_RUNTIME.md) 把任务相关路由、固定工作集、经验巩固和真实内存验证分开定义。
- 资源池原型：`scripts/bounded_workspace.py` 提供固定槽位、字节预算、LRU 替换与 pinned 保护。它只管理元数据，**尚未接入 GGUF 后端，不代表真实权重卸载或 RAM 已降低**。
- 9B 知识路径：[教师知识导入设计](docs/TEACHER_KNOWLEDGE_PIPELINE.md) 区分可检索事实库、任务轨迹和后续参数蒸馏。
- 知识记录检查/去重：`python scripts/prepare_knowledge_corpus.py --input data/teacher-knowledge.jsonl --output data/knowledge-clean.jsonl`。该脚本只做结构校验和确定性去重，不验证事实本身；使用 `--verified-only` 可只导出带外部来源/验证器信息的记录。
- 新增测试：`tests/test_bounded_workspace.py` 覆盖工作集准入、LRU、锁定和知识数据去重。需要运行完整测试套件后再判断是否通过。

当前这一步建立的是可测试的框架与知识数据入口，不是已完成的 Android 动态权重分页实现。4096 MiB 仍须以整应用设备实测为准。

- 教师抽取提示模板：[prompts/teacher_knowledge_extraction.md](prompts/teacher_knowledge_extraction.md)，以及示例 JSONL：[examples/knowledge-records.example.jsonl](examples/knowledge-records.example.jsonl)。示例中的模型哈希是占位符，导入真实数据前必须替换。


## 运行时仓库边界修正（2026-10-09）

本仓库的旧实验基线仍包含 GGUF/llama.cpp 脚本，但 **AIMENG 不再把 GGUF 视为整个系统的固定前提**。实际 Android 运行软件是独立仓库 [bslsjdk/ai](https://github.com/bslsjdk/ai)，应以它的当前实现与 [项目工作记忆](https://github.com/bslsjdk/ai/blob/main/docs/PROJECT_MEMORY.md) 为准。该运行时当前指定的首选模型是 Ornith-1.5-9B-MLX-4bit（Safetensors / affine 4-bit）；GGUF/llama.cpp 保留为旧基线/兼容路径。

- [AIMENG 与 AI 运行时集成契约](docs/RUNTIME_INTEGRATION_CONTRACT.md) 明确两个仓库的职责、模型工件可替换原则、训练产物与部署工件的区别，以及下一步必须从真实运行时代码确认的接口。
- 训练侧的知识记录和任务轨迹使用与模型权重格式无关的 schema。只有部署适配层需要知道运行时具体接受什么格式。
- 当前工作集是调度器原型，不是实际动态权重分页。后续必须对接 `bslsjdk/ai` 的真实加载、generate、KV/预算和性能遥测入口，再验证整应用 4096 MiB 硬约束。


## 训练数据基础设施（模型格式无关）

- [训练优先路线图](docs/TRAINING_FIRST_ROADMAP.md)：先构建与验证 AI 能力，再决定学生模型和手机部署格式。
- [任务轨迹 schema](schemas/task_trajectory.schema.json)：定义输入、候选目标、独立验证结果、来源、隐私审查与数据拆分。
- [训练数据审计脚本](scripts/validate_training_data.py)：检查 JSONL 结构、重复 sample ID、pass/fail 是否具备独立验证证据，以及同一 task_id 是否跨 train/validation/test 泄漏。
- [轨迹示例](examples/trajectory.example.jsonl)：仅为格式演示，状态为 unknown，不是可直接用于监督训练的正例。

运行数据审计：
```bash
python scripts/validate_training_data.py --input examples/trajectory.example.jsonl --report runs/data-audit.json
```

此脚本只检查结构和数据拆分，不证明目标答案正确、来源许可有效或模型训练成功。只有带有可靠独立标签的数据才应进入监督训练。


## 自有 AI 架构研究（主方向）

本项目目标不是只给现成大模型添加插件，而是研究持续学习、任务相关计算、受限工作集和可恢复状态共同组成的自有 AI 架构。当前设计基线：
- [自学习架构与运行平台设计](docs/SELF_LEARNING_ARCHITECTURE.md)
- [自有模型工件格式 v0 草案](docs/ARTIFACT_FORMAT_V0.md)

这些是设计基线，不代表机制已经实现。现有 GGUF/Ornith 工作流只作为教师或比较基线。先把学习闭环、格式原型和资源测量做成可验证实现，再扩大模型和接入 Android；不提前锁定任何现成模型格式。


## 并发训练实验（新原型）

- [并发训练设计与验收方法](docs/PARALLEL_TRAINING.md)：明确独立专家并行、同一模型分布式训练和错误的多优化器并发之间的区别。
- scripts/parallel_training.py：按声明的 RAM/VRAM 预算与最大 worker 数调度独立训练进程，分别保存日志/输出并生成运行报告。
- examples/parallel-training-plan.example.json：可运行的双任务示例，使用 scripts/toy_train.py 训练两个独立的极小分类模型并分别保存 checkpoint。该示例只验证并发管线，不是语言模型训练。
- scripts/toy_train.py：纯 Python 训练 smoke test，分别拟合两种不同的合成分类任务。
- tests/test_parallel_training.py：覆盖计划校验、超预算拒绝、输出目录隔离、dry-run，以及两个独立小模型并发训练和 checkpoint 分离。

先只运行 --dry-run 验证配置，再用两个小任务比较串行与并发的真实耗时、峰值内存和验证质量。这里的内存数值是调度预算，不是操作系统强制限制；该原型没有实现梯度同步，也没有声称已经加速模型训练。


## 资源感知奖励实验（设计提案）

[资源感知奖励机制](docs/RESOURCE_AWARE_REWARD.md) 记录了“难题解出后给予额外奖励、资源占用扣分”的离线实验方案。难度奖励必须由固定基准模型的重复失败率和独立验证支持；任何硬内存/计算限制都不能被奖励抵消。该文档目前只是研究提案，尚未接入训练器，也没有声称已证明有效。


## Ornith -> AIMENG 学生蒸馏基础框架

可执行流程与当前限制见 [蒸馏运行手册](docs/DISTILLATION_RUNBOOK.md)。新增 `scripts/import_teacher_demos.py` 用于导入本地/远程教师生成的 JSONL；导入记录默认全部 pending、unassigned、ineligible。训练入口 `scripts/train_student_sft.py` 已加入 task_id 拆分泄漏检查、assistant-only loss mask、数据 SHA-256 与 run manifest，并兼容常见 Transformers 评估参数版本。CI 检查两批候选数据和关键训练门槛。

**边界仍然重要：**这建立了可验证的通用 Hugging Face causal-LM response-SFT 路径，不代表已经接通 Ornith 自动推理，也不代表 AIMENG 自有非标准学生架构已原生训练。首次真实训练仍需可训练学生 checkpoint、经独立核验并完成拆分的数据，以及一次端到端 smoke test。


学生评测与晋级另有可执行入口：`scripts/evaluate_student_sft.py` 在固定 held-out/regression JSONL 上生成可比较指标，`scripts/gate_student_release.py` 只有在质量提升、回归受控和 Android 整应用内存实测通过时才更新版本指针。具体格式与命令见 [蒸馏运行手册](docs/DISTILLATION_RUNBOOK.md)。


## 一键训练入口

日常训练请优先使用 [一键学生训练流水线](docs/ONE_COMMAND_TRAINING.md)：`scripts/run_student_pipeline.py` 默认使用 LoRA 进行参数高效微调，并自动预检数据、运行原模型基线评测、执行 SFT、评测候选模型及保存输入快照与分步日志；需要全量微调时显式传入 `--full-finetune`。它会在关键门槛失败时停止，不覆盖已有运行。Android 整应用内存必须另行实测，训练成功不会自动晋级模型。

## 免费 GPU：Ornith 教师生成 → 学生训练

新增 [免费 GPU 蒸馏操作手册](docs/FREE_GPU_ORNITH_TO_STUDENT.md)、教师生成器 `scripts/generate_ornith_teacher_demos.py` 和任务模板 `data/teacher_seed/gpu_tasks.example.jsonl`。教师路径已调整为官方量化仓库 `ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M`，由 CUDA 版 `llama-server` 加载，不再加载原始 Transformers BF16 checkpoint 后临时量化。生成的数据全部保持待核验、未拆分、不可训练。先用 `--limit 2` 做 smoke test，并检查启动日志确认 GPU offload，再生成批次；独立核验和数据拆分完成后，才进入既有一键学生训练流水线。当前已提交生成器与流程代码，但尚未在真实 Kaggle GPU 上实测成功。
