# AIMENG 一键学生训练流水线

这是日常训练唯一推荐的入口：`scripts/run_student_pipeline.py`。它把预检、基线评测、SFT、候选评测和运行记录整合成一次命令，减少手工复制参数导致的训练浪费。默认使用 LoRA 参数高效微调，只更新少量适配器参数，以减少优化器内存和训练成本；评测器可以直接加载生成的 LoRA 适配器。

## 一次命令完成训练与评测

在仓库根目录运行：

```bash
python scripts/run_student_pipeline.py \
  --data data/verified/approved_candidates.jsonl \
  --model /path/to/trainable-hf-student \
  --heldout eval/heldout.jsonl \
  --regression eval/regression.jsonl \
  --output runs/student-2026-10-09 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 8 \
  --max-length 1024 \
  --gradient-checkpointing
```

默认采用 LoRA（r=8、alpha=16）。只有确认训练资源充足、确实需要更新全部学生参数时，才额外添加 `--full-finetune`。不要把教师 9B 推理工件直接当成学生模型。\n\n只需要替换四个路径/名称：已核验候选数据、兼容的可训练 Hugging Face causal-LM checkpoint、冻结 held-out 测试集、冻结旧能力回归集，以及一个全新的输出目录名。输出目录必须不存在；脚本不会覆盖旧运行。

## 自动执行顺序

1. 检查输入文件和参数，复制数据与评测集快照。
2. 运行候选数据结构校验。
3. 运行训练 dry-run，确认存在已核验且获准的 train 与 validation 数据，并检查 split 泄漏。此步骤不加载模型权重。
4. 在原始学生模型上运行基线评测，尽量在训练前发现评测集或模型接口问题。
5. 执行 SFT，并保存 checkpoint、数据哈希、配置和日志。
6. 在相同快照上评测候选模型，生成可比较的 baseline/candidate 指标。
7. 保存 `pipeline_config.json` 及每一步的日志；任一步失败就停止后续步骤并保留现场，避免盲目继续训练。

## 安装环境

先在训练机/训练云环境按其硬件选择匹配的 PyTorch，再安装本项目的可选依赖：

```bash
pip install -r requirements-sft.txt
```

不要把 9B 教师推理包误当成学生 checkpoint。此入口只支持标准 Hugging Face causal LM；Ornith 可以作为教师生成演示，但若其权重是 GGUF/MLX 推理工件，不可直接当作此处的可训练学生模型。

## 训练前的数据要求

- 每条可训练记录必须是独立核验通过的样本，并具备非空验证方法和证据。
- 每条样本必须明确分配 train、validation 或 test；同一 task_id 不可跨 split。
- train 和 validation 都必须非空；test 永远不会被 optimizer 使用。
- 新导入的教师输出默认 pending/unassigned/ineligible。不能因为教师说“我已验证”就自动授权训练。
- held-out 与 regression 使用固定的 `aimeng.eval_task.v1` JSONL 格式，两个集合 task_id 不可重叠。

## 训练后仍需单独完成的事

该流水线完成训练与质量评测，但不会假装已经完成设备部署或自动晋级。候选指标里的内存字段会明确标记为未测量。需要在目标 Android 手机上测量整应用峰值 RAM，再使用 `scripts/gate_student_release.py` 执行晋级门。AIMENG 手机运行时的硬约束是整应用峰值低于 4096 MiB；晋级门默认要求低于 3800 MiB 留出余量。模型文件大小不能替代实测。

若流水线失败，先检查输出目录里的 `pipeline_config.json` 和 `logs/`，修复原因后使用一个新的输出目录重跑。不要在失败目录上盲目续跑，因为当前脚本有意拒绝模糊的覆盖/恢复操作。
