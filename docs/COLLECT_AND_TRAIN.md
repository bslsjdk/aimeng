# 实验采集器与第一版预算控制器

## 1. 先准备兼容后端

采集器调用已有的 `llama-cli`，不会替你编译 llama.cpp，也不会假装一定启用了 CUDA。Colab 里先确认：
```bash
llama-cli --version
llama-cli --help
```
必须使用能打开目标 GGUF 的 llama.cpp 构建。若要测 GPU，必须用实际启用 CUDA 的构建并单独确认后端日志；仅传 `--accelerator CUDA-T4` 只是写入标签，不会开启 GPU。

## 2. 准备任务文件

任务文件为 JSONL，一行一个任务，字段包括 `task_id`、`prompt`、`dataset_split`、`task_family`、`verifier`、`expected`。模板见 `examples/tasks.example.jsonl`。默认支持：
- `exact_match`：输出去掉首尾空白后必须与 expected 完全相同；
- `contains`：输出必须包含 expected；
- `json_object`：输出必须能解析为 JSON object；
- 其他 verifier 会标记为 unknown，不进入监督标签。

示例任务只是格式演示，不是正式基准。训练前应替换成独立、无隐私的评测任务，覆盖不同任务族和难度。

## 3. 运行三档预算采集

```bash
python scripts/collect_budget_traces.py \
  --model models/ornith-1.5-9b-q4km/Ornith-1.5-9B-Q4_K_M.gguf \
  --tasks examples/tasks.example.jsonl \
  --output data/telemetry.jsonl \
  --llama-cli llama-cli \
  --accelerator CPU \
  --max-tasks 3
```

每个任务会依次跑 small/medium/full 三档。采集器默认覆盖指定输出文件；显式传 `--append` 才会重复追加；传 `--resume` 则会跳过已有的已完成 task/budget，适合从 Colab 中断处继续；恢复时必须提供匹配的 `--model-sha256` 与 `--dataset-sha256`，防止不同模型或任务集混在一起。采集器记录端到端耗时，并尝试从 Linux `/proc/<pid>/status` 与 `/proc/<pid>/smaps_rollup` 采样子进程 RSS/PSS。后端实际生效的上下文、TTFT、token/s、GPU 显存目前没有可靠的通用 CLI 解析接口，因此保持 null；不能用估算或空格分词冒充真实 token 数。第一次只建议用少量任务检查流程。

预算值（1024/2048/4096 context 与 128/256/512 max tokens）是可配置的实验档位，不保证适合所有模型、设备或内存约束。手机实验必须单独校准，且整应用峰值 RAM 不得超过 4096 MiB。不要把 Colab GPU 的内存结果代替 Android 测量。

## 4. 校验与训练

```bash
python scripts/validate_telemetry.py data/telemetry.jsonl --require-paired-budgets --check-task-split-leakage
python scripts/train_budget_controller.py --input data/telemetry.jsonl --output-dir outputs/budget_controller
python scripts/evaluate_budget_controller.py --input data/telemetry.jsonl --model-dir outputs/budget_controller --output outputs/budget_controller/test_report.json
```

控制器只使用 completed 且独立 verifier 给出 pass/fail 的配对任务；目标是“最低一档通过质量验证的预算”。如果所有预算都失败/unknown，该任务不提供安全目标。训练按任务预先记录的 train/validation/test split 分开，至少需要 12 个有效 train task-pairs、3 个 validation pairs 和 3 个有效 test pairs，并且训练标签至少包含两档预算。达不到门槛就停止，不合成假数据。

第一版 MLP 是研究基线，不是生产策略。训练完成后运行 `scripts/evaluate_budget_controller.py`，在独立 test split 上对比控制器预测、固定 full-budget 的质量通过率、平均延迟与可用 PSS。测试对必须足够且可客观评分；上线前仍需在目标设备上复验失败/回退率与资源硬限制。采集器的 PSS 采样、命令行选项和后端能力需要在目标环境验证；schema 通过不代表数据真实或模型加速。



## 5. 任务内在线适应决策器（首版）

当前新增的 `scripts/online_adaptation.py` 只根据一轮推理的结构化观测，决定停止或建议升一档预算。它是独立、可测试的策略模块，不会自行启动模型，也不会更新 GGUF、LoRA 或全局控制器。

准备一个观测文件，例如 `runs/observation.json`：

```json
{
  "budget": "small",
  "quality_label": "fail",
  "status": "completed",
  "whole_app_pss_mib": 2500,
  "memory_measurement_source": "android_whole_app"
}
```

运行：

```bash
python scripts/online_adaptation.py --input runs/observation.json
```

决策规则：

- 独立验证器给出 `pass`：停止本任务。
- 给出 `fail`，且整应用 PSS 有实测值、来源标记为 `android_whole_app` 并低于默认软门槛 3584 MiB：建议从 small 升到 medium，或从 medium 升到 full。
- PSS 缺失、来源不可信（例如只有 llama-cli 子进程 PSS）、质量为 `unknown`、超时、OOM 或执行失败：不盲目增加预算。
- 达到 4096 MiB 硬限制：立即输出内存违规停止决策；安全约束不能被质量奖励覆盖。
- full 预算仍失败：停止并记录失败，不无限循环。
- 所有决策都将 `persistent_update` 设为 false；跨任务的持久学习仍须经过数据审核、离线训练、独立测试和版本回滚门槛。

**重要：** `whole_app_pss_mib` 必须来自目标 Android 上对整个应用进程/进程组的真实测量，并将 `memory_measurement_source` 设为 `android_whole_app`。仅有数值而无可信来源标记时，决策器会拒绝升档。采集器记录的 llama-cli 子进程 PSS 不能冒充整个 Android 应用的 PSS。该脚本当前是策略原型，尚未接入实际推理执行器；单元测试通过也不代表已验证真实加速或手机内存安全。

## 自适应思考策略与 batch 实验

可选的 `--strategy-memory runs/verified-experiences.jsonl` 会让同一个 9B 模型根据历史已验证结果选择 `direct`、`decompose` 或 `verify` 提示策略。策略记忆不修改 GGUF 权重；没有足够经验时退回 `direct`。只有使用已知独立验证器的 full-budget 结果才会写入跨任务经验。

采集器也会向 llama.cpp 请求真实的 `-b` batch 值：small=64、medium=128、full=256。batch 主要影响提示词处理吞吐和临时缓冲区，较大的值不保证更快，也可能提高内存峰值。这些只是实验档位；运行前应通过所用版本的 `llama-cli --help` 确认参数支持，并在目标设备配对测量。遥测记录的是请求值，不会谎称后端实际生效值已被确认。

## GPU offload 与预测

- 默认 `--gpu-layers 0`，不声称使用 GPU。只有确认 `llama-cli` 是 CUDA 构建且日志显示层已 offload 后，才传 `--gpu-layers 99 --accelerator CUDA-T4`；这两个参数只表达请求/环境标签，实际 offload 仍要看后端日志。
- 训练完成后可查看 `outputs/budget_controller/manifest.json`。研究性预测示例：

```bash
python scripts/predict_budget.py \
  --model-dir outputs/budget_controller \
  --task-family closed_fact \
  --expected-output-type text \
  --device-class colab_gpu \
  --accelerator CUDA-T4 \
  --input-tokens 256
```

预测结果不是自动执行授权。生产执行器仍必须检查质量门槛、实际可用资源与安全回退。
