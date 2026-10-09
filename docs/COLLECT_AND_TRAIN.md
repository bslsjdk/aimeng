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

每个任务会依次跑 small/medium/full 三档。采集器默认覆盖指定输出文件；显式传 `--append` 才会追加，避免普通重跑时意外混入重复任务。采集器记录端到端耗时，并尝试从 Linux `/proc/<pid>/status` 与 `/proc/<pid>/smaps_rollup` 采样子进程 RSS/PSS。后端实际生效的上下文、TTFT、token/s、GPU 显存目前没有可靠的通用 CLI 解析接口，因此保持 null；不能用估算或空格分词冒充真实 token 数。第一次只建议用少量任务检查流程。

预算值（1024/2048/4096 context 与 128/256/512 max tokens）是可配置的实验档位，不保证适合所有模型、设备或内存约束。手机实验必须单独校准，且整应用峰值 RAM 不得超过 4096 MiB。不要把 Colab GPU 的内存结果代替 Android 测量。

## 4. 校验与训练

```bash
python scripts/validate_telemetry.py data/telemetry.jsonl --require-paired-budgets --check-task-split-leakage
python scripts/train_budget_controller.py --input data/telemetry.jsonl --output-dir outputs/budget_controller
python scripts/evaluate_budget_controller.py --input data/telemetry.jsonl --model-dir outputs/budget_controller --output outputs/budget_controller/test_report.json
```

控制器只使用 completed 且独立 verifier 给出 pass/fail 的配对任务；目标是“最低一档通过质量验证的预算”。如果所有预算都失败/unknown，该任务不提供安全目标。训练按任务预先记录的 train/validation/test split 分开，至少需要 12 个有效 train task-pairs、3 个 validation pairs 和 3 个有效 test pairs，并且训练标签至少包含两档预算。达不到门槛就停止，不合成假数据。

第一版 MLP 是研究基线，不是生产策略。训练完成后运行 `scripts/evaluate_budget_controller.py`，在独立 test split 上对比控制器预测、固定 full-budget 的质量通过率、平均延迟与可用 PSS。测试对必须足够且可客观评分；上线前仍需在目标设备上复验失败/回退率与资源硬限制。采集器的 PSS 采样、命令行选项和后端能力需要在目标环境验证；schema 通过不代表数据真实或模型加速。


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
