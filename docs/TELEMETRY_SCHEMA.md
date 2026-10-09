# 推理遥测与反事实预算数据规范 v1

本规范用于 A/B 基线、B 阶段遥测、C 阶段动态预算原型和 D 阶段反事实采集。数据采用 JSON Lines：每行一条完整运行记录。不要写入用户隐私、完整敏感提示或未经脱敏的工具凭据。

## 1. 设计原则

- 一条记录对应一个任务在一个明确预算配置下的一次运行。
- 同一任务的不同预算运行必须共享 `task_id`，但拥有不同的 `run_id` 和 `budget_id`。
- 保存计划值和实际观测值，不能把“请求的配置”冒充“后端实际采用的配置”。
- 不支持的字段写 `null` 并在 `unsupported_features` 中说明；不要猜测后端行为。
- 质量标签分为 `pass`、`fail`、`unknown`。unknown 不可直接作为正负奖励标签。
- 通道/层跳过若只是模拟，`execution_mode` 必须为 `simulation`，且 `verified_compute_skipped=false`。
- 所有耗时单位为毫秒，内存单位为 MiB，吞吐单位为 token/s；采样时间使用 UTC ISO-8601。

## 2. JSONL 单条记录示例

```json
{
  "schema_version": "aimeng.telemetry.v1",
  "run_id": "run-000042-medium",
  "task_id": "task-000042",
  "pair_group_id": "pair-000042",
  "timestamp_utc": "2026-10-09T12:00:00Z",
  "model": {
    "name": "Ornith-1.5-9B-Q4_K_M.gguf",
    "sha256": null,
    "backend": "llama.cpp",
    "backend_version": null
  },
  "environment": {
    "device_class": "colab_gpu",
    "accelerator": "T4",
    "gpu_vram_total_mib": 15360,
    "gpu_peak_allocated_mib": null,
    "gpu_peak_reserved_mib": null,
    "android_ram_limit_mib": 4096,
    "device_profile_id": "colab-t4-profile-v1"
  },
  "task": {
    "task_family": "json_validation",
    "dataset_split": "train",
    "prompt_template_id": "template-v1",
    "input_tokens": 512,
    "expected_output_type": "json",
    "privacy_safe_features": {}
  },
  "budget": {
    "budget_id": "medium-v1",
    "n_ctx_requested": 2048,
    "n_ctx_actual": 2048,
    "n_batch_requested": 128,
    "n_batch_actual": 128,
    "max_tokens": 256,
    "threads": 4,
    "concurrency": 1,
    "kv_cache_type_k": null,
    "kv_cache_type_v": null,
    "decode_profile": "baseline",
    "execution_mode": "normal",
    "unsupported_features": ["kv_cache_type_not_reported_by_backend"]
  },
  "result": {
    "status": "completed",
    "quality_label": "pass",
    "quality_score": 1.0,
    "quality_verifier": "json_schema_v1",
    "failure_reason": null,
    "output_tokens": 128
  },
  "performance": {
    "ttft_ms": 420,
    "total_latency_ms": 3200,
    "tokens_per_second": 40.0,
    "latency_p50_ms": null,
    "latency_p95_ms": null
  },
  "memory": {
    "process_peak_rss_mib": null,
    "process_peak_pss_mib": null,
    "backend_buffer_peak_mib": null,
    "kv_cache_estimated_mib": null,
    "gpu_peak_allocated_mib": null,
    "gpu_peak_reserved_mib": null,
    "measurement_method": "backend_or_os_telemetry"
  },
  "routing": {
    "controller_version": "rules-v1",
    "decision_reason": "fixed_medium_budget",
    "requested_channel_groups": null,
    "verified_compute_skipped": false,
    "fallback_stage": 0,
    "fallback_reason": null
  },
  "provenance": {
    "config_sha256": null,
    "dataset_snapshot_sha256": null,
    "random_seed": 1234
  }
}
```

示例中的数值仅展示字段格式，不是实际测试结果。真实实验必须填入真实观测值；不可用的字段保持 null。

## 3. 必填与推荐字段

### 必填身份字段
`schema_version`、`run_id`、`task_id`、`timestamp_utc`、模型身份、后端身份、任务族、数据拆分、预算配置、运行状态、质量标签、总耗时、内存测量状态、控制器版本和配置版本。

### 质量验证
按任务族选择验证器：
- 代码：测试结果、编译结果或静态检查；
- 数学/封闭事实：精确答案或规范化答案比较；
- JSON/工具调用：schema/参数约束校验；
- 开放式文本：独立人工标注或成对比较；
- 无法可靠判定：`unknown`。

保留 `quality_verifier` 和失败原因，避免把不同验证器产生的分数直接混为同一种标签。对质量率应报告样本量和置信区间。

### 资源测量
- Android：优先记录峰值 PSS，同时记录 RSS、后端缓冲区、KV Cache 配置与估算值；若能取得真实 KV Cache 字节数，应优先于估算值。
- CUDA/Colab：记录 `max_memory_allocated`、`max_memory_reserved`、设备总显存及实际 batch/序列长度。
- 不同平台不可比的内存字段不得直接混合排序。

## 4. 反事实预算采集

对预先固定、按任务族分层抽样的任务子集，分别运行小/中/完整预算。尽量固定模型文件、提示模板、采样参数、后端版本和硬件；随机性不可完全固定时，记录种子并对多个种子重复。

每组必须能通过 `task_id` 与 `pair_group_id` 配对。记录每档预算的质量、延迟、吞吐、峰值内存、失败原因和实际采用参数。训练/验证/测试必须按任务或来源分组拆分，不能把同一任务的不同预算版本分散到不同拆分造成泄漏。

初始目标是学习：在质量约束通过的配置中，预测最低成本档位。质量失败的快速配置不得被当作优胜样本。

## 5. 预算动作与分级回退

预算动作仅限当前后端确实支持并能记录实际值的选项，例如上下文、批大小、最大生成 token、线程、并发、KV Cache 类型和停止条件。参数可用性需通过后端能力探测，不得假设所有后端都支持同样的 KV Cache 或动态更改方式。

推荐策略：
1. 质量失败且资源允许：增加预算重试，并记录前后配置；
2. 内存压力升高：降低并发、上下文和临时缓冲区/批大小，必要时缩短生成上限；
3. 仍不稳定时：切换已验证的保守解码配置；
4. 只有预计峰值仍低于硬限制且资源允许时，才尝试完整执行路径；
5. 无安全配置可用时，中止并返回明确失败原因，不得冒险 OOM。

回退必须记录阶段、触发条件、旧预算、新预算、质量结果与资源测量。

## 6. 数据进入训练前的检查

- 拒绝重复 `run_id`、缺失任务身份或不合法枚举；
- 将缺失测量保留为 null，不能填零；
- 排除崩溃、超时与 OOM 样本是不正确的做法；应保留并标注失败，以避免选择偏差；
- 将 `unknown` 质量样本排除在监督质量标签之外，但可用于分析覆盖率与失败分布；
- 按任务族报告各预算的质量、P50/P95 延迟、峰值内存和失败/回退率；
- 新控制器必须在独立测试集和目标设备校准集上比较旧版规则基线。
