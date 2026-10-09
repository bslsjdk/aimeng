# AIMENG 蒸馏与受控自学习：可执行手册

本文说明当前代码实际支持的第一条端到端路径。它不声称 Ornith 已被自动调用，也不声称模型已训练。

## 当前已实现的流程

1. 教师推理端（手机或本地/云端）生成 JSONL 演示；模型调用方式由实际可用后端决定。
2. `scripts/import_teacher_demos.py` 将通用教师输出导入 AIMENG SFT schema。导入内容一律 `pending`、`unassigned`、`training_eligible=false`。
3. 人工、可执行测试或可信外部来源独立核验。只有填写验证方法、证据，并明确分配 train/validation/test 后，才可能允许训练。
4. `scripts/validate_sft_candidates.py` 检查结构和重复 ID；`scripts/train_student_sft.py --dry-run` 再检查训练资格、split 泄漏和数据快照。
5. 兼容的 Hugging Face causal LM 执行 response-level SFT。脚本只对最后一条 assistant 回复计算 loss；如果 chat template 的 token 边界不一致，会直接停止而不是静默训练错误目标。
6. 训练后保留 run manifest、数据 SHA-256、样本 ID、配置、验证 loss 和 checkpoint。新模型必须通过独立迁移测试与旧能力回归测试，才可被人工晋级。

## 1. 教师输出格式

每行一个对象：

```json
{
  "schema_version": "aimeng.teacher_demo.v1",
  "task_id": "python-exceptions-001",
  "topic": "python.exceptions",
  "difficulty": "basic",
  "prompt": "给出一个最小示例，说明 Python 如何捕获 ValueError。",
  "response": "示例代码及解释……",
  "expected_verifier": "python_unit_test",
  "teacher": {
    "model": "Ornith-1.5-9B",
    "prompt_version": "aimeng-distill-v1",
    "artifact_sha256": null
  },
  "provenance": {
    "source_ref": null,
    "license_review": "pending",
    "privacy_review": "pending"
  }
}
```

哈希只有在实际可访问模型工件时才填写，不能伪造。不要将私密数据、凭据或未经许可的数据放进样本。

## 2. 导入并审计

```bash
python scripts/import_teacher_demos.py \
  --input runs/teacher_raw.jsonl \
  --output data/teacher_seed/imported_candidates.jsonl

python scripts/validate_sft_candidates.py \
  --input data/teacher_seed/imported_candidates.jsonl

python scripts/train_student_sft.py \
  --data data/teacher_seed/imported_candidates.jsonl \
  --model /path/to/trainable-hf-student \
  --output runs/sft-smoke \
  --dry-run
```

新导入的数据预期会被拒绝实际训练，因为它们还没有核验、拆分和授权。这个拒绝是安全门，不是故障。

## 3. 核验与拆分合同

对每条获准训练的记录，至少需要：
- `verification.status = "verified"`
- 非空 `verification.method`
- 非空字符串数组 `verification.evidence`
- `training_eligible = true`
- `split` 明确为 train、validation 或 test
- 非空 `task_id`，且同一任务 ID 不跨 split

程序任务优先运行测试；确定性数学/转换优先用独立检查器；事实记录保存可信来源与适用版本；没有可靠裁判的开放回答必须人工审核或继续保持 unknown/pending。教师自信度、教师自评和同一教师重复回答都不是独立证据。

## 4. 训练与节省计算

- 先用很小的已核验数据做 smoke test，确认损失确实只落在 assistant 目标上。
- 每次训练固定数据哈希、模型引用、随机种子、超参数和样本 ID。
- 只针对独立评测发现的失败类别生成补充样本；合并时保留旧能力回放样本，避免只学新题。
- 不要仅凭 eval loss 判断进步。必须同时检查 held-out 成功率、迁移任务、回归任务和资源指标。
- test split 永远不进入 optimizer，也不用于筛选教师答案或调参。
- 本脚本适用于标准 Hugging Face causal LM。若 AIMENG 自有学生不是兼容的 causal LM，必须实现其专用训练器，不能把不兼容架构伪装成已经训练。

## 5. 当前限制与明确未完成项

- 尚未连接到 Ornith 的自动推理 API；教师生成可以由实际可用的本地/远程后端产出 JSONL。
- 当前训练入口面向标准 Hugging Face causal LM，不是自有张量/模块架构的原生训练器。
- 自动事实核验、语义近重复聚类、自动分组拆分、评测晋级/回滚编排还需后续实现。
- 训练框架不证明 Android 部署满足 4096 MiB；那必须在目标手机上测整应用峰值。
- 没有可用且兼容的可训练学生 checkpoint 时，不要把 Ornith GGUF/MLX 推理工件直接交给此训练器。

## 6. 完成标准

框架的基础验收应包括：CI 通过、pending 样本绝不会进入训练、split 泄漏会被拒绝、assistant-only masking 有单测、dry-run 可重复、真实小规模 smoke test 能保存 checkpoint 与 manifest、独立 held-out 与回归评测可复现。只有最后几项也完成后，才算训练链路端到端跑通。
