# Teacher seed batch v0

这是 AIMENG 的第一批**训练样本候选**，用于演示怎样把教师知识快速转成可审计的 SFT 数据，而不是声称已经得到 20 条经过独立认证的真理。

## 内容

- `sft_candidates.jsonl`：20 条结构化候选样本，覆盖数据质量、SFT/蒸馏、持续学习、任务记忆、验证和资源约束。
- 每条包含 `messages`（对话式监督目标）、`sample_id`、`task_id`、主题、难度、验证要求、来源状态和数据划分状态。
- 目前所有条目均为 `verification.status=pending`、`split=unassigned`、`training_eligible=false`。这是刻意的：教师生成答案不是独立验证。

## 怎样把它变成真正可训练的数据

1. 让教师按 `prompts/teacher_knowledge_extraction.md` 生成更多同 schema 的候选记录。
2. 运行结构校验、重复检测和隐私/许可审查。
3. 按样本的验证方式执行程序测试、独立来源核查或人工审核；失败样本标记 rejected，无法判断的保持 pending。
4. 先按 task/source/content-cluster 分组，再划分 train/validation/test，避免近重复泄漏。
5. 仅将通过验证且被批准的记录导出为训练输入；保留原始候选与审计报告。

## 关键区别

JSONL 是交换格式，不是模型权重。检索知识库、SFT 样本和已训练参数是三种不同产物。把文本放进仓库并不会自动训练模型；必须有可训练学生模型、tokenizer、训练器和独立评测。

## 快速查看

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("data/teacher_seed/sft_candidates.jsonl")
rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
print("candidate records:", len(rows))
print("pending:", sum(r["verification"]["status"] == "pending" for r in rows))
print("eligible:", sum(r["training_eligible"] for r in rows))
PY
```

预期：20 条候选、20 条 pending、0 条可直接训练。这是质量闸门，不是缺陷。
