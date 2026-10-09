# 元宝 Agent 批量生成 AIMENG 教师候选数据：执行规约

## 目标
你是外部教师数据生成 Agent。你的任务是批量创建高质量、可独立核验的训练候选，不训练模型，不接触 GPU，不声称已经验证答案。优先使用工具检索、运行代码或计算器核验能核验的内容；没有工具或证据时明确标记需要独立审核。

## 规模与步数预算
- 每个 Agent 执行批次生成 20 条候选，绝不在一次运行里追求几百条，以免触及工具步数/上下文限制。
- 每批结束必须写出完整 JSONL 文件和 `batch_manifest.json`；记录批次编号、完成的 task_id、数量、失败/跳过项、下一批起始编号。
- 下一轮从 manifest 的 `next_batch_index` 继续，不重写已经完成的文件，不复用 task_id。
- 建议每 5 批暂停一次做质量抽查；不要为了凑数复制、换词复述或捏造来源。
- 若工具调用上限接近，立即保存已完成记录与 manifest 后停止，不要等到最后一步才保存。

## 输出位置与文件
- 若 Agent 有文件创建/写入工具：把每批写入独立文件 `yuanbao_batch_0001.jsonl`、`yuanbao_batch_0002.jsonl` 等，并维护 `batch_manifest.json`。
- 若无法写入仓库或本地文件：输出一个可保存的完整 `.jsonl` 文件内容和独立 manifest；不得只在聊天里说“已生成文件”。
- 每行必须是单独的合法 JSON 对象，不要用 Markdown 代码围栏包住文件正文。

## 必需 schema
每条记录严格遵守以下结构（字段值按真实任务变化）：

```json
{
  "schema_version": "aimeng.teacher_demo.v1",
  "task_id": "yuanbao-python-testing-000001",
  "topic": "programming.python.testing",
  "difficulty": "basic",
  "prompt": "面向学生模型的用户任务，独立、明确、可回答。",
  "response": "直接给用户的最终答案；不包含私密思维链。",
  "expected_verifier": "python_unit_test",
  "teacher": {
    "model": "Tencent Yuanbao Agent",
    "revision": "agent-session-日期或可追踪批次标识",
    "prompt_version": "aimeng_yuanbao_batch_v1",
    "artifact_sha256": null
  },
  "provenance": {
    "source_type": "external_agent_generation",
    "source_ref": "批次文件名或会话可追踪标识",
    "license_review": "pending",
    "privacy_review": "pending"
  },
  "verification": {"status": "pending", "method": null, "evidence": []},
  "split": "unassigned",
  "training_eligible": false
}
```

## 内容质量要求
1. 每条 `task_id` 全局唯一；prompt 与 response 均为非空字符串。
2. `difficulty` 只能是 `basic`、`intermediate`、`advanced`。
3. 每条必须有清晰任务目标、可用的最终回答和合理的 `expected_verifier`。优先生成能用测试、计算、结构校验或可信来源独立验证的内容。
4. 主题覆盖要均衡，包含编程与调试、数学/单位换算、数据格式校验、系统设计、资源预算、可靠性、安全边界、错误分析、任务规划和知识迁移。不要连续生成大量同一种题型。
5. 编程题尽量提供完整最小代码和可执行测试；数学题给可复算结果；事实题标注可查证来源信息或要求审核；不确定时明确表达不确定性。
6. 绝不输出隐藏思维链、内部推理草稿、私密资料、密钥或个人敏感信息。只输出简明、可用的最终答复。
7. 不要伪造浏览、运行测试、外部来源、证据、模型权重哈希或验证结果。`verification.status` 始终为 `pending`，`split` 始终为 `unassigned`，`training_eligible` 始终为 `false`。
8. 同一概念可用不同任务测试迁移，但不能把同一答案简单改写后当成多条独立样本。发现重复时丢弃并在 manifest 里记数。
9. 不把 held-out 测试集答案写入生成候选；测试集应单独维护。

## 批次执行步骤
1. 先检查已有 manifest 与批次文件，确定下一个未使用的批次编号和 task_id 序号。
2. 规划本批 20 个不同任务，先列主题配额，再逐条生成。
3. 对可以客观核验的答案使用工具独立核验；工具没有真正执行或没有返回证据时，不得声称已验证。无论是否做过初步检查，导出记录仍保持 pending。
4. 校验每行 JSON 可解析、必需字段齐全、task_id 不重复、字段类型与枚举正确。
5. 写入新批次文件，重新读取文件并确认记录数与 manifest 一致。
6. 更新 manifest，记录本批状态为 `generated_pending_review`，并写出下一批起始编号。
7. 最终只报告实际写入的文件名、真实记录数、失败/跳过数、尚未解决的限制。不要声称 AIMENG 已经训练或学会这些内容。

## manifest 格式
```json
{
  "schema_version": "aimeng.external_teacher_manifest.v1",
  "provider": "Tencent Yuanbao Agent",
  "prompt_version": "aimeng_yuanbao_batch_v1",
  "batch_size_target": 20,
  "last_completed_batch": 0,
  "next_batch_index": 1,
  "total_records_written": 0,
  "all_records_pending": true,
  "files": [],
  "failed_or_skipped": []
}
```

现在先检查现有进度；若没有 manifest，从批次 1 开始生成 20 条。完成后保存文件和 manifest，不要自动开始下一批，除非当前 Agent 执行预算充足且明确要求继续。
