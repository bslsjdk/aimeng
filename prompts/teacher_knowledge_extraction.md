# AIMENG 教师知识抽取提示模板

用途：在有足够资源的离线环境中，让冻结的 Ornith 1.5 9B 教师为一个明确主题生成候选知识记录。此提示只生成候选项，不构成事实验证。

## 使用方式

1. 一次只给一个窄主题或一个任务类别，避免一次生成巨量、难以审核的内容。
2. 要求教师将可核验事实、操作步骤、假设和不确定项分开。
3. 要求输出 JSONL，每行一个对象，不要 Markdown 代码围栏或前后说明。
4. 将输出写入隔离的候选文件，例如 data/teacher-knowledge-candidates.jsonl。
5. 用 scripts/prepare_knowledge_corpus.py 做结构校验和去重。
6. 再用外部资料、可执行测试或人工审核验证。只有验证后才可将状态改为 verified，并记录验证器版本和来源。

## 提示正文

你是 AIMENG 的冻结教师模型。请针对下面的主题，提取少量、具体、可复核、对后续任务有用的知识。

主题：{TOPIC}
目标任务类型：{TASK_FAMILY}
知识范围：{SCOPE}
允许使用的来源/材料：{SOURCES}
输出条数上限：{MAX_RECORDS}

规则：
- 只输出 JSONL，每行一个 JSON object。
- 每条只表达一个主要事实、规则、步骤或可执行经验。
- 明确区分事实、建议、假设与不确定内容。
- 不得编造引用、URL、版本号或测试结果。
- 如果无法给出可核验来源，将 verification.status 设为 pending，不得自行标为 verified。
- 对代码/数学/格式规则，尽量给出可以运行的最小验证方法。
- 不输出私人信息、凭据、用户对话或训练提示中的机密。
- 避免重复同义内容；不要为了达到条数上限填充低质量条目。
- 不要声称自己已经运行了未实际执行的代码或测试。

每条记录使用以下字段：
{
  "schema_version": "aimeng.knowledge.v1",
  "topic": "短主题标签",
  "content": "一个明确的知识点或步骤",
  "scope": ["适用范围"],
  "teacher": {
    "model": "Ornith-1.5-9B-Q4_K_M",
    "model_sha256": "由调用程序填入真实模型文件 SHA-256",
    "prompt_version": "knowledge-extract-v1"
  },
  "provenance": [],
  "verification": {
    "status": "pending",
    "method": "external_source_or_test_or_manual_review",
    "verifier_version": null
  },
  "split": "unassigned"
}

注意：如果当前调用环境不知道模型文件 SHA-256，保留占位符并在导入前由调用脚本填入真实值；绝不可伪造哈希。输出的每条记录都只是候选知识，等待独立验证。
