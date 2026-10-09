# AIMENG Teacher SFT Batch Prompt v1

## Purpose
Generate auditable supervised fine-tuning (SFT) candidates from a narrow topic. Output quality and verifiability matter more than raw sample count.

## Rules
- Generate independent, non-duplicative examples. Each should teach one principal concept or behavior.
- Give a self-contained user request and a direct, correct assistant target.
- Do not invent citations, URLs, test results, model capabilities, or claimed tool execution.
- For code, include a minimal executable test idea; for arithmetic, make the result reproducible; for factual claims, specify an independent source-check plan.
- Do not include private conversations, credentials, personal sensitive information, or unlicensed source text.
- Every record starts with verification status pending, split unassigned, and training_eligible false. The teacher must never certify its own answer.
- Output JSONL only, one JSON object per line, without Markdown fences or surrounding commentary.

## Required schema
{
  "schema_version": "aimeng.sft_candidate.v1",
  "sample_id": "unique stable identifier",
  "task_id": "group identifier for leakage-safe splitting",
  "messages": [
    {"role": "user", "content": "question or task"},
    {"role": "assistant", "content": "target response"}
  ],
  "topic": "lowercase.hierarchical.topic",
  "difficulty": "basic",
  "expected_verifier": "specific independent check",
  "verification": {
    "status": "pending",
    "method": "independent_review_required",
    "evidence": []
  },
  "split": "unassigned",
  "training_eligible": false
}

Allowed difficulty values: basic, intermediate, advanced.
Allowed roles: system, user, assistant. A candidate should contain at least one user and one assistant message.
Allowed verification states for later review: pending, verified, rejected. Only a trusted review process may set verified and training_eligible true.
Allowed split values: unassigned, train, validation, test. Split by task/source/content cluster before export, not by randomly splitting near-duplicate rows.

## Self-check before output
1. Every line parses as JSON.
2. Every sample_id is unique in this batch.
3. No obvious duplicate paraphrases.
4. Each answer is self-contained and calibrated about uncertainty.
5. The verification status remains pending.
