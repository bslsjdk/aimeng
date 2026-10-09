"""Minimal Hugging Face causal-LM SFT harness for reviewed AIMENG candidates.

This is a real training entry point, not a claim that training has already run.
Requires a compatible local model/tokenizer and transformers, datasets, torch.
Only verified, explicitly split records are loaded; test records are never optimized on.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any

def load_records(path: Path, split: str) -> list[dict[str, Any]]:
    rows=[]; seen=set()
    with path.open(encoding="utf-8") as f:
        for n,line in enumerate(f,1):
            if not line.strip(): continue
            try: row=json.loads(line)
            except json.JSONDecodeError as e: raise ValueError(f"{path}:{n}: invalid JSON: {e.msg}") from e
            sid=row.get("sample_id")
            if not sid or sid in seen: raise ValueError(f"{path}:{n}: missing/duplicate sample_id {sid!r}")
            seen.add(sid)
            v=row.get("verification",{})
            if row.get("training_eligible") is not True or v.get("status") != "verified":
                continue
            if not v.get("method") or not isinstance(v.get("evidence"), list) or not v["evidence"]:
                raise ValueError(f"{path}:{n}: verified eligible row requires verification method and evidence")
            if row.get("schema_version") != "aimeng.sft_candidate.v1":
                raise ValueError(f"{path}:{n}: unsupported schema_version")
            if row.get("split") not in {"train", "validation", "test"}:
                raise ValueError(f"{path}:{n}: eligible row needs an explicit split")
            if row.get("split") != split: continue
            messages=row.get("messages")
            if not isinstance(messages,list) or not any(m.get("role")=="user" for m in messages) or not any(m.get("role")=="assistant" for m in messages):
                raise ValueError(f"{path}:{n}: invalid messages")
            rows.append(row)
    return rows

def render(row: dict[str, Any], tokenizer) -> str:
    msgs=[{"role":m["role"],"content":m["content"]} for m in row["messages"]]
    if not hasattr(tokenizer,"apply_chat_template") or not tokenizer.chat_template:
        raise ValueError("Tokenizer has no chat_template; provide a compatible model/tokenizer instead of guessing a template")
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",required=True); p.add_argument("--model",required=True)
    p.add_argument("--output",required=True); p.add_argument("--epochs",type=float,default=1.0)
    p.add_argument("--lr",type=float,default=2e-5); p.add_argument("--batch-size",type=int,default=1)
    p.add_argument("--grad-accum",type=int,default=8); p.add_argument("--max-length",type=int,default=1024)
    p.add_argument("--dry-run",action="store_true",help="Validate data/config without loading model weights")
    a=p.parse_args(); path=Path(a.data)
    train=load_records(path,"train"); val=load_records(path,"validation")
    if not train: raise SystemExit("Refusing to train: zero verified, training-eligible train records")
    if not val: raise SystemExit("Refusing to train: zero verified, training-eligible validation records")
    if a.epochs<=0 or a.lr<=0 or a.batch_size<1 or a.grad_accum<1 or a.max_length<8: raise SystemExit("Invalid training hyperparameters")
    print(json.dumps({"train_records":len(train),"validation_records":len(val),"model":a.model,"dry_run":a.dry_run,"note":"Validation records are reserved for evaluation, not optimizer updates."},ensure_ascii=False))
    if a.dry_run: return 0
    try:
        from datasets import Dataset
        from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
    except ImportError as e: raise SystemExit("Missing dependencies. Install compatible torch, transformers, and datasets.") from e
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=False)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(a.model,trust_remote_code=False)
    def convert(rows):
        texts=[render(r,tok) for r in rows]
        ds=Dataset.from_dict({"text":texts})
        def enc(batch): return tok(batch["text"],truncation=True,max_length=a.max_length)
        return ds.map(enc,batched=True,remove_columns=["text"])
    tr=convert(train); ev=convert(val)
    args=TrainingArguments(output_dir=a.output,num_train_epochs=a.epochs,learning_rate=a.lr,per_device_train_batch_size=a.batch_size,per_device_eval_batch_size=1,gradient_accumulation_steps=a.grad_accum,evaluation_strategy="epoch",save_strategy="epoch",logging_steps=10,report_to="none",load_best_model_at_end=True,metric_for_best_model="eval_loss",greater_is_better=False,save_total_limit=2)
    trainer=Trainer(model=model,args=args,train_dataset=tr,eval_dataset=ev,data_collator=DataCollatorForLanguageModeling(tokenizer=tok,mlm=False))
    trainer.train(); metrics=trainer.evaluate(); trainer.save_model(a.output); tok.save_pretrained(a.output)
    Path(a.output,"aimeng_run_summary.json").write_text(json.dumps({"train_records":len(train),"validation_records":len(val),"eval_metrics":metrics,"data_path":str(path)},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(metrics,ensure_ascii=False)); return 0
if __name__=="__main__": raise SystemExit(main())
