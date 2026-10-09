# External Agent teacher → AIMENG student workflow

**Preferred low-GPU-cost route:** use Yuanbao Agent (or another authorized external agent with tools) to generate candidate demonstrations in bounded batches. This consumes the agent provider's resources rather than the training GPU. Reserve the free CUDA GPU for student training. The local Ornith GGUF/llama.cpp generator remains an optional fallback if external-agent output is unavailable or insufficient.

Ready-to-use batch contract: [`prompts/yuanbao_teacher_batch_agent.md`](../prompts/yuanbao_teacher_batch_agent.md). It specifies 20 records per batch, JSONL schema, unique IDs, resumable manifest, quality checks, and strict pending/ineligible status. For a 50-step tool budget, do not spend all steps on generation: reserve steps for reading progress, validating JSON/schema, writing files, re-reading them, and updating the manifest.

## Recommended flow

1. Give Yuanbao Agent the batch contract. Ask it to generate 20 records and save both a JSONL file and `batch_manifest.json` using its available file tools. If it cannot write files, have it return the complete JSONL content in manageable chunks for saving.
2. After each batch, confirm the file exists and the manifest's record count matches the actual lines. Continue from `next_batch_index`; never regenerate completed IDs or overwrite previous batches.
3. Import each batch with `scripts/import_teacher_demos.py`. The importer resets every record to pending, unassigned, and ineligible regardless of what the external agent claims.
4. Independently verify candidate answers, remove duplicates, then assign leakage-safe train/validation/test splits. Keep evaluation questions separate from generated training prompts.
5. Only after enough verified examples and frozen eval suites exist, open the free GPU session, train the student, evaluate it, and measure Android app RAM separately.

External Agent generation is not a verification oracle. For code, run tests; for calculations, independently recompute; for facts, check trustworthy sources; for open-ended answers, review quality. A large unreviewed dataset is just a large pile of potential errors.

---
# Free-GPU Ornith teacher → AIMENG student workflow

This workflow uses the official **quantized GGUF teacher** `ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M` through llama.cpp, instead of loading the ~19 GB BF16 Transformers checkpoint and quantizing it at runtime. The official Q4_K_M file is about 5.78 GB on disk; that is not the same as total runtime RAM/VRAM. Official model files and llama.cpp usage are documented at https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF.

## Resource and compatibility notes

- Select a CUDA GPU runtime. Use a **CUDA-enabled llama.cpp build** with `llama-server` available on PATH. A CPU-only build may run slowly and will not satisfy the intended GPU route.
- The generator starts `llama-server` itself with `--n-gpu-layers 99`, waits for `/health`, calls its local OpenAI-compatible chat API, and shuts the server down at exit. GPU offload is requested, but the actual backend/logs must be checked to confirm it worked.
- Keep the first run tiny: `--limit 2`, `--context-size 2048`, and `--max-new-tokens 256`. The 5.78 GB file size is not a guarantee that every free GPU can load the model with the selected context/KV cache.
- Do not keep teacher and student in VRAM simultaneously. Generate demonstrations, let the teacher process exit, save the JSONL to persistent notebook output/storage, and only then start student training.
- This teacher path uses llama.cpp, not the Python Transformers/bitsandbytes stack. The student SFT environment has its own Python dependency requirements; keep teacher inference and student training phases separate if their dependencies conflict.
- The model may produce inaccurate answers. Every generated record remains pending and ineligible for training.

## Phase 1: generate teacher demonstrations

Upload or create a task JSONL file with one task per line. See `data/teacher_seed/gpu_tasks.example.jsonl`.

Make sure a CUDA-enabled `llama-server` executable is installed and works in the notebook. The official model page documents the `llama-server -hf ...:Q4_K_M` invocation. Then run the smoke test:

```bash
python scripts/generate_ornith_teacher_demos.py \
  --tasks data/teacher_seed/gpu_tasks.example.jsonl \
  --output /kaggle/working/teacher_smoke.jsonl \
  --limit 2 \
  --context-size 2048 \
  --max-new-tokens 256
```

The default model is `ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M`. Use `--model` to select another compatible GGUF quantization. The script refuses to overwrite existing output and refuses to silently fall back to the BF16 Transformers model. If the smoke test succeeds, inspect the output and the llama.cpp startup logs for GPU offload. Only then generate a larger batch with a new output filename.

The generator records the model reference and requested generation settings, removes explicit `<think>` / `<analysis>` blocks if present, and always writes:
- `verification.status = pending`
- `split = unassigned`
- `training_eligible = false`

The recorded model-reference hash identifies the model string; it is **not** a hash of the downloaded model weights. The script intentionally does not invent a weight-file SHA-256.

## Phase 2: review and prepare a trainable dataset

Import the raw output into the normalized candidate format:

```bash
python scripts/import_teacher_demos.py \
  --input /kaggle/working/teacher_smoke.jsonl \
  --output data/teacher_seed/imported_candidates.jsonl \
  --teacher-model "ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M"
```

Then independently verify examples, reject incorrect/duplicated answers, group related tasks to prevent leakage, and assign train/validation/test splits. The importer intentionally does not approve its own output. Do not just flip every record to eligible; code should be tested, calculations independently checked, factual claims sourced, and open-ended examples reviewed. Keep held-out evaluation tasks separate from teacher prompt-generation tasks.

## Phase 3: train the student on a free GPU

Use the existing one-command student pipeline after switching to a compatible student-training environment. Its `--model` must be a **trainable Hugging Face causal-LM checkpoint**; it must not point to the Ornith teacher merely because the teacher is available. Begin with a small, compatible student and LoRA, batch size 1, gradient accumulation, and gradient checkpointing when supported.

```bash
python scripts/run_student_pipeline.py \
  --data data/verified/approved_candidates.jsonl \
  --model /path/to/trainable-hf-student \
  --heldout eval/heldout.jsonl \
  --regression eval/regression.jsonl \
  --output runs/student-first-smoke \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 8 \
  --max-length 1024 \
  --gradient-checkpointing
```

The paths above are examples and must exist before execution. A training run is not ready until the verified dataset and frozen evaluation suites exist. The current repository's seed candidates are still pending, so they cannot be passed straight into training.

## What counts as success

A successful teacher generation run proves only that the teacher generated candidate text. A successful SFT run proves only that the compatible student training job ran and saved a checkpoint. Keep both claims separate. Compare held-out transfer and old-capability regression, preserve the previous checkpoint, and measure whole-app peak RAM on the target Android phone before promotion; the hard limit remains below 4096 MiB, with a preferred release gate below 3800 MiB.
