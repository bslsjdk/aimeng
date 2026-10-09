# Free-GPU Ornith teacher → AIMENG student workflow

This is the first practical route for a free CUDA notebook (for example, a Kaggle GPU session). It uses the official **Transformers checkpoint** `ornith-ai/Ornith-1.5-9B`, not the Apple-only MLX artifact. The official model card describes the BF16 checkpoint as roughly 19 GB, so this generator loads it with bitsandbytes NF4 4-bit quantization. This is an inference configuration, not a claim that 4-bit teacher outputs are identical to BF16 outputs.

## Important resource and compatibility notes

- Select a CUDA GPU runtime before installing packages. The script intentionally refuses CPU fallback.
- Keep batch size at 1 for the first run. Start with `--limit 2` and inspect outputs before generating a large corpus.
- Do not try to keep the teacher and student in VRAM simultaneously. Generate demonstrations, let the teacher process exit, and only then start student training.
- Teacher inference currently needs Transformers 5.8.1+ per the official model instructions, while the existing student SFT requirements pin Transformers below 5. Treat these as separate dependency phases/environments. Do not blindly upgrade/downgrade packages mid-run or overwrite the notebook's CUDA-enabled PyTorch installation. Save the generated JSONL to persistent notebook output/storage before changing environments or restarting.
- A free GPU's actual VRAM varies. If the 4-bit model still fails to load, reduce context/input limits, verify the runtime has a GPU, and inspect the exception. Do not silently switch to CPU.
- The model may produce inaccurate answers. Every generated record stays pending and ineligible for training.

## Phase 1: generate teacher demonstrations

Upload or create a task JSONL file with one task per line. See `data/teacher_seed/gpu_tasks.example.jsonl`.

In a CUDA notebook, install teacher-only dependencies while preserving the notebook's existing CUDA PyTorch build:

```bash
python -m pip install -r requirements-teacher-gpu.txt
```

Smoke test first:

```bash
python scripts/generate_ornith_teacher_demos.py \
  --tasks data/teacher_seed/gpu_tasks.example.jsonl \
  --output /kaggle/working/teacher_smoke.jsonl \
  --limit 2 \
  --max-input-tokens 2048 \
  --max-new-tokens 256 \
  --trust-remote-code
```

If that succeeds and the answers are sensible, run the full task file with a new output name and omit `--limit`. Never overwrite a previous run. Keep the output JSONL and notebook logs as artifacts.

The generator records model/revision and generation settings, removes explicit `<think>` / `<analysis>` blocks if present, and always writes:
- `verification.status = pending`
- `split = unassigned`
- `training_eligible = false`

The recorded model-reference hash identifies the model name/revision string; it is **not** a hash of the model weights. The script intentionally does not invent a weight-file SHA-256.

## Phase 2: review and prepare a trainable dataset

Import the raw output into the normalized candidate format:

```bash
python scripts/import_teacher_demos.py \
  --input /kaggle/working/teacher_smoke.jsonl \
  --output data/teacher_seed/imported_candidates.jsonl \
  --teacher-model "ornith-ai/Ornith-1.5-9B"
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
