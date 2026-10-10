# Google Colab D0 training and recovery guide

## What lives where

- GitHub stores the training code, this notebook, the baseline configuration, dataset-source notes, and this operating guide.
- Google Drive stores the corpus and run artifacts persistently. Colab's `/content` filesystem is temporary, so never use it as the only copy of a checkpoint.
- Each run directory contains:
  - `training_state.pt`: resumable model + optimizer state + completed step count.
  - `diffusion_checkpoint.pt`: final inference checkpoint for generation tests.
  - `report.json`: metrics and configuration summary.
  - `training.log`: persistent stdout/stderr log captured by the notebook.

The notebook uses `MyDrive/aimeng_corpus.txt` and `MyDrive/aimeng_runs/`. Do not delete those Drive paths after disconnecting Colab.

## Start / resume

1. Open [the D0 Colab notebook](https://colab.research.google.com/github/bslsjdk/aimeng/blob/feat/google-colab-d-training/notebooks/AIMENG_D0_Google_Colab.ipynb).
2. Select a GPU runtime and run cells from top to bottom. The GPU-check cell intentionally stops if no GPU is attached.
3. The 20-step preflight writes to `aimeng_runs/D0_preflight`; the 500-step baseline writes to `aimeng_runs/D0_8192n_w32_k128`.
4. The trainer atomically writes a resumable checkpoint every 25 completed steps (configurable). If Colab disconnects, reconnect, mount Drive, rerun the notebook cells, and rerun the training cell. If `training_state.pt` exists, the notebook automatically passes `--resume`.
5. Resume is rejected if the corpus SHA-256 or core model/training settings differ. For a changed dataset or configuration, use a new output directory. To continue beyond 500 steps, raise `TRAIN_STEPS` while keeping the existing corpus and core settings.

The most recent safe state is saved even when the memory guard stops a run. Checkpoint files are written atomically so a partially written file does not replace the previous good state.

## Dataset and license

The current notebook fallback is up to 50,000 instruction/answer records from `BelleGroup/train_0.5M_CN` via `scripts/prepare_speaking_corpus.py`. It is a Chinese instruction-format starter only, not a sufficient full curriculum or evidence of general language competence. Review upstream dataset terms/license before use. See [LANGUAGE_CURRICULUM.md](LANGUAGE_CURRICULUM.md) for required curriculum layers and evaluation gates. Review the upstream dataset card and license before redistributing the downloaded corpus. The notebook stores the corpus in Drive and does not commit it to Git.

For a Chinese or multilingual experiment, supply a UTF-8 corpus that you have permission to use. Record its exact source, version, license, preprocessing, and SHA-256 with the run. Keep train/validation/test data separated by document or source where possible; the current trainer's simple contiguous 90/10 split is only a baseline and may leak repetitive/source-specific patterns.

## Compute/time efficiency

- First run the 20-step preflight; don't start the longer run if GPU, corpus, or checkpoint output is broken.
- Keep batch size 1 and the initial D0 dimensions until measured throughput and GPU memory are known.
- Checkpoint cadence defaults to 25 steps. Increase it only if checkpoint I/O is a measurable bottleneck; lower it if interruptions are frequent.
- A resumed run skips already completed batches deterministically and restores optimizer state rather than restarting optimization.
- Keep the complete log. Compare initial and held-out validation loss, but do not treat a lower loss on one small dataset as proof of language mastery.
- Colab availability and free compute quotas vary; there is no guaranteed daily five-hour allocation.

## What to archive back into GitHub

Do not put multi-megabyte or larger model weights and raw corpora into normal Git history. Keep those in Drive, and archive the small reproducibility bundle instead: `report.json`, `training.log`, the exact config, corpus SHA-256, and a short experiment note. If a checkpoint is small enough and licensing permits, choose a GitHub Release or Git LFS deliberately rather than committing it blindly. The source branch itself is already pushed to GitHub; training results only exist after a user-run Colab session produces them.

## Important limits

This is a character-level next-character research prototype. Its current context representation and routing are experimental, and D0 is a pipeline/quality baseline, not a trained general-purpose language model. The 4 GiB Android runtime constraint applies to eventual mobile inference; Colab GPU memory does not prove the Android app meets that constraint.
