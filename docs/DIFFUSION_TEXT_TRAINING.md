# AIMENG sparse-diffusion text training (first prototype)

This is a terminal-side research prototype. Android is not part of the training loop. The first configuration uses 128 graph neurons, 32-wide states, at most 8 active nodes per diffusion round, fanout 8 and at most 6 rounds. These are deliberately small starting values, not claims about the final model size.

## Run locally or in Kaggle/Colab

    python -m pip install torch
    python scripts/train_diffusion_text.py --text data/corpus.txt --steps 10000 --output runs/experiment-001
    python scripts/train_diffusion_text.py --checkpoint runs/experiment-001/diffusion_checkpoint.pt --prompt "你好" --generate-tokens 120

data/corpus.txt must be UTF-8 text. Keep a separate held-out corpus for honest final evaluation in later iterations; this prototype reserves the last 10% of the supplied text for validation. Do not use the built-in smoke corpus as real training evidence. The process checks Linux PSS (falling back to RSS if PSS cannot be read) every 25 steps and safely stops/checkpoints at 2560 MiB by default.

Without --text, the script uses a built-in synthetic demonstration corpus solely to verify that training, checkpoint writing, and loss measurement execute. The checkpoint and report.json explicitly label that source.

## First model size

- 128 neurons
- state width 32
- top-k 8 active nodes per round
- fanout 8
- maximum 6 diffusion rounds
- character-level output vocabulary

A character-level model is an engineering baseline, not a claim that characters are the best tokenization. Its purpose is to test graph-state propagation and text loss before increasing capacity. After the first run, compare 128/256/512 neurons under the same corpus, split, seed and compute budget.

## Honest interpretation

The hidden state is updated through repeated graph message passing. Only selected active source/destination nodes run the learned update function, while the full bounded state/energy arrays remain allocated. Thus this prototype sparsifies neural updates but still has indexing, aggregation, routing and readout costs; measure wall time and peak memory rather than assuming sparse always means faster.

Text prediction is character-by-character for this first baseline. Internal diffusion runs multiple rounds before a prediction; this does not magically remove sequence dependence from generating arbitrary text. The learned halt distribution is trained with a small expected-step penalty, while runtime early stopping additionally requires a high halt score and a small state delta. Evaluate both output quality and steps used before relying on early stopping.

The built-in CI run is only a smoke test. A lower validation loss on a tiny repeated corpus does not demonstrate general language competence or useful reasoning. Terminal generation is provided only to validate checkpoint loading and text output; use a real, quality-controlled corpus before judging language quality.
