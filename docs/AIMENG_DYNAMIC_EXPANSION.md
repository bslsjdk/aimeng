# AIMENG dynamic neuron expansion (cloud-side prototype)

## Goal

Do not treat 1,024 hidden units as a permanent ceiling. Begin with a small recurrent model and widen it only when controlled evaluation indicates persistent underfitting. Keep this separate from Minecraft data generation and from the Android runtime implementation.

## Expansion rule

1. Train the current width to a fixed budget.
2. Evaluate on fixed train/validation splits and held-out map seeds.
3. If validation reaches its target, do not expand.
4. If the train/validation score gap is too large, do not expand yet: investigate data coverage, leakage, or regularization.
5. Expand only after a configurable plateau with both scores still low and a small generalization gap.
6. Grow gradually (for example 256 -> 512 -> 1,024 -> 2,048 -> 4,096), then retrain and reevaluate.
7. Keep the old checkpoint and compare old/new with matched training compute, same data splits, and several seeds. Retain the wider model only if it improves held-out task performance enough to justify latency, compute, and memory costs.

## Function-preserving GRU widening

`scripts/dynamic_expansion.py` includes a cloud-side PyTorch GRU prototype. It copies each existing GRU gate block into the wider model, zeroes new recurrent connections into the old subnetwork, copies the old readout weights, and initializes the new readout columns to zero. The new model should therefore produce the same logits immediately after expansion (up to floating-point tolerance), while the new units can begin learning as the readout weights move away from zero.

After widening, **create a new optimizer**: optimizer momentum/Adam state tensors from the old shape cannot be reused as-is. Reset the recurrent hidden state at an episode boundary, or pass an old-size state which the prototype pads with zeros.

## Do not auto-expand from one metric

A poor validation score alone is not proof of insufficient capacity. A large train/validation gap is a warning for overfitting, and a flat learning curve may be caused by bad labels, optimization, or data coverage. The included `CapacityMonitor` is a conservative trigger prototype, not a proof of capacity exhaustion. Review the decision and benchmark before accepting a wider model.

## Limits and next validation

- This is a cloud-side training prototype and requires PyTorch; it is not usable in the user's Android terminal without that dependency.
- It is not wired into an existing AIMENG trainer yet.
- It does not automatically resize an exported AIMG package or the Android runtime. Exporter/runtime compatibility must be implemented and tested separately.
- Keep deployed runtime RAM below 4 GiB; measure actual end-to-end peak memory and latency after export.
- The Minecraft grid generator's labels cover synthetic waypoint navigation only. Dynamic expansion cannot turn synthetic performance into evidence of full Minecraft competence.
