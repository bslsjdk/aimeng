# Recursive Neural-Unit Training and Growth Protocol

Status: research/training specification only. This document does not claim that AIMENG currently trains or grows neural units autonomously.

## 1. Main proposal

Investigate a staged, recursive growth loop:

1. Train a stable base model that can solve a broad set of tasks and invoke declared computation interfaces.
2. Train an initial library of small, compatible computational units.
3. Detect recurring, evidenced capability gaps in the unified core model.
4. Propose the smallest candidate unit or modification likely to address one gap.
5. Train the candidate in an isolated workspace.
6. Evaluate it against a frozen baseline, held-out tasks, and regression suites.
7. Admit it to the external library only when it passes predeclared gates.
8. Measure its usefulness during integration trials; integrate into a new version only if evidence supports the cost.
9. Use subsequent task evidence to find the next gap and repeat.

This is a falsifiable research hypothesis. Repeatedly adding units does not guarantee increasing intelligence; growth can cause interference, redundancy, overfitting, routing mistakes, and higher latency or memory usage.

## 2. What is a “neural unit” in the first implementation?

Do not begin by attempting to add arbitrary individual neurons to an existing dense checkpoint. A single neuron has meaning only through its incoming and outgoing connections, normalization, surrounding nonlinearities, and training context. Most inference backends do not support independent loading and unloading of arbitrary neurons.

Start with units whose interfaces can be declared and tested:
- low-rank adapters at known insertion points;
- channel/neuron blocks with explicit compatible tensor shapes;
- small subgraphs using an allowlisted operator set;
- only later, structural graph patches or neuron/edge-level topology changes.

All unit types should share a canonical manifest, identity/version, dependency contract, provenance, artifact hash, resource estimate, evaluation scope, and lifecycle state. The unit contract is shared even if physical implementation granularity differs.

## 3. Training the seed population

The first population should not be a huge random swarm. Start with a small, deliberately diverse set of units covering distinct, measurable behaviors or computational transformations.

For each seed unit:
1. define its interface and target capability;
2. construct training examples that exercise the target behavior;
3. train it with the base model frozen initially, unless the chosen architecture requires joint training;
4. train the core model's controller to invoke it only when appropriate;
5. test standalone interface correctness and end-to-end task utility;
6. measure incremental benefit against the same base model without the unit.

Include a no-unit baseline and, where useful, a random-selection control. Do not label units “reasoning neurons” or “memory neurons” solely from their names; their function must be supported by behavior and tests.

## 4. Candidate generation: do not equate errors with a need for more units

A failed task must first be classified. Potential causes include missing context, poor reasoning policy, a tool error, ambiguity, verifier weakness, a resource limit, or a genuine representational/capability gap.

Candidate generation is justified when multiple independent traces indicate a recurring gap that existing units and alternative strategies fail to address. The controller proposes:
- the target capability and evidence;
- the smallest plausible structural change;
- expected benefit and measurable success criteria;
- compute, memory, and latency budget;
- a training plan and an evaluation plan fixed before training.

The generator may be the unified model proposing a structured candidate, but candidate artifacts must be validated by separate deterministic checks and independent evaluations. The model's own confidence is not sufficient evidence.

## 5. How a new unit can be trained

Use the smallest useful candidate and isolate its parameters from the known-good base model.

A preferred first experiment is adapter/subgraph training:
1. freeze and hash the base checkpoint;
2. prepare approved training examples, with development and held-out data kept separate;
3. initialize the candidate according to its architecture;
4. train candidate parameters on the target task distribution;
5. optionally distill selected intermediate behavior from the base model or a teacher, if this is permitted and measurable;
6. train or calibrate the invocation controller on positive and negative examples, including cases where the unit must not be invoked;
7. export an immutable candidate artifact and manifest;
8. run structural, numerical, compatibility, and task evaluations.

Do not assume that duplicating a trained neuron creates a new ability. If cloning is used as an initialization, break symmetry through a controlled transformation and train it on evidence that requires a distinct contribution. Compare against training a fresh candidate and against not adding a unit.

## 6. Admission gates

A candidate is admitted to the external validated library only if it:
- passes artifact hash, schema, dependency, tensor-shape, dtype, quantization, and backend compatibility checks;
- demonstrates improvement over a frozen baseline on predeclared development tasks;
- passes held-out generalization checks and does not train on the final test set;
- stays within measured resource budgets;
- has acceptable latency and invocation error rates;
- does not cause unacceptable regression on existing capabilities;
- has a reproducible artifact, training configuration, and evaluation report.

The unit may be marked narrow-scope if it only helps a restricted domain. Failed, inconclusive, or unavailable evaluation is not a pass. Promotion is reversible and must preserve the prior known-good model.

## 7. Recursive growth policy

Each growth cycle should have a fixed budget for candidate count, training steps, artifact size, and evaluation cost. Generate a small candidate batch, compare candidates under the same conditions, and retain only candidates that pass.

Prefer a scorecard over one opaque score:
- held-out quality gain;
- first-pass success and verified repair gain;
- transfer to related unseen tasks;
- regression rate;
- invocation precision/recall or unnecessary activation rate;
- incremental parameter bytes and actual peak resident memory;
- training cost, cold/warm latency, and inference throughput.

A candidate with negligible quality gain and high resource cost should be rejected. Two redundant candidates should not both be retained unless their combined benefit is measured. Periodically test whether units can be merged, compressed, distilled into a smaller compatible unit, or removed without loss.

## 8. Preventing catastrophic forgetting and growth collapse

Keep the base checkpoint immutable during candidate training. Evaluate new candidates against a fixed replay/regression suite representing prior capabilities. If later joint training is attempted, do it as a separate versioned experiment with a rollback target.

Control:
- overfitting to the trigger examples;
- repeated creation of near-duplicate units;
- invocation loops or mutually dependent units;
- unit dependency cycles;
- unbounded library growth;
- hidden resource spikes from simultaneous staging buffers;
- false-positive candidate promotion;
- train/test leakage;
- degradation caused by quantization or backend conversion.

Maintain a dependency DAG and reject cycles. Impose a hard maximum on candidate units, training steps, concurrent loads, and memory consumption per cycle.

## 9. Compare growth strategies rather than assuming one winner

Run matched experiments:
- **G0**: fixed base model, no growth;
- **G1**: base model plus manually authored validated units;
- **G2**: base model plus learned candidates, no recursive generation;
- **G3**: recursive candidate generation and gated admission;
- **G4**: candidate generation plus periodic merge/distill/prune;
- **G5**: controlled neuron/connection-level topology growth, only if the architecture and backend support it.

Measure quality, held-out transfer, regressions, memory, latency, training cost, and growth efficiency (benefit per added byte and per training compute). The experiments must be able to show that recursive growth is worse than a fixed model or a small curated library.

## 10. Mobile deployment invariants

The whole Android application runtime peak must remain below 4096 MiB, with a target promotion gate below 3800 MiB. Count base weights, active units, activations, KV cache, staging buffers, backend allocations, UI, and runtime together.

Keep candidate training off-device unless a separately measured training setup fits the same explicit budget. The phone can initially perform inference-time selection, loading, evaluation of lightweight checks, and evidence collection, while candidate training occurs in a controlled development environment. Never assume external storage alone makes the RAM cost zero: loading and conversion require temporary memory.

## 11. Recommended first implementation sequence

1. Finalize the unit manifest and lifecycle-state schema.
2. Implement a registry validator and compatibility/resource preflight checks.
3. Create a small static example library using dummy or toy units.
4. Implement a candidate evaluation harness comparing baseline vs. baseline-plus-unit.
5. Add a candidate admission report with explicit pass/fail/inconclusive outcomes.
6. Add one real adapter or subgraph only after identifying the actual model architecture, checkpoint format, training stack, and inference backend.
7. Prototype model-proposed candidates only after the gated pipeline works.
8. Explore topology-changing neuron growth only after a compatible training and runtime representation exists.

## 12. Success criterion

The central claim is not “more units means a smarter model.” The claim to test is: a unified model can propose or select a small, useful computational extension, learn it in isolation, validate its transfer, and selectively retain or integrate it so that measured task capability improves at an acceptable resource cost.

Until the candidate-generation and admission loop is implemented and tested, recursive growth remains a proposed architecture, not an existing autonomous-learning capability.
