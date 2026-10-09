# Core Neuron Replacement and Activity Tracing

Status: implementation contract and research plan. These schemas and validators do not implement tensor-level hot swapping or prove causal contribution.

## Goals

1. Treat the core model itself as versioned computational units, not only an external candidate library.
2. Give each unit a stable identity, revision, parameter/tensor mapping, explicit input/output contract, dependencies, and replacement policy.
3. Record which units were considered, activated, skipped, deactivated, replaced, or flagged unhealthy for each run.
4. Separate observed participation from measured contribution. A unit being active does not prove it helped.
5. Permit replacement of one unit at a safe boundary, with compatibility checks, baseline comparison, and rollback.

## What “individual unit” means in each phase

A mathematical scalar neuron inside a dense matrix is not automatically a safe hot-swappable software component. Tensor packing, shared matrix operations, normalization, residual paths, quantization scales, and backend kernels can couple its implementation to neighboring values.

Use staged granularity:
- Phase A: adapters or small declared subgraphs with stable tensor contracts.
- Phase B: channel blocks or neuron groups whose tensor slices and dependencies are explicit.
- Phase C: individual neuron/edge patches only when the model architecture and runtime preserve exact indexing, calibration, and numerical behavior.

The manifest supports several granularities, but does not claim every granularity is immediately executable. Unsupported replacement modes must stay disabled or require a restart.

## Unit identity and replacement

A unit identity is stable across revisions; each replacement creates a new immutable revision and artifact hash. Never overwrite the active artifact in place. A candidate must:
1. match the base model revision, interface shapes, dtypes, layout, quantization, and dependency contract;
2. pass integrity and backend/operator checks;
3. pass unit-level tests and end-to-end held-out/regression suites;
4. be evaluated against the unchanged baseline, including latency and peak-memory impact;
5. be activated only at a safe point, with the prior revision available for rollback.

If a defect is suspected, quarantine or shadow-test the candidate first. Do not automatically replace a unit solely because one task failed: the error may come from missing context, routing, another unit, decoding, or an invalid evaluation.

## Activity and outcome records

The activity trace schema defines one append-only event per unit/run. Record at least:
- unit ID and revision, model revision, run/task reference, event type;
- why the unit was selected or skipped;
- activation count/duration and measured latency/memory when available;
- input/output fingerprints and optional activation-summary references;
- outcome status and evidence references.

Do not store raw prompts, private user data, or full activations by default. Use fingerprints, aggregate summaries, explicit access control, retention limits, and opt-in debug captures when needed. The event log should be append-only; corrections should create a linked event rather than silently rewriting history.

Helpful/harmful are evidence-backed labels, not guesses from correlation. Prefer controlled ablations or counterfactual runs: compare the same task, input, seed/configuration, and model revision with the unit enabled versus disabled/replaced. Report uncertainty when interactions prevent clean attribution.

## How records improve the system

Aggregate traces by unit revision and task family. Flag candidates for review when there is repeated, reproducible evidence of regressions, unnecessary activation cost, instability, or low marginal value. Propose a replacement, but gate it through isolated training and validation. Feed only verified correction signals into training; unknown or inconclusive outcomes must not become negative labels.

## Runtime and safety constraints

All simultaneous weights, activations, KV cache, staging buffers, backend allocations, logs, and UI/runtime overhead count toward the Android application peak RAM budget (<4096 MiB; target release gate <3800 MiB). Trace logging must be bounded and asynchronous; avoid retaining full activations by default. Hot swapping must not occur while a unit is in use; defer it to a safe boundary or restart when required.

## First implementation milestones

1. Commit manifest and trace contracts.
2. Add validators and tests for metadata and attribution invariants.
3. Add a registry that checks duplicate IDs, dependency closure/cycles, base-revision mismatches, and legal state transitions.
4. Build a deterministic toy network whose units can genuinely be enabled, disabled, replaced, and ablated.
5. Add paired evaluation reports and trace aggregation.
6. Only then connect real adapters/subgraphs and investigate finer-grained core tensor replacement.

Passing schema tests proves only that records follow the contract. It does not prove a model can hot-swap individual neurons, that trace attribution is causal, or that quality improved.
