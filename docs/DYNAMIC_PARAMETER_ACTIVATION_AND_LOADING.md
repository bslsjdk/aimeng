# Dynamic Parameter Activation and Loading Architecture (Research Proposal)

Status: architecture proposal and experiment contract. This document does not claim that dynamic parameter paging or adaptive model growth is implemented.

## 1. Goal

AIMENG investigates one unified core model that can adapt the amount of computation and the resident parameter working set to the current task. The goal is not merely to select among a fixed set of experts. It is to coordinate:

1. **Compute activation**: which parameterized operations participate in the current forward pass.
2. **Parameter residency**: which weights are resident in RAM/VRAM/NPU memory at a given moment.
3. **Capacity expansion**: whether the current capability is insufficient and a validated optional subgraph/adapter should be made available.
4. **Learning and promotion**: whether an explored change measurably improves held-out performance without regressions.

These are distinct mechanisms. Skipping computation does not necessarily unload weights; paging weights does not necessarily reduce total FLOPs; increasing available parameters does not guarantee better answers.

## 2. Relation to MoE and the neuron-level hypothesis

This project does not categorically reject MoE. Routing can be one useful mechanism, but it is not the whole research target. The broader hypothesis is a single unified model with a very large population of computational units whose participation, working-set size, memory residency, and reasoning policy can adapt to the current task.

The intended direction is more dynamic than selecting one or several fixed experts: the model should be able to choose how much of its available computational substrate to use, which valid pathways to execute, which associated parameters to keep resident, and when to stop expanding because added capacity is not worth its cost. This is a research hypothesis, not a claim that such a system is already implemented or guaranteed to outperform MoE.

A neuron-level conceptual model does not imply that individual neurons are independently loadable in current inference engines. Computation activation, parameter residency, and topology changes are separate controls. Implementation units must respect tensor dependencies, normalization, residual paths, quantization metadata, and backend-supported execution granularity.

## 3. Unified-model invariant

There is one logical core model and one user-facing reasoning process. External storage, a scheduler, a verifier, a cache, and a library of validated computational modules are infrastructure, not additional autonomous AI agents. A module is a callable component of the same model execution, with an explicit interface and provenance.

Potential reusable units:
- contiguous tensor/block groups supported by the backend;
- complete layers or sublayers;
- low-rank adapters;
- validated computational subgraphs with input/output contracts.

A single neuron is not automatically a useful loading unit: its meaning depends on surrounding activations, weights, normalization, residual paths, and downstream connections.

## 4. Three control loops

### A. Task-level capacity controller

Before and during a task, estimate the task class, uncertainty, available memory, deadline, and current capability evidence. Choose an initial execution profile and a bounded next-step policy.

The controller may expand the working set only when the expected value of added capacity exceeds measured latency and memory costs, and the hard resource limits remain satisfied. It must not use self-reported confidence as the only trigger.

### B. Runtime residency manager

Manage the difference between total model artifact size and the parameters currently resident on the execution device.

Required states for each unit: `unloaded`, `loading`, `resident`, `in_use`, `evict_pending`, `failed`. Each unit needs a stable ID, artifact hash, byte size, dependencies, backend compatibility, version, load cost, and validation status.

Required safety rules:
- never evict a unit while an in-flight operation still references it;
- load dependencies before dispatching a computation;
- coalesce concurrent loads of the same unit;
- handle partial-load failure and restore a consistent state;
- keep a small reserve for allocator overhead, activations, KV cache, and backend buffers;
- measure actual process/system memory, not only the sum of weight-file sizes;
- use atomic artifact/version switching and retain a known-good rollback target.

### C. Capability growth and consolidation

If the current model fails a task, collect the external failure signal and investigate possible causes. A failure may be due to missing knowledge, bad reasoning, prompt ambiguity, a tool failure, a faulty verifier, or a resource limit; it must not automatically trigger parameter growth.

Generate candidate changes in an isolated candidate area. Candidates can be adapters, structured subgraphs, or revised routing/activation policies. Run the same baseline and candidate on fixed validation, held-out transfer, and regression sets. Promote only if predefined acceptance conditions are met; otherwise discard or roll back.

The logical core remains one model. Candidate modules are versioned extensions to its computation, not separate chatting agents.

## 5. Execution lifecycle

1. Receive task and restore task state.
2. Estimate initial resource/capacity profile from observable features and prior verified results.
3. Build a dependency-closed compute plan. Reject plans whose estimated peak exceeds the configured budget.
4. Load required parameter units; verify hashes and backend compatibility.
5. Execute, capture tool/verifier evidence, latency, and memory telemetry.
6. If incomplete, classify the reason using evidence. Expand compute or residency only when allowed; otherwise return a bounded failure or request more evidence.
7. Validate output against the task's required criteria.
8. Persist trace, resource measurements, and outcome. Do not promote a candidate merely because it was used successfully once.
9. Freeze the candidate state and release unneeded working-set units when safe.

## 6. Required experiment baselines

Compare at least:
- **B0**: static full resident model / existing backend baseline where feasible;
- **B1**: fixed small working set;
- **B2**: dynamic load/evict with a fixed compute graph;
- **B3**: dynamic compute activation with all relevant weights already resident;
- **B4**: combined dynamic compute activation and dynamic residency;
- **B5**: B4 plus validated candidate adapters/subgraphs.

These ablations isolate whether improvements come from compute sparsity, paging, or additional learned capacity. Do not compare only parameter counts: report quality and resource measurements together.

For every run, record:
- model and artifact hashes, backend/runtime version, device, task split, random seed where relevant;
- resident and peak bytes by memory domain, process RSS/PSS where available, KV-cache and activation estimates;
- load/eviction counts, bytes transferred, cache hit rate, cold/warm latency, total latency, throughput, failures/OOMs;
- first-pass success, verified repair rate, repeated-error rate, held-out transfer, regression rate;
- energy/battery proxy if reliably measurable.

Report median and tail latency, not only the best run. Separate cold-start from warm-cache measurements. A lower resident parameter count is not itself a success if quality collapses or repeated paging makes latency unusable.

## 7. Android resource gate

The whole Android runtime has a hard peak RAM ceiling below 4096 MiB; the target release gate is below 3800 MiB to preserve margin. This includes model pages, activations, KV cache, staging buffers, runtime, UI and backend allocations. These limits are measured on-device, not inferred from parameter file sizes.

Every load plan must have a bounded staging buffer and explicit peak-memory estimate. If a load would exceed the budget, do not attempt it: evict safe units, choose a smaller plan, or fail gracefully. An NPU's memory may be separately accounted for by the platform, but its buffers and transfer staging still need measurement.

## 8. Critical engineering risks

- **Weight paging can be slower than computation.** Measure storage latency and transfer granularity.
- **Fine-grained units may have poor locality.** Start with backend-supported blocks, layers, or adapters rather than arbitrary individual weights.
- **Changing the graph may invalidate assumptions.** Preserve tensor shapes, normalization, residual connections, quantization metadata, and backend operator constraints.
- **Dynamic growth can overfit.** Use held-out tasks, regression suites, versioning, and rollback.
- **The controller can learn to cheat metrics.** Make quality acceptance a hard gate; resource rewards cannot compensate for incorrect output.
- **Verifiers can be wrong or unavailable.** Store verifier identity, version, scope, evidence, and limitations.
- **Self-reported uncertainty is not calibrated probability.** Calibrate with observed outcomes and retain an abstain/stop option.
- **Concurrency can break memory accounting.** Reserve memory for in-flight work and serialize loads/evictions where the backend is not thread-safe.

## 9. Staged implementation plan

Stage 1 — **Trace-only simulation**: model units, dependencies, load/evict state, byte budget, and simulated latency. No model weights are changed. Test state transitions and failure recovery.

Stage 2 — **Real residency manager around a supported backend**: load/unload only backend-supported units. Verify artifact hashes, actual resident memory, cold/warm latency, and recovery after partial failure.

Stage 3 — **Dynamic compute graph**: add a small number of valid block-level activation choices. Confirm skipped operations are truly not executed; metadata masking alone is not evidence of saved compute.

Stage 4 — **Capacity expansion through adapters**: create candidate low-rank adapters or small subgraphs, validate on held-out tasks, and promote/rollback with immutable provenance.

Stage 5 — **Fine-grained parameter structures**: consider dynamic neuron/connection-level growth only if the backend can execute such a graph efficiently and the earlier stages demonstrate measurable gains.

Each stage must have an independent benchmark and rollback plan. Do not skip directly to dynamic single-neuron loading.

## 10. Acceptance criteria

A prototype is successful only if it:
1. keeps measured whole-app peak RAM below the hard ceiling;
2. maintains task quality within a predefined non-inferiority margin or improves it;
3. demonstrates that the intended compute was skipped or parameters actually became non-resident;
4. improves at least one declared resource metric without hiding regressions in other metrics;
5. survives load failure, OOM prevention, interrupted tasks, and rollback tests;
6. generalizes to held-out tasks and does not rely on repeated exposure to the same examples.

The numeric quality margin, latency budget, and target task suite must be fixed before running the final comparison to avoid moving the goalposts.

## 11. Current implementation boundary

This is a research architecture specification. Existing AIMENG correction-trace and verifier code provides parts of the evidence/promotion infrastructure, but it does not implement a runtime parameter pager, dynamic compute graph, adaptive parameter-count controller, or actual model growth. These must be implemented and measured separately.


## 12. First executable artifact: trace-only workspace simulator

The first small implementation is `scripts/simulate_parameter_workspace.py`, with tests in `tests/test_parameter_workspace_simulation.py`. It models units, dependency-closed loads, a byte budget, LRU eviction, in-use pins, deferred unload, and rollback after a simulated load failure. The tests are included in the idea-cycle CI workflow.

Example plan:

```json
{
  "budget_bytes": 100,
  "units": [
    {"unit_id": "base", "size_bytes": 40, "dependencies": []},
    {"unit_id": "math", "size_bytes": 30, "dependencies": ["base"]},
    {"unit_id": "vision", "size_bytes": 50, "dependencies": []}
  ],
  "actions": [
    {"op": "load", "unit_id": "math"},
    {"op": "begin_use", "unit_id": "math"},
    {"op": "end_use", "unit_id": "math"},
    {"op": "load", "unit_id": "vision"}
  ]
}
```

Run it with:

```bash
python scripts/simulate_parameter_workspace.py --input plan.json --output runs/workspace-simulation.json
```

This is intentionally a **trace-only simulation**. It validates scheduler logic and produces simulated resident-byte accounting, but it does not touch model weights, release OS memory, measure Android RAM, or skip actual model computation. Its success is only a prerequisite for a real backend integration, not evidence of runtime speedup or memory reduction.


## 13. Neuron-population control: the intended long-term model

The long-term hypothesis is a single model containing a very large computational substrate. A task should not automatically activate the entire substrate. The model should adapt its effective working capacity and reasoning procedure to the task, while preserving one coherent model state and one user-facing reasoning process.

### 13.1 Four controls, not one vague activation switch

1. **Participation control**: select which neurons, channels, blocks, or computational paths contribute to the current step. The smallest unit must be supported by the actual architecture and backend.
2. **Memory control**: decide which parameter blocks are resident, which are fetched from storage, and which can be evicted after their final in-flight use. Participation masks alone do not release weight memory.
3. **Reasoning-policy control**: choose among direct response, decomposition, broad association, counterexample search, tool-assisted checking, or deeper iterative reasoning. These are execution modes of the same core model, not separate AI agents.
4. **Plasticity control**: if repeated evidence shows a capability gap, propose a candidate adapter/subgraph or, in later research, a topology change. Keep it isolated until independent evaluation and regression gates pass.

### 13.2 A bounded feedback loop

For each task, estimate a starting compute and memory budget; activate a dependency-valid path; observe task progress, verifier evidence, latency, and memory; then choose among continue, expand, switch strategy, stop, or abstain. Expansion must have a measurable reason and remain within the hard resource budget. The model's own confidence may be one feature, never the sole authority.

A failed attempt does not automatically mean “add neurons.” First distinguish missing knowledge, an unsuitable reasoning strategy, a bad tool call, an ambiguous task, a verifier failure, and a resource limit. Only evidence consistent with a capability gap should trigger a capacity-growth trial.

### 13.3 Do not confuse logical neurons with physical memory pages

The design must track at least three different quantities:
- active compute units for the current operation;
- parameter bytes actually resident in each memory domain;
- total available model/module artifact size on storage.

These values can differ dramatically. For example, masking channels can reduce executed operations only if the backend really skips them; it may leave all weights resident. Paging weights can reduce resident memory while increasing I/O and latency. A neuron may share tensors with many other neurons and therefore cannot be unloaded independently.

Start with the smallest units that the actual backend can load and execute efficiently. Fine-grained neuron/connection activation is a later experimental target, not a software promise.

### 13.4 Evidence required to support the hypothesis

Compare fixed compute, dynamic compute only, dynamic residency only, and combined control under matched task sets and quality thresholds. Record actual operation counts where available, resident/peak memory, load bytes, cold/warm latency, quality, first-pass success, repair success, held-out transfer, and regressions. Include a no-growth baseline and a growth-enabled candidate arm.

The hypothesis is supported only if dynamic control preserves or improves task quality while demonstrating a real, measured resource benefit. A bigger neuron count, a more complicated diagram, or the model claiming it “thought harder” is not evidence.
