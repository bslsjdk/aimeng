# Dynamic Parameter Activation and Loading Architecture (Research Proposal)

Status: architecture proposal and experiment contract. This document does not claim that dynamic parameter paging or adaptive model growth is implemented.

## 1. Goal

AIMENG investigates one unified core model that can adapt the amount of computation and the resident parameter working set to the current task. The goal is not merely to select among a fixed set of experts. It is to coordinate:

1. **Compute activation**: which parameterized operations participate in the current forward pass.
2. **Parameter residency**: which weights are resident in RAM/VRAM/NPU memory at a given moment.
3. **Capacity expansion**: whether the current capability is insufficient and a validated optional subgraph/adapter should be made available.
4. **Learning and promotion**: whether an explored change measurably improves held-out performance without regressions.

These are distinct mechanisms. Skipping computation does not necessarily unload weights; paging weights does not necessarily reduce total FLOPs; increasing available parameters does not guarantee better answers.

## 2. Relation to MoE

This is not a claim that every MoE is inferior. Conventional sparse MoE typically routes tokens or activations through a fixed collection of experts. AIMENG's research question is broader: can a model coordinate a variable compute graph and a bounded, dynamically managed parameter working set, potentially at finer granularity than whole experts, while preserving coherent behavior?

The design may use routing-like decisions as one component. It must not assume that arbitrary individual weights can be removed without changing model function. Useful units must preserve computational dependencies and valid tensor shapes.

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
