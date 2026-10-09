# External and Internal Neural Units: One Capability, Multiple Residency States

Status: research specification. This document defines a design and testable contracts; it does not claim that arbitrary neurons can already be inserted into a production model or that autonomous structural learning has been implemented.

## 1. Core hypothesis

An external neural unit and an internal neural unit should not be treated as fundamentally different kinds of intelligence. They are computational structures that may occupy different lifecycle and deployment states.

- **Base-internal**: part of the original model architecture and checkpoint.
- **External candidate**: created through model-driven learning, human construction, or an imported source, but not yet approved for use.
- **External validated**: passed compatibility and scoped evaluation checks; available to the same core model when requested.
- **Loaded/active**: its parameters are resident and its computation is connected to the current execution plan.
- **Integrated**: incorporated into a versioned model graph/checkpoint or an equivalent compiled representation.
- **Retired/quarantined**: unavailable for normal execution but retained with provenance for rollback or analysis.

These are states of a computational artifact, not separate AI agents. One unified core model remains responsible for task reasoning and decisions.

The hypothesis does not mean that any arbitrary neuron can be copied into any model. A unit's function depends on its incoming/outgoing connections, tensor dimensions, normalization, position, training distribution, and runtime semantics. Integration requires a compatible interface and a validated graph transformation.

## 2. Two origins, one unit contract

The library may accept units from at least three origins:

1. **Model-originated learning**: the core model identifies a capability gap, proposes a structural or parameter change, and runs an isolated learning trial.
2. **Human-authored units**: a developer creates a candidate adapter, connection block, subgraph, or compatible neuron group.
3. **Imported units**: an existing compatible module is brought into the library with source, license/provenance, artifact hash, and compatibility metadata.

Origin affects trust and provenance, not the execution contract. All origins must pass the same structural, compatibility, resource, and evaluation gates. A model-generated unit is not trusted merely because the model generated it; a human-authored unit is not trusted merely because a human supplied it.

## 3. Canonical unit manifest

Every stored unit should have a versioned manifest containing at least:

- `unit_id`, `unit_version`, `origin_kind`, `origin_ref`, `created_at`;
- artifact file hashes and serialization format;
- base model family, checkpoint hash, tokenizer/config revision where relevant;
- granularity: `neuron`, `neuron_group`, `channel_block`, `layer`, `adapter`, or `subgraph`;
- input/output tensor contracts, dtype, shape, layout, quantization metadata;
- required dependencies, insertion points, upstream/downstream connections;
- parameter bytes, estimated activation bytes, scratch/staging requirement, load cost;
- backend compatibility and supported operations;
- training data provenance and data eligibility/split information where applicable;
- validation suite/version, measured scope, results, limitations, regression report;
- lifecycle state, parent revision, promotion record, rollback target.

Manifest metadata does not itself prove that a unit is valid. Hash verification proves artifact byte identity only; separate checks must establish graph compatibility and task utility.

## 4. External does not mean text memory

A neural unit must contain executable numerical structure or a well-defined reference to it, not merely a prose description of a skill. A unit may consist of:

- parameters and their required connections;
- an adapter attached at a named compatible insertion point;
- a callable subgraph with declared tensor inputs/outputs;
- a structural patch that transforms a known graph revision into a new graph revision.

A natural-language note can explain when to use a unit, but it is metadata, not the unit's learned computation.

## 5. Lifecycle and transitions

Recommended lifecycle:

`PROPOSED -> STRUCTURE_CHECKED -> COMPATIBILITY_CHECKED -> EVALUATING -> VALIDATED_EXTERNAL -> LOADABLE -> ACTIVE`

Optional integration path:

`VALIDATED_EXTERNAL -> INTEGRATION_TRIAL -> INTEGRATED_VERSIONED`

Failure paths:

`any_candidate_state -> REJECTED`, `ACTIVE -> QUARANTINED`, or `INTEGRATED_VERSIONED -> ROLLED_BACK`.

Every transition records the old/new state, unit hash, base-model revision, actor/reason, verifier identity/version, evidence references, resource budget, and timestamp. Promotion must be atomic. Never overwrite the known-good base checkpoint in place.

## 6. How the core model uses an external unit

1. Identify a possible capability need. A failed task alone is insufficient; classify alternative causes such as missing context, poor strategy, tool failure, ambiguity, verifier failure, or resource limit.
2. Search the unit registry using task features and validated applicability metadata.
3. Check base revision, tensor contracts, dependencies, backend support, validation scope, and artifact hashes.
4. Estimate total peak memory, including weights, activations, KV cache, staging buffers, runtime and backend allocations.
5. Load only if the plan remains within the hard budget and expected benefit justifies load latency.
6. Execute through the same core model's graph and capture output, latency, memory, and verifier evidence.
7. Unload when no in-flight computation references the unit; preserve cache only when budget permits.
8. Update the unit's measured applicability evidence. One successful use does not automatically promote or integrate it.

A unit must never be allowed to inject arbitrary host code into the runtime. Initial units should use a declarative graph/adapter format and an allowlisted operator set. If custom code is ever needed, it requires a separately isolated sandbox with strict resource limits.

## 7. What “integrating into the model” means

Integration is not copying a file into a checkpoint directory. It must define a reproducible transformation from a known base revision to a new versioned computation graph or parameter artifact.

Supported integration modes may include:

- **Reference integration**: the graph references a validated external adapter/subgraph, while its weights remain separately stored.
- **Packaged integration**: unit weights are bundled into a new immutable model package but remain logically identifiable and removable.
- **Structural integration**: a validated graph transformation adds nodes/edges or changes connections and emits a new checkpoint/graph revision.

Each mode needs a reverse transformation or a known-good rollback target. Integration must verify shape compatibility, numerical stability, quantization/backend compatibility, and regressions on the fixed evaluation suite.

Reference integration is the safest first step. It provides the conceptual experience of the unit participating as part of the model without requiring immediate surgery on the base checkpoint. It also allows the runtime to load and unload the unit on demand.

## 8. How new units are learned

The unified model may propose a candidate when repeated, evidenced failures suggest a stable capability gap. Candidate creation happens in an isolated trial:

1. collect an approved training signal and preserve the exact task/artifact/verifier trace;
2. determine whether a compatible adapter, connection block, or subgraph is the smallest testable change;
3. train or construct only in a bounded candidate workspace;
4. freeze the candidate artifact and record its hash;
5. evaluate against the unchanged baseline on development, held-out transfer, and regression suites;
6. reject, revise, or promote to the validated external library according to predeclared gates;
7. integrate only after external validation and compatibility checks.

The model's own assertion that it learned something is not evidence. Do not train on the final held-out test set, and do not treat a single task success as proof of general capability.

## 9. Fine-grained neurons versus practical first implementation

The long-term research concept may describe individual neurons and their connections as units. The first implementation must use the smallest units the actual model and backend can represent, load, execute, and unload efficiently.

Recommended progression:

1. compatible low-rank adapter or small named subgraph;
2. channel/neuron block with explicit tensor and connection contracts;
3. supported layer/sublayer;
4. genuinely dynamic neuron/edge topology, only after the backend can serialize, execute, train, validate, and roll back it.

Do not claim neuron-level memory savings if the backend still keeps the entire layer resident. Track separately: active operations, resident parameter bytes, total stored artifact bytes, and activation/scratch memory.

## 10. Resource and safety invariants

- The Android whole-application runtime peak must remain below 4096 MiB; the target promotion gate is below 3800 MiB.
- Budget estimates include the base model, active unit weights, activations, KV cache, loading/staging buffers, runtime, UI, and backend allocations.
- Reject a load or integration plan before execution if its estimated peak exceeds the budget.
- Never evict a unit with in-flight references or while a dependent active unit requires it.
- Verify hashes before loading and recheck compatibility after model/backend revision changes.
- Use atomic state changes and preserve a known-good rollback version.
- Unit outputs must be scoped to their tested domain; unknown compatibility or unavailable validation must remain explicit.
- Keep candidate training isolated from the stable base model.

## 11. Experiments that can falsify the design

Compare under matched tasks and budgets:

- base model without external units;
- base model plus retrieval-only notes (control for text-memory benefit);
- base model plus a compatible validated external unit;
- same unit packaged/integrated into a versioned model graph;
- unit disabled or randomly selected (control for routing/selection quality).

Measure task success and independent verification, first-pass success, repair rate, held-out transfer, regression rate, actual peak memory, bytes loaded, cold/warm latency, throughput, load failures, and rollback success.

The hypothesis gains support only if units provide repeatable held-out benefits or a useful resource/quality trade-off, while preserving compatibility and rollback. If a unit helps only the exact task used to create it, mark its scope narrow rather than claiming a general ability.

## 12. Implementation boundary

This document is a design contract. It does not yet implement a canonical unit registry, structural graph mutation, learned neuron creation, real parameter paging, or runtime integration. The existing parameter-workspace simulator only simulates residency bookkeeping. Before implementing real integration, inspect the actual training architecture, checkpoint format, inference backend, and supported graph operations.
