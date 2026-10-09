# Composable Neural Network Architecture

Status: proposed architecture for experiments, not a trained or runnable language model.

## 1. Design objective

AIMENG should not assume a conventional Transformer is the only possible core. The research target is a neural network whose first-class computational units have stable identities, explicit interfaces, measurable participation, and controlled replacement. Units combine into larger functional circuits; circuits exchange learned representations and contribute to answer generation.

"Neuron" in this project is an engineering term for a versioned computational unit. At first, a unit may be a small vector-valued module, channel block, gated MLP, attention-like unit, or learned subgraph rather than a single scalar biological-style neuron. The architecture must state its granularity instead of implying that all such units are interchangeable.

## 2. Proposed hierarchy

- **Scalar/feature unit:** small learned transformation or feature detector, where practical.
- **Composable neural unit:** owns parameters, an input/output tensor contract, activation policy, and revisioned health/evaluation records.
- **Circuit:** a directed group of units whose connections and interfaces are declared.
- **Working network:** the units/circuits selected for one task, under a bounded compute and memory budget.
- **Core model:** input encoder, working network, iterative coordination/update steps, and output decoder, trained as one model system.

The hierarchy is an implementation convenience, not a claim that units have consciousness, intentions, or independent human-like understanding.

## 3. Forward pass

1. Encode the user input into learned representation vectors.
2. Initialize a small working set using a learned task-conditioned selector.
3. Each active unit receives typed tensors from upstream units and produces output tensors plus optional bounded diagnostic statistics.
4. A coordination mechanism combines outputs, checks dependencies, and decides whether to continue the current circuit, activate another compatible unit, or stop under the compute budget.
5. Repeat for a bounded number of rounds. Keep state explicit and distinguish persistent learned parameters from per-request activations.
6. A decoder maps the final representation to answer tokens.
7. Record unit participation and costs; evaluate the final answer separately from internal confidence estimates.

This is message passing over a learned computation graph. The first prototype should use synchronous rounds and a deterministic scheduler; asynchronous event-driven execution can be tested later, once correctness is measurable.

## 4. What makes units independently replaceable

Every unit revision needs:
- stable unit ID and immutable revision/artifact hash;
- declared input/output shapes, dtypes, layout and quantization;
- explicit connection endpoints and dependency versions;
- bounded parameter, activation and scratch-memory estimates;
- compatibility tests and task-family evaluation results;
- activation/deactivation policy and safe replacement boundary;
- a rollback reference to the prior known-good revision.

A candidate replacement must preserve the unit's external tensor contract or go through a graph migration that updates dependent units. Never replace a tensor slice blindly if shared normalization, quantization, or neighboring connections depend on it. Initially replace small modules/subgraphs; finer-grained units are a later experiment.

## 5. Training plan

### Stage A: executable toy network
Build a tiny network with several independently parameterized units and explicit edges. Prove that a unit can be activated, skipped, replaced by a compatible revision, and rolled back. Use deterministic synthetic tasks so a failure can be reproduced.

### Stage B: end-to-end supervised learning
Train encoder, unit parameters, coordination mechanism, and decoder jointly on a small task dataset. Start with fixed topology and all units available. Ensure gradients flow through the composed network and compare it with a parameter/compute-matched baseline.

### Stage C: learned participation
Train a selector to choose units while optimizing task loss plus measured compute cost. Penalize unnecessary activation, but do not reward sparsity so strongly that the network simply refuses to compute. Compare against fixed activation and random selection.

### Stage D: fault injection and replacement
Deliberately degrade one unit in a controlled test, detect the regression, train a candidate replacement, evaluate on held-out tasks, then swap only that unit revision and compare with the baseline. Test rollback as a required path.

### Stage E: growth
Detect repeatable capability gaps; propose a candidate unit or circuit; train it in isolation or with a controlled subset of parameters; evaluate held-out transfer, regressions, latency and peak memory; admit it only if it adds measurable value. Repeat with fixed budgets and periodically merge, distill or prune redundant units.

### Stage F: language capability
Only after the toy network, training loop, instrumentation, and replacement path work, scale toward token-level language modeling. Compare against a small standard baseline on the same data and compute budget. A novel graph is not inherently better; it must demonstrate quality, efficiency, or replaceability advantages.

## 6. Core experiments

- **A0:** fixed dense/fully connected toy baseline.
- **A1:** composable units with fixed participation.
- **A2:** learned dynamic unit selection.
- **A3:** iterative multi-round message passing.
- **A4:** single-unit replacement with rollback.
- **A5:** gated addition of trained candidate units.
- **A6:** compare against a small Transformer or MLP baseline at matched parameter, training-token, and compute budgets.

Report task accuracy/loss, held-out generalization, calibration, activated-unit count, total FLOPs or a clearly stated proxy, wall latency, peak resident memory, load/eviction overhead, replacement success rate, and regressions after growth. Do not use activation frequency alone as evidence that a unit is useful.

## 7. Activity logging and attribution

Record which units were considered, selected, activated, skipped, errored, deactivated, or replaced; record unit and model revisions, input/output fingerprints, duration, memory, and references to evaluation evidence. Keep raw prompts and full activations out of routine logs by default.

An active unit is not automatically a helpful unit. Use paired ablations or counterfactual runs where practical, keep unknown outcomes unknown, and only turn evidence-backed corrections into training signals.

## 8. Memory and mobile constraints

For Android, the full application peak must remain below 4096 MiB, with a target release gate below 3800 MiB. Count weights, activations, temporary buffers, cached states, runtime/backend allocations, logging and UI. Dynamic unit selection reduces compute only if inactive units actually skip work; unloading parameters reduces residency but adds transfer and latency costs. Measure both.

## 9. Important unresolved research questions

- What is the smallest unit granularity that remains trainable and computationally efficient?
- Which unit function family should be the first primitive: gated MLP, low-rank adapter, small recurrent state-space block, attention-like unit, or mixed set?
- How should selector gradients be trained without unstable discrete decisions?
- How much interaction between units can be allowed while preserving safe single-unit replacement?
- Does iterative routing improve generalization enough to offset scheduling and memory overhead?

Answer these experimentally, not by assuming biological resemblance guarantees better intelligence.

## 10. Current implementation status

This document is a design specification. It does not claim that a composable neural network, language model, learned selector, single-unit hot swap, or recursive growth loop already exists. The next concrete milestone is a deterministic toy network with explicit units, edges, a forward pass, activity traces, and tests for activation/replacement/rollback.
