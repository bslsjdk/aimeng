# Iterative recurrent concept-cloud network

This experiment extends the first concept-cloud router with a persistent neuron-state update loop.

## Intended behavior

1. Encode the recent context into an initial network state.
2. Activate a bounded set of neurons.
3. Keep several candidate concept clouds alive at once rather than committing to a single concept.
4. Let active neurons send messages over a sparse graph.
5. Update the same state through a gated recurrent rule.
6. Re-score candidate clouds and repeat for a configurable number of cycles.
7. Score tokens only inside the candidate clouds at the final cycle.

The network's state is repeatedly revised within one prediction. This is not a Transformer stack of independent attention layers. The current prototype resets the state for each new input sequence; persistent conversation/task state across successive inputs is a separate future feature.

## Run a smoke test

```bash
python scripts/experiment_recurrent_concept_cloud.py --self-test
```

Tests:

```bash
python -m unittest tests.test_concept_cloud tests.test_recurrent_concept_cloud
```

## Important caveats

- Cloud assignments are currently synthetic/explicit. No semantic clustering algorithm has been validated.
- Training and inference both score only router-proposed clouds. The gold target cloud is never injected into token scoring. Token loss is applied only when the router includes the target cloud; router loss still trains on every example. Report router top-k recall separately.
- The returned token-logit tensor is dense for ease of integration, but token scores outside the candidate clouds are masked. The prototype still allocates this dense tensor, so it does not yet prove memory savings or end-to-end speedup.
- The prototype is not integrated into the Android AIMG runtime or the existing production model exporter.
- A fixed cycle count is used initially. Adaptive stopping should only be added after quality, stability, and latency are measured.
- "Continuously running" here means recurrent state refinement during inference, not that every neuron runs continuously while the phone is idle. The eventual mobile scheduler should sleep inactive compute and resume it when a task or event arrives.

## Required evaluation

Compare full-vocabulary and recurrent-cloud models on the same training/validation/test split, tokenizer, seed set, and training budget. Measure language loss, top-1 token accuracy, cloud recall@k, ambiguity/sense accuracy, actual scored token count, active-neuron count, wall-clock latency, throughput, peak RAM, parameter count, and serialized size. Include random cloud assignments as a negative control. Do not claim a win from sparse masks alone.


### Iteration notes: sparse message work and state-aware reactivation

- Message projections now gather active source neurons before the learned message
  projection, rather than projecting every neuron and masking inactive sources
  afterward. The graph still accumulates messages into destination neurons, so
  this is a compute optimization whose benefit must be measured end-to-end.
- Active-neuron re-selection now combines context affinity with affinity from the
  just-updated recurrent state. This is intended to reduce loss of useful state
  between cycles; it is an experimental change, not a proven quality improvement.
- CI records a three-seed synthetic multi-hop comparison, including longer
  sequence tests, and publishes summary metrics plus a JSON artifact. These
  results are screening evidence only; the workload is small and synthetic.
