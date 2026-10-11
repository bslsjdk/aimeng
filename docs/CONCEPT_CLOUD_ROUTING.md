# Concept-cloud sparse routing experiment

## Research hypothesis

Given a context, first predict a small set of likely next-concept clouds, then score tokens only inside those clouds. The objective is to avoid computing a full vocabulary distribution at every decoding step.

This is deliberately separated into two claims:

1. **Sparse computation:** only selected cloud members receive token scores.
2. **Parameter compression:** exact duplicate concept representations are deduplicated, while related concepts share reusable parameters and retain separate residuals.

The current prototype tests claim 1 only. It does **not** yet prove parameter compression or semantic clustering quality.

## Critical semantic rule

A high similarity score is not proof of equivalence. Cloud construction must distinguish:

- exact duplicates / verified equivalent representations: eligible for deduplication;
- related concepts: may share a common representation but must keep concept-specific information;
- polysemous forms and context-dependent senses: must remain distinguishable, with the active sense inferred from context.

Never merge two senses merely because they share a spelling or often co-occur.

## Prototype

Run the shape/gradient smoke test:

```bash
python scripts/experiment_concept_cloud.py --self-test
```

The model takes an explicit token-to-cloud assignment. During training, it uses the gold token's cloud for the local token loss and a separate router cross-entropy loss to train context-to-cloud prediction. During inference, it routes to the top-k predicted clouds. Tests assert that tokens outside the selected cloud are not scored.

## What this does not establish yet

- The current prototype still stores one embedding row per vocabulary token.
- The routing output tensor is dense for API convenience, although only selected token scores are calculated.
- The prototype does not yet produce semantic cloud assignments or prove that those assignments preserve meaning.
- A successful smoke test is not evidence of lower end-to-end latency or higher language-model quality.

## Required next experiment

Compare against a full-vocabulary decoder with identical data, tokenizer, context encoder, training budget, and random seeds. Evaluate at least:

- validation next-token loss and top-1 accuracy;
- router top-k recall of the gold token's cloud;
- average number of token scores actually calculated per step;
- measured end-to-end latency and throughput;
- parameter count and serialized model size;
- ambiguous-sense accuracy and exact-duplicate consistency;
- performance at matched parameter budgets and matched quality targets.

Use train-only statistics to build clouds. Keep validation/test data out of clustering. Include an unclustered baseline, random clouds as a negative control, and a cloud-assignment ablation. Report confidence intervals across multiple seeds.

## Decision rule

Do not claim success merely because fewer tokens are scored. A useful result must retain language quality and sense distinctions while reducing measured compute and/or model size. Sparse routing also needs a fallback path when the router misses the gold cloud; otherwise a single routing error can make the correct token impossible to emit.
