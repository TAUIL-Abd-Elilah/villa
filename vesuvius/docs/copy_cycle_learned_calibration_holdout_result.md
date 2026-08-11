# Cycle-conditioned copy calibration: development holdout result

Status: **negative; this lane stops before validation**.

The model and gates were frozen before source wraps 3 and 4 completed. The exact model
was trained only on source wraps 1 and 2, then scored once on the three untouched
directions `3->2`, `3->4`, and `4->3`: 4,087 eligible cells in total.

## Primary result

| Arm | Penalized target distance | Change from baseline | Coverage |
|---|---:|---:|---:|
| baseline | 19.0593 | - | 100% |
| **cycle + displacement** | **17.8628** | **6.28% better** | 100% |
| cycle only | 19.4250 | 1.92% worse | 100% |
| local displacement only | 18.6602 | 2.09% better | 100% |
| fitted scalar displacement | 18.6664 | 2.06% better | 100% |
| physical 4.8/4.317 scalar | 18.5487 | 2.68% better | 100% |

The combined arm improved all three held-out directions: 7.87% on `3->2`, 9.06% on
`3->4`, and 3.73% on `4->3`. It also beat the best displacement-only control by 3.70%
(an incremental gain equal to 3.60% of the baseline penalty), while preserving every
eligible prediction. Valid-cell p95 improved from 60.0531 to 58.2232.

These positive secondary results do not override the preregistered gate. Aggregate
distance improved by 6.28%, below the required 10%. Sheet-switch rate fell from 28.99%
to 26.69%, a 7.93% relative reduction; because the baseline rate exceeded 0.5%, the
frozen rule required a 25% reduction (at most 21.75%). Both conditions failed.

## Frozen gate

| Condition | Result |
|---|---|
| aggregate distance improves at least 10% | **fail** |
| at least two of three directions improve | pass (3/3) |
| no direction worsens by more than 2% | pass |
| beats every displacement-only control | pass |
| incremental gain over best control >= 1% of baseline | pass (3.60%) |
| coverage unchanged in every direction | pass |
| p95 no more than 5% worse | pass (3.05% better) |
| sheet-switch rule | **fail** |

The conjunction is false. The model will not be refit on wraps 3 and 4, PHerc0500P2
validation inference will not run, and the PHerc0343P test remains sealed. Parameters
will not be retuned on this holdout.

## Reproducibility record

- scorer implementation: `3530ecfb31635ede8e776f6eb60311d5b9017afe`
- inference implementation: `65e7ea1c9db5a4a215a0dfb7e965a63f90c40379`
- checkpoint SHA-256: `22cf4392f2f61e6a5548c7b68148e97fed4ee772abf4f842cc6b8d1ef3ca1370`
- manifest SHA-256: `c41622dd79bf02f7bcd451252bee4d2f635470b5547cf14b388dabc0a03d1264`
- model SHA-256: `c0372c888fa90b428075cbb9536e99589e9df034eb3a06c8a88d4f7c252b7e04`
- source-3 receipt SHA-256: `8e5a69330eb517c7f29e8cbbfcaaa9bd9741403a87dc9705c2e939ac60fe68f1`
- source-4 receipt SHA-256: `f16966f77eab19891e58f1b21ce2d93e1a0d4e1bca084157469c2f0f654e9da8`
- score SHA-256: `e61f902d3cf1ec8ee8c2e230c09ef7e554fa2487aadbfebcabbd14d9f53805ee`

Artifacts: [frozen model](copy_cycle_results/development_holdout_model.json) and
[complete score](copy_cycle_results/development_holdout_score.json).

The first automated scoring launch produced no output: it supplied a clean-checkout
copy of the manifest whose line endings gave SHA-256 `3b5e...`, and the scorer rejected
it against the already locked `c416...` byte hash. The successful invocation changed no
model, receipt, manifest content, code, metric, or gate; it supplied the exact `c416...`
manifest bytes already recorded in both the model and inference receipts.
