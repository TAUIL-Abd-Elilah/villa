# Copy-cycle benchmark usage

The protocol is split into three commands so data verification, expensive model
inference, and scoring leave independent receipts.

## 1. Verify benchmark bytes

```bash
python -m vesuvius.neural_tracing.evaluation.prepare_copy_cycle_benchmark \
  --manifest vesuvius/docs/copy_cycle_pherc0500p2_manifest.json \
  --destination /data/pherc0500p2
```

Add `--download-missing` only when missing files should be fetched. Existing
files with the wrong size or SHA-256 are rejected and never silently replaced.

## 2. Run forward and round-trip inference

Copy the phase-specific example config, replace local paths, and run. The
validation and sealed-test examples intentionally carry identical `copy_args`:

```bash
python -m vesuvius.neural_tracing.inference.run_copy_cycle_experiment \
  --config /work/copy-cycle-development.json
```

The output directory must be absent or empty. The runner hashes all ground
truth and the checkpoint, loads the model and volume once, then saves both
forward branches and both return branches for every source. It atomically
updates `run_receipt.json` after each completed source. A partial receipt has
`"completed": false` and must not be scored as a complete phase.

Development may use a subset of its source block. Validation and sealed test
must contain every source endpoint declared by their manifest edges.

## 3. Score the validation grid

```bash
python -m vesuvius.neural_tracing.evaluation.score_copy_cycle_experiment \
  --receipt /work/validation/run_receipt.json \
  --manifest vesuvius/docs/copy_cycle_pherc0500p2_manifest.json \
  --data-root /data/pherc0500p2 \
  --mode grid \
  --output /work/validation/validation_score.json
```

The scorer recomputes return-branch choice, derives branch-to-target assignment
from baseline only, evaluates all 40 frozen `(alpha, tau)` settings, and applies
the preregistered coverage and direction gates. It writes strict JSON; undefined
metrics are `null`, never nonstandard `NaN` tokens.

## 4. Authorize and run the unseen test

Only a `validation_positive` result can authorize test inference. Publish an
authorization object before running the test:

```json
{
  "schema_version": 1,
  "status": "authorized",
  "validation_status": "validation_positive",
  "implementation_commit": "FULL_GIT_COMMIT",
  "selected": {"alpha": 0.5, "tau": 24.0},
  "validation_score_path": "/work/validation/validation_score.json",
  "validation_score_sha256": "64_LOWERCASE_HEX_CHARACTERS",
  "validation_receipt_path": "/work/validation/run_receipt.json",
  "validation_receipt_sha256": "64_LOWERCASE_HEX_CHARACTERS",
  "validation_public_url": "https://github.com/OWNER/REPO/tree/RESULTS_COMMIT/results",
  "overlap_audit_utc": "2026-08-11T01:00:00Z"
}
```

The runner verifies both referenced files byte-for-byte. It requires the score
to be a positive four-direction PHerc0500P2 validation grid, the score and
receipt to use the exact checked-out implementation and frozen checkpoint, and
the selected parameters and requested inference `copy_args` to match. The
authorization must also point to the
public GitHub copy of those results. The overlap audit timestamp must be no
more than 72 hours old. Then run:

Publishing result artifacts creates a different Git commit. Run the sealed
phase from a separate clean worktree checked out at the exact
`implementation_commit`; do not change code after validation.

```bash
python -m vesuvius.neural_tracing.inference.run_copy_cycle_experiment \
  --config /work/copy-cycle-pherc0343p-test.json \
  --test-authorization /work/test-authorization.json
```

Score only the already-selected parameters:

```bash
python -m vesuvius.neural_tracing.evaluation.score_copy_cycle_experiment \
  --receipt /work/pherc0343p-test/run_receipt.json \
  --manifest vesuvius/docs/copy_cycle_pherc0343p_manifest.json \
  --data-root /data/pherc0343p \
  --mode fixed --alpha 0.5 --tau 24 \
  --output /work/pherc0343p-test/test_score.json
```

The fixed scorer reports every directed edge, source-stay and wrong-sign nulls,
real-versus-shifted residual discrimination, and the complete primary-gate
decision. It refuses grid search on a sealed-test receipt, refuses parameters
that differ from the receipt, and refuses to overwrite an existing score file.
