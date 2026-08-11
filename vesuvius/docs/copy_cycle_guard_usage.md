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

The native `vc.Volume` reader is preferred when its Python binding is
available. Otherwise inference uses the project's existing read-only Zarr
reader and persistent `volume_cache_dir`; this is useful on machines where the
C++ binding is not built. The receipt records the selected backend plus the
Python, Torch, Zarr, codec, filesystem, and volume-cartographer versions. The
fallback does not reinterpret data: it exposes the same ZYX slices to the crop
reader.

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
  "method": "copy_cycle_scalar_grid_v1",
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

## 5. Fit and score the frozen learned-calibration extension

The supervised extension is separate from the negative scalar v1 arm. Fit its
publicly frozen model on completed source-1/2 development receipts:

```bash
python -m vesuvius.neural_tracing.evaluation.train_copy_cycle_calibration \
  --receipt /work/development-source01/run_receipt.json \
  --receipt /work/development-source02/run_receipt.json \
  --manifest vesuvius/docs/copy_cycle_pherc0500p2_manifest.json \
  --data-root /data/pherc0500p2 \
  --training-stage holdout \
  --implementation-commit FULL_GIT_COMMIT \
  --output /work/calibration-holdout-model.json
```

Then score the untouched source-3/4 development holdout exactly once:

```bash
python -m vesuvius.neural_tracing.evaluation.score_copy_cycle_calibration \
  --model /work/calibration-holdout-model.json \
  --receipt /work/development-source03/run_receipt.json \
  --receipt /work/development-source04/run_receipt.json \
  --manifest vesuvius/docs/copy_cycle_pherc0500p2_manifest.json \
  --data-root /data/pherc0500p2 \
  --stage development_holdout \
  --implementation-commit FULL_GIT_COMMIT \
  --output /work/calibration-holdout-score.json
```

Both commands reject incomplete receipts, duplicate directions, altered
inference arguments, non-frozen model parameters, provenance mismatches, and
existing output files. The supplied implementation commit must be the exact
HEAD of a clean worktree. Same-scroll scoring also requires the training
commit, checkpoint, manifest, volume, scale, TIFXYZ voxel convention, crop,
runtime, and copy arguments to match; the prospective cross-scroll test may
change only the preregistered dataset-specific fields. The score includes the
combined model, cycle-only and displacement-only ablations, fitted scalar
control, physical 4.8-um scaling control, every direction, and every
preregistered gate.

If and only if the development holdout passes, refit once on all six frozen
development directions by adding the source-3 and source-4 receipts and using
`--training-stage final`. Run the untouched validation inference, then score
its single complete receipt:

```bash
python -m vesuvius.neural_tracing.evaluation.score_copy_cycle_calibration \
  --model /work/calibration-final-model.json \
  --receipt /work/validation/run_receipt.json \
  --manifest vesuvius/docs/copy_cycle_pherc0500p2_manifest.json \
  --data-root /data/pherc0500p2 \
  --stage validation \
  --implementation-commit FULL_GIT_COMMIT \
  --output /work/calibration-validation-score.json
```

A learned validation may authorize the unseen test only with this object:

```json
{
  "schema_version": 1,
  "method": "copy_cycle_local_linear_v1",
  "status": "authorized",
  "validation_status": "validation_positive",
  "implementation_commit": "FULL_GIT_COMMIT",
  "calibration_model_path": "/work/calibration-final-model.json",
  "calibration_model_sha256": "64_LOWERCASE_HEX_CHARACTERS",
  "validation_score_path": "/work/calibration-validation-score.json",
  "validation_score_sha256": "64_LOWERCASE_HEX_CHARACTERS",
  "validation_receipt_path": "/work/validation/run_receipt.json",
  "validation_receipt_sha256": "64_LOWERCASE_HEX_CHARACTERS",
  "validation_public_url": "https://github.com/OWNER/REPO/tree/RESULTS_COMMIT/results",
  "overlap_audit_utc": "2026-08-11T01:00:00Z"
}
```

Use that file with the sealed-test runner command in section 4. The runner
checks the final model, validation score, and validation receipt hashes; exact
code, checkpoint, manifest, runtime, crop, volume, and copy-argument
provenance; all four directions; and every frozen positive gate. It records
the selected method and model SHA in the sealed inference receipt. Score that
receipt only with the same model:

```bash
python -m vesuvius.neural_tracing.evaluation.score_copy_cycle_calibration \
  --model /work/calibration-final-model.json \
  --receipt /work/pherc0343p-test/run_receipt.json \
  --manifest vesuvius/docs/copy_cycle_pherc0343p_manifest.json \
  --data-root /data/pherc0343p \
  --stage sealed_test \
  --implementation-commit FULL_GIT_COMMIT \
  --output /work/calibration-sealed-test-score.json
```

The sealed scorer rejects a different model SHA even when all other
provenance fields match.
