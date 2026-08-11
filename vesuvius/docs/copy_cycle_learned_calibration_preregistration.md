# Cycle-conditioned copy calibration: development holdout protocol

Status: **locked before scoring source wraps 3 and 4** on 2026-08-11.

The preregistered scalar cycle correction in
`copy_cycle_guard_preregistration.md` was negative on the first three
development directions. This extension asks a different, explicitly
supervised question:

> Can a small calibration learned from development ground truth transfer to
> untouched source surfaces, and does the round-trip vector add value beyond
> displacement calibration alone?

This is not a rewrite of the v1 result. V1 remains reportable as negative.
The learned calibration may proceed to validation only if it passes the
development holdout and novelty gates below.

## Evidence available before this lock

Complete inference and scores were available for source wraps 1 and 2, giving
the directions `1 -> 2`, `2 -> 1`, and `2 -> 3`. A development sweep compared
six no-intercept linear feature sets (world/local coordinates crossed with
cycle-only/displacement-only/both), four ridge penalties (`0.1`, `1`, `10`,
`100`), and four correction caps (`8`, `16`, `32`, unbounded): 96 settings.

The best world-coordinate setting improved micro-averaged penalized distance
by 9.63%, but world coordinates are not rotation invariant and are rejected as
the frozen method. The chosen local-frame setting (`both`, ridge `10`, cap `8`)
improved the three held-out-source folds by 9.08%, with per-direction changes
of +1.51%, +13.23%, and +10.59%. These are exploratory selection numbers, not
validation evidence.

At lock time, source 3 inference was still running. Its forward files existed,
but its round-trip receipt was incomplete and neither source 3 nor source 4
had been scored or used to fit a calibration. Sources 3 and 4 are therefore
the frozen development holdout, containing directions `3 -> 2`, `3 -> 4`, and
`4 -> 3`.

## Frozen model

For each valid source cell, construct an orthonormal chart frame from local
column and row tangents and `cross(column, row)`. Interior tangents use central
differences; a lattice edge or one-sided hole uses the available immediate
neighbor. Gram-Schmidt removes the column component from the row tangent.
Degenerate frames are not calibrated. Express vectors in that frame.
The six inputs are, in order:

1. round-trip vector `Z - X` (three local components); and
2. forward displacement `Y - X` (three local components).

The training target is the vector from `Y` to its nearest point on the dense
central-70% target surface, expressed in the same frame. Training uses source
wraps 1 and 2 only. Each feature is divided by its training RMS; it is not
mean-centered. Fit a no-intercept 6-by-3 ridge regression with penalty `10.0`.
At inference, transform the predicted correction back to world coordinates
and cap its Euclidean norm at `8.0` effective voxels. Cells without a valid
local frame or round trip retain the baseline prediction, so the method never
improves a score by deleting support.

All fitted arms use the same training rows: baseline-eligible cells with a
finite forward prediction, selected return prediction, and local frame. At
application time, the combined and cycle-only arms require a return prediction;
the displacement-only arms apply wherever their forward inputs are valid. No
application mask uses target ground truth.

The implementation must deterministically serialize feature RMS values,
coefficients, training receipt hashes, manifest hash, implementation commit,
ridge, cap, and feature order. Loading must reject a mismatched schema or
provenance record.

## Frozen controls and overlap rule

The same source-1/2 training rows and source-3/4 holdout rows must score:

- unmodified baseline;
- the frozen six-feature cycle-plus-displacement model;
- a three-feature local displacement-only ridge model with the same penalty
  and cap;
- a three-feature local cycle-only ridge model with the same penalty and cap;
- the physical scalar-resolution control that multiplies forward displacement
  by `4.8 / 4.317`, without a correction cap; and
- a training-fitted scalar displacement control, with no intercept and the
  same 8-voxel correction cap.

The fitted scalar correction is
`beta = sum((Y-X) dot (target-Y)) / sum(||Y-X||^2)` on the common local-frame
training rows and produces `Y + beta * (Y-X)`. The physical scalar produces
`X + (4.8 / effective_voxel_size_um) * (Y-X)`. Both apply to every finite
forward cell; only the fitted scalar correction is capped.

Villa PR #1284 already implements optional scalar displacement scaling. Scalar
or displacement-only gains are therefore controls, not novelty claims. The
learned method is considered distinct only if adding the round-trip features
beats both displacement-only controls on untouched sources.

## Development holdout gate

All conditions must hold on the three source-3/4 directions before any
PHerc0500P2 validation inference is run:

1. aggregate penalized target distance improves by at least 10% over baseline;
2. at least two of three directions improve, and no direction worsens by more
   than 2%;
3. it has strictly lower aggregate penalized distance than the local
   displacement-only and both scalar controls;
4. its aggregate gain over the better displacement-only control is at least
   1% of the baseline penalized distance;
5. all-eligible coverage is unchanged from baseline;
6. valid-cell p95 target distance is no more than 5% worse; and
7. the v1 sheet-switch relative-improvement/non-inferiority rule passes.

Failure stops this lane. Parameters may not be retuned on sources 3 or 4.

## Validation and sealed test

If the development holdout passes, refit the exact frozen model once on all
six PHerc0500P2 development directions and serialize it publicly. Run the
existing four-direction PHerc0500P2 validation block once. Validation must
pass conditions 1 and 3-7 above, improve at least three of four directions,
and retain the original v1 coverage requirements per direction. A positive
validation receipt must be public before the unseen PHerc0343P test is
authorized.

The authorization must bind the exact final-model, validation-score, and
validation-receipt SHA-256 values to the unchanged implementation commit. The
sealed inference receipt records that model SHA, and sealed scoring must reject
any other model.

PHerc0343P remains sealed if either development holdout or validation fails.
No world-coordinate model, new feature, changed ridge penalty, changed cap, or
post-hoc threshold may replace the frozen method after this lock.
