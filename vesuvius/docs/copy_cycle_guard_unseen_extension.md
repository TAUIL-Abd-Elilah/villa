# Cycle-guarded neural copy tracing: unseen-scroll extension

Status: **locked before any model forward pass** on 2026-08-11.

This document extends, but does not rewrite, the preregistration in
`copy_cycle_guard_preregistration.md`. The original PHerc0500P2 protocol was
published at commit `2b17ce3f2` before downloading or inspecting the model.
Checkpoint inspection then proved that PHerc0500P2 was one of five training
datasets. No inference was run. This extension therefore promotes a different,
model-unseen scroll to the primary sealed test before candidate development.

All algorithms, parameter grids, target-assignment rules, metrics, null arms,
coverage accounting, and anti-gaming rules in v1 remain unchanged except where
this document explicitly changes the primary test set and counts.

## Checkpoint fact that triggered the extension

The exact official checkpoint is frozen in
`copy_cycle_checkpoint_provenance.json`:

- Hugging Face revision `4da532350f40d84347991731fc25124fc07afbc1`
- 1,698,235,447 bytes
- SHA-256
  `22cf4392f2f61e6a5548c7b68148e97fed4ee772abf4f842cc6b8d1ef3ca1370`
- training scrolls in the embedded config: PHerc0139, PHerc1667,
  PHerc0500P2, PHercMANBp, and PHercParis4
- PHerc0343P is not listed

The file was loaded with `weights_only=True` and `mmap=True` to inspect its
configuration. No model was instantiated and no forward pass was made.

PHerc0500P2 remains the development/validation scroll because exposure is not
a problem for tuning a guard around this fixed production model. Its sealed
v1 block remains a secondary within-training-scroll replication, not the
headline generalization result.

## Primary sealed test: PHerc0343P

Discord guidance from Bruniss on 2026-07-26 identifies PHerc0343P as consecutive
annotations from a small fragment and instructs users to score only the inner
70% of each TIFXYZ. Seven public segments carry the explicit sequence labels
`-5, -4, -3, -2, -1, 0, 1` and form six consecutive edges:

`-5--4, -4--3, -3--2, -2--1, -1-0, 0-1`

Both directions are evaluated, producing twelve sealed directed tasks. All
files and URLs are frozen in `copy_cycle_pherc0343p_manifest.json`.

The bucket also contains an older generically named `20250511003658-tifxyz`
surface. It is excluded before inference because it is not part of the
explicitly numbered `b2` sequence and is spatially disjoint. On central-70%
ground-truth geometry at level 1, its best median distance to any numbered
surface is 576.1 voxels, while consecutive numbered-pair medians are
23.6--43.3 voxels. The excluded file remains hashed in the manifest so this
decision is auditable.

### Volume and coordinate convention

- Native CT:
  `PHerc0343P/volumes/20260304131111-2.215um-0.4m-111keV-masked.zarr`
- Inference `--volume-scale 1`
- Effective model voxel: 4.43 um, close to the checkpoint's stated 4.8 um
  training resolution
- TIFXYZ files are stored in native 2.215 um coordinates. The scorer divides
  coordinates by 2 before applying every v1 distance, densification, residual,
  penalty, and threshold. Thus all frozen numerical gates remain in effective
  inference voxels.

### Sealing rule

PHerc0343P model outputs may not be generated until all of the following are
public on the branch:

1. the implementation and unit tests;
2. the complete PHerc0500P2 development/validation receipt;
3. the selected `alpha` and `tau` or a validation-negative declaration;
4. exact commands and output schemas; and
5. a fresh GitHub/Discord overlap audit.

If v1 is validation-negative, PHerc0343P remains unopened. If it passes,
PHerc0343P is run once. Runtime repairs follow the v1 no-partial-metric rule.

## Adjusted primary success gate

The six v1 conditions are retained, with only the task-count condition changed:

1. coverage is at least 90% of baseline on every one of twelve directions and
   in aggregate;
2. aggregate penalized mean target distance improves by at least 10%;
3. at least nine of twelve directed tasks improve penalized mean distance;
4. the same v1 sheet-switch relative-improvement/non-inferiority rule passes;
5. candidate valid-cell p95 is no more than 5% worse than baseline; and
6. real cycle residual ranks bad cells better than the shifted-residual null.

The source-stay and wrong-sign correction arms must also be reported and beaten
as defined in v1. Every direction is shown separately.

## Rollout endpoint

Run two six-step chains, `-5 -> 1` and `1 -> -5`, with the v1 fixed chart-side
and rollback rules. Later targets after a stop receive the missing penalty. The
PHerc0500P2 rollouts remain secondary.

## Interpretation

A passing PHerc0343P result would demonstrate prospective transfer of an
inference-time guard to a scroll absent from the checkpoint's declared training
datasets. It would still cover one unseen fragment, not all scroll conditions.
The result must be described as cross-scroll evidence, not a guarantee of
production reliability or a guaranteed prize.
