# Cycle-guarded neural copy tracing: preregistration v1

Status: **locked before model inference** on 2026-08-11.

This document freezes the first real-scroll evaluation of a cycle-consistency
guard for `scrollprize/copy_displacement_latest`. At the time of this lock, the
PHerc0500P2 ground-truth coordinates had been inspected only to verify files,
choose non-overlapping wrap blocks, and establish that consecutive wraps have
usable spatial overlap. No checkpoint prediction, round-trip residual, guarded
output, validation score, or test score had been generated.

The experiment asks a narrow, falsifiable question:

> Can a round trip from a source surface to a predicted adjacent wrap and back
> identify or correct neural copy-tracing errors without sacrificing more than
> 10% of otherwise scoreable surface support?

This is a core tracing experiment, not a claim that cycle consistency alone
solves unwrapping.

## Why this lane

The March 2026 Kaggle Surface Detection winner already establishes a strong
nnU-Net ensemble plus topology-oriented postprocessing baseline. The current
ecosystem also contains several active merger-repair efforts, including
`Jinhojeong/vesuvius-unmerge`, `IyanDopico/vesuvius-sheet-tools`, Villa PR #975,
and the phantom evaluation in Villa PR #1380. Rebuilding a generic segmenter or
another merger splitter would therefore be substantially duplicative.

By contrast, Villa's iterative copy inference can chain one predicted wrap into
the next but has no round-trip check, confidence map, rollback rule, or
cycle-based correction. Public searches on 2026-08-11 found diagnostics and
scale corrections, but no implementation of this closed-loop guard for the
copy-displacement model. The overlap audit must be refreshed before a result PR
is submitted.

The Discord evidence used for dataset choice is Bruniss's 2026-07-26 guidance
that PHerc0500P2 contains consecutive annotated wraps and that only the inner
70% of each surface should be used. The local Discord archive ends on
2026-08-08; it is not evidence about later work.

## Frozen data and provenance

- Scroll: `PHerc0500P2`
- CT volume: `20250528085330-4.317um-1.2m-111keV-masked.zarr`
- Volume URL:
  `https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0500P2/volumes/20250528085330-4.317um-1.2m-111keV-masked.zarr/`
- Ground truth: the thirteen public `*-on-20250528085330-4.317um.tifxyz`
  surfaces listed and hashed in `copy_cycle_pherc0500p2_manifest.json`
- Model: `scrollprize/copy_displacement_latest`
- Model file and SHA-256: to be recorded immediately after download, before the
  first inference run; this metadata addition may not alter any gate below.
- Coordinates and distances are measured in level-0 4.317 um voxels.

The benchmark downloader must reject any byte whose SHA-256 differs from the
manifest. Ground-truth files are never edited in place.

The public checkpoint may have training exposure to PHerc0500-family data. Its
embedded configuration and release metadata will be audited after download.
If exposure cannot be ruled out, this experiment is an ablation of a guard on a
fixed production model, not evidence of model generalization to an unseen
scroll. A later prospective second-scroll replication is required for a broad
claim.

## Frozen split

An edge `a-b` means both directed copy tasks, `a -> b` and `b -> a`.

| Split | Consecutive-wrap edges | Directed tasks |
|---|---|---:|
| Development | 1-2, 2-3, 3-4 | 6 |
| Validation | 5-6, 6-7 | 4 |
| Sealed test | 9-10, 10-11, 11-12, 12-13 | 8 |

Edges 4-5, 7-8, and 8-9 are unused buffers. Thus no surface in development or
validation is an input or target in the sealed test. Wrap 8 is unused.

- Development outputs may be inspected while implementing the scorer and
  guard.
- Validation may be run once for parameter selection after unit tests pass.
- Test remains sealed until the implementation commit, model hash, selected
  parameters, and validation receipt are public. Test is opened once.
- A runtime failure may be repaired only without inspecting partial test
  metrics. The failure and repair commit must be logged.

## Ground-truth support and target assignment

For every surface, the scoreable UV rectangle is the central 70% of the valid
UV bounding box: 15% is removed from each side in row and column. This rule is
fixed from the dataset author's guidance.

Ground-truth surfaces are densified deterministically from valid neighboring
TIFXYZ quads at at most 5-voxel sample spacing. Quads with an edge longer than
60 voxels are excluded as discontinuities. For a directed task, a source cell
is eligible only when:

1. it is valid and inside the source's central-70% mask; and
2. its nearest point on the target's central-70% dense surface is at most 80
   voxels away.

Eligibility depends only on ground truth, never on an evaluated arm. A missing
or non-finite prediction at an eligible cell is a failure, not a reason to
remove that cell.

The model emits chart-side labels `front` and `back`, while independent source
meshes may have arbitrary UV orientation. For each source, branch-to-target
assignment is the minimum-cost one-to-one assignment computed from the
**unmodified baseline only**, using the penalized mean distance below. The
mapping is then reused unchanged for every candidate and null arm. At a block
endpoint, the better baseline branch is assigned to its one in-split target.
This convention makes the baseline as strong as possible and prevents a
candidate from improving its own assignment.

## Frozen baseline and candidate

The baseline is the repository's unmodified one-pass copy inference with the
checkpoint defaults, TTA enabled, and no iterative guard. Exact arguments,
repository commit, hardware, wall time, and output hashes must be recorded.

For each baseline branch `Y = F(X)`, the same fixed model is run once on `Y`.
Of the two return branches, `Z` is the one with the smaller median
`||Z - X||_2` over finite central-70% source cells. This choice uses no target
ground truth. The per-cell cycle residual is:

`r = ||Z - X||_2`

The guarded/corrected candidate is:

`Y' = Y - alpha * (Z - X)`

and remains valid only where the baseline is valid, the return is finite, and
`r <= tau`. No morphology, connected-component filtering, manual mask editing,
or test-specific repair is allowed in v1.

The only candidate grid is:

- `alpha` in `{0, 0.25, 0.5, 0.75, 1.0}`
- `tau` in `{4, 8, 12, 16, 24, 32, 48, infinity}` voxels

Validation selection is lexicographic:

1. retain at least 90% of baseline eligible coverage in aggregate and on each
   of the four validation directions;
2. minimize aggregate penalized mean target distance;
3. minimize aggregate all-eligible sheet-switch rate;
4. prefer smaller `alpha`, then larger `tau`.

If no non-no-op setting beats the no-op (`alpha=0`, `tau=infinity`) on
penalized mean distance in at least three of four validation directions, v1 is
declared validation-negative and the test is not opened.

## Frozen metrics

Target and wrong-sheet distances use dense central-70% ground-truth point
clouds. The wrong-sheet cloud contains every PHerc0500P2 wrap except the
assigned target, including the source surface.

For each directed task report:

- eligible cells and valid predicted cells;
- coverage = valid / eligible;
- target-distance mean, median, p90, and p95 on valid cells;
- fractions within 10, 20, and 40 voxels;
- penalized mean target distance, assigning `min(distance, 80)` to a valid cell
  and 80 to a missing cell;
- a sheet switch when `wrong_distance + 5 < target_distance`;
- switch rate over valid cells and over all eligible cells.

Aggregate metrics are micro-averaged over eligible cells. Every directed task
is also shown separately; aggregate improvement cannot hide a failed edge.

### Primary sealed-test success gate

All conditions must hold:

1. candidate coverage is at least 90% of baseline coverage both in aggregate
   and on every test direction;
2. aggregate penalized mean target distance improves by at least 10%;
3. at least six of eight directed tasks improve penalized mean distance;
4. if baseline all-eligible switch rate is at least 0.5%, candidate switch rate
   improves by at least 25% relative; otherwise it may worsen by at most 0.1
   percentage point;
5. candidate valid-cell p95 target distance is no more than 5% worse than
   baseline; and
6. the real cycle residual ranks bad baseline cells better than the shifted
   residual null described below.

This gate is intentionally hard. A negative result remains publishable as a
bounded finding but is not evidence that the guard improves tracing.

## Nulls and anti-gaming checks

The scorer must include these frozen controls:

- **Source-stay null:** use `X` as the prediction. It checks that a metric does
  not reward failing to move to the neighboring wrap.
- **Wrong-sign correction:** `Y + alpha * (Z - X)` with the selected alpha and
  tau. The proposed correction must beat this arm.
- **Shifted residual:** cyclically shift `r` by +17 rows and +11 columns within
  each lattice. Compare real and shifted residual AUROC/AUPRC for detecting a
  bad baseline cell, defined as target distance over 40 voxels or a sheet
  switch. This is diagnostic and cannot replace the primary geometric gate.
- **Coverage accounting:** missing eligible cells always receive the 80-voxel
  penalty. Deleting difficult regions cannot improve the primary metric for
  free.

## Rollout evaluation

After one-step test scoring, run the baseline and selected guard from wrap 9
toward wrap 13 and independently from wrap 13 toward wrap 9. The baseline's
first-step target assignment fixes the chart-side label for each chain. The
same label is retained thereafter.

At each step report retained support, cycle-residual quantiles, nearest target
distance, and wrong-sheet rate. If guarded support falls below 90% of the
previous accepted step, the guard rolls back and stops; all later expected
target cells receive the 80-voxel missing penalty. A stop is therefore safe but
not scored as successful completion.

Rollout is a secondary endpoint because only two chains are available. It must
not be substituted for a failed one-step primary gate.

## Visual evidence

The result generator must emit the same fixed panels for all eight test
directions, not selected examples:

1. source, target, baseline, and candidate 3D/UV overlays;
2. baseline target-error, cycle-residual, and accepted-mask heatmaps with fixed
   ranges;
3. CT cross-sections through the median-error and p95-error regions selected by
   the scorer, not by hand; and
4. both complete rollout strips.

Raw machine-readable JSON, output TIFXYZ files, command lines, hashes, and the
plot-generation command are required alongside summary figures.

## Interpretation limits

This v1 test contains one small fragment, four sealed edges, and eight directed
tasks. Cells within an edge are highly correlated; they are not treated as
independent samples. Any bootstrap interval is descriptive and block-based,
and the exact per-direction table is primary. A second scroll, ideally one
outside checkpoint training, is required before claiming robust cross-scroll
improvement.
