# Copy-cycle guard overlap audit — 2026-08-11

This audit was performed before the first model forward pass. It is a bounded
search, not proof that no private or differently named implementation exists.
It must be refreshed within 72 hours of authorizing the sealed test.

## Sources checked

- Fetched `ScrollPrize/villa` and searched its pull requests for `cycle
  consistency`, `cycle guard`, `round trip`, `copy displacement`, and `sheet
  switch`. Upstream `main` was `d0de07fb694e16934d5db0a5c8fc4fdd223fca7c`.
- Searched public GitHub code for combinations of Vesuvius, cycle consistency,
  round trip, and copy displacement.
- Searched the local Discord export described by
  `discord_channel_summaries_last_2_months/00_OVERVIEW_INDEX.md`: 15 channels,
  3,523 messages, with `#general` and `#unrolling-vc3d` covered through
  2026-08-08. The archive does not establish what was posted after that date.
- Inspected the public ScrollAnchor README and the archived Plumbline post.

## Closest public work

### ScrollAnchor

[`olgaiv39/scroll-anchor`](https://github.com/olgaiv39/scroll-anchor) diagnoses
normal drift and neighboring-sheet switches on an already reconstructed TIFXYZ
by sampling CT intensity along surface normals. It emits review confidence and
has optional conservative correction proposals. Its published real-geometry
benchmark injects controlled corruptions; the README says naturally occurring
real annotation failures have not yet been evaluated against full ground truth.

This is complementary but not the same experiment. The copy-cycle guard:

- operates on the output of Villa's neural adjacent-wrap displacement model;
- runs the same model forward and back and uses `||F_back(F(X)) - X||` without
  CT-profile thresholds;
- evaluates baseline, correction, rejection, source-stay, wrong-sign, and
  shifted-residual arms on natural consecutive annotated wraps; and
- freezes an unseen-scroll PHerc0343P test before generating any model output.

### Plumbline

[Plumbline](https://github.com/ScrollPrize/villa/pull/1013) scans 2D ink
prediction renders for row-direction, spacing, texture, and sheet-jump
anomalies. It is a review-prioritization tool and does not run or correct the
3D neural copy-displacement model.

### Villa displacement scaling

[Villa PR #1284](https://github.com/ScrollPrize/villa/pull/1284) adds opt-in
displacement scaling across voxel sizes. It addresses coordinate-scale transfer,
not round-trip confidence, branch selection, correction, or rollout rollback.

### Other cycle-consistency discussion

The Discord export contains one 2026-08-05 question about using cycle
consistency to compare max-flow and random-walk track-graph methods. That is a
different surface-construction lane; no linked implementation of neural
copy-displacement round trips was found.

## Decision

Proceed with the copy-cycle guard as a distinct core-tracing experiment. Claims
must remain narrow: it is not a general replacement for ScrollAnchor, Plumbline,
or tracing, and value is unproven until the frozen validation and unseen-scroll
gates are run. A fresh GitHub and Discord overlap audit is mandatory before the
sealed test and before any result PR.
