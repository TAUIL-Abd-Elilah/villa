import math
from types import SimpleNamespace

import numpy as np
import torch

from satisfaction_metrics import (
    _ListPatchAtlas,
    evaluate_patch_satisfaction_packed,
    report_absolute_winding_diagnostic,
)
from tracks import get_track_satisfied_counts_in_chunks


class _IdentityTransform:
    def __call__(self, points):
        return points

    def inv(self, points):
        return points


def _spiral_point(theta, winding, *, dr=10.0, radial_noise=0.0):
    radius = (winding + theta / (2 * math.pi)) * dr + radial_noise
    return torch.tensor(
        [0.0, math.sin(theta) * radius, math.cos(theta) * radius],
        dtype=torch.float32,
    )


def _one_quad_patch(theta, winding, *, radial_noise=0.0):
    # The native evaluator scores the mean of four corners.  Repeating the
    # analytic point keeps that mean exactly on the requested winding while
    # retaining a normal one-quad patch layout.
    point = _spiral_point(theta, winding, radial_noise=radial_noise)
    return SimpleNamespace(
        zyxs=point.repeat(2, 2, 1),
        valid_quad_mask=torch.ones((1, 1), dtype=torch.bool),
        area=1.0,
    )


def _evaluation(patches):
    return evaluate_patch_satisfaction_packed(
        _IdentityTransform(), torch.tensor(10.0), patches,
        _ListPatchAtlas(patches, torch.device('cpu')),
        -1, 1, include_splicing=False)


def _absolute_anchor(patch_id, winding, *, point_id=1):
    return {
        'id': 10,
        'name': 'absolute-anchor',
        'source_file': 'abs_winding.json',
        'metadata': {'winding_is_absolute': True},
        'points_by_patch': {
            patch_id: [{
                'id': point_id,
                'winding_annotation': float(winding),
                'on_patch': {'id': patch_id, 'ij': [0.0, 0.0]},
            }],
        },
    }


def test_whole_winding_shift_stays_native_satisfied_but_direct_anchor_flags_it():
    patches = [
        _one_quad_patch(0.2, 40),
        _one_quad_patch(0.2, 41),
        _one_quad_patch(0.2, 42),
        _one_quad_patch(0.2, 63),
    ]
    evaluation = _evaluation(patches)

    # This is the native metric's blind spot: each exactly-on-winding patch
    # passes even though three carry an absolute annotation for winding 40.
    assert evaluation.profiles['strict'].satisfied_patches.tolist() == [True] * 4

    report = report_absolute_winding_diagnostic(
        ['correct', 'shifted-one', 'shifted-two', 'shifted-twenty-three'], evaluation,
        [
            _absolute_anchor('correct', 40),
            _absolute_anchor('shifted-one', 40),
            _absolute_anchor('shifted-two', 40),
            _absolute_anchor('shifted-twenty-three', 40),
        ],
    )

    comparisons = {entry['patch_id']: entry for entry in report['comparisons']}
    assert comparisons['correct']['difference_windings'] == 0
    assert comparisons['shifted-one']['difference_windings'] == 1
    assert comparisons['shifted-two']['difference_windings'] == 2
    assert comparisons['shifted-twenty-three']['difference_windings'] == 23
    assert all(entry['native_strict_satisfied'] for entry in comparisons.values())
    assert report['summary']['anchors_disagreeing'] == 3
    assert report['summary']['disagreeing_and_native_strict_satisfied'] == 3


def test_direct_anchor_diagnostic_does_not_turn_small_radial_noise_into_a_switch():
    # Two voxels are inside both default native tolerances for dr=10.  The
    # expected absolute winding and native target still agree, so the diagnostic
    # reports no sheet switch rather than treating ordinary surface scatter as one.
    patches = [_one_quad_patch(0.2, 40, radial_noise=2.0)]
    evaluation = _evaluation(patches)
    assert evaluation.profiles['strict'].satisfied_patches.tolist() == [True]

    report = report_absolute_winding_diagnostic(
        ['noisy-but-correct'], evaluation,
        [_absolute_anchor('noisy-but-correct', 40)],
    )

    assert report['summary']['anchors_compared'] == 1
    assert report['summary']['anchors_disagreeing'] == 0
    assert report['comparisons'][0]['difference_windings'] == 0


def test_only_usable_absolute_anchors_are_compared():
    patches = [_one_quad_patch(0.2, 40)]
    evaluation = _evaluation(patches)
    nonabsolute = _absolute_anchor('p', 9)
    nonabsolute['metadata']['winding_is_absolute'] = False
    nonintegral = _absolute_anchor('p', 40.25, point_id=2)
    unknown_patch = _absolute_anchor('missing', 99, point_id=3)

    report = report_absolute_winding_diagnostic(
        ['p'], evaluation, [nonabsolute, nonintegral, unknown_patch])

    assert report['comparisons'] == []
    assert report['summary']['anchors_compared'] == 0
    assert report['summary']['skipped']['not_absolute'] == 1
    assert report['summary']['skipped']['nonintegral_annotation'] == 1
    assert report['summary']['skipped']['unknown_patch'] == 1


def test_track_mode_windings_are_available_without_changing_default_result_shape():
    tracks = [
        np.stack([_spiral_point(theta, 40).numpy() for theta in (0.1, 0.2, 0.3)]),
        np.stack([_spiral_point(theta, 41).numpy() for theta in (0.1, 0.2, 0.3)]),
    ]
    config = {
        'satisfaction_radius_tolerance': 0.45,
        'satisfaction_distance_tolerance': 6.0,
    }
    default_result = get_track_satisfied_counts_in_chunks(
        _IdentityTransform(), torch.tensor(10.0), tracks, config)
    satisfied, totals, modes = get_track_satisfied_counts_in_chunks(
        _IdentityTransform(), torch.tensor(10.0), tracks, config,
        chunk_size=1, return_mode_windings=True)

    assert len(default_result) == 2
    assert torch.equal(default_result[0], satisfied)
    assert torch.equal(default_result[1], totals)
    assert satisfied.tolist() == totals.tolist() == [3, 3]
    assert modes.tolist() == [40, 41]
