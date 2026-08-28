import unittest

import numpy as np
import torch

from tracks import (
    get_track_satisfied_counts,
    get_track_satisfied_counts_in_chunks,
)


DR_PER_WINDING = 12.81
BASE_WINDING = 40
METRICS_CONFIG = {
    'satisfaction_radius_tolerance': 0.45,
    'satisfaction_distance_tolerance': 6.0,
}


class _IdentityTransform:
    def __call__(self, points):
        return points

    @property
    def inv(self):
        return self


def _track_on(winding, point_count=24):
    theta = np.linspace(0.3, 1.3, point_count)
    radius = (
        winding * DR_PER_WINDING
        + theta / (2 * np.pi) * DR_PER_WINDING
    )
    return np.stack([
        np.full_like(theta, 1000.0),
        np.sin(theta) * radius,
        np.cos(theta) * radius,
    ], axis=-1).astype(np.float32)


class TrackWindingModeTests(unittest.TestCase):
    def test_chunked_modes_preserve_complementary_winding_signal(self):
        displacements = (0.0, 0.5, 1.0, 2.0, 5.5, 23.0)
        tracks = [
            _track_on(BASE_WINDING + displacement)
            for displacement in displacements
        ]

        satisfied, total, modes = get_track_satisfied_counts_in_chunks(
            _IdentityTransform(), torch.tensor(DR_PER_WINDING), tracks,
            METRICS_CONFIG, chunk_size=2, return_mode_windings=True,
        )

        self.assertEqual(satisfied.tolist(), [24, 0, 24, 24, 0, 24])
        self.assertEqual(total.tolist(), [24] * len(tracks))
        self.assertEqual(modes[[0, 2, 3, 5]].tolist(), [40, 41, 42, 63])
        # Exact half-winding controls are rounding ties.  Float evaluation can
        # select either neighbour, and no diagnostic should rely on that mode
        # because the ordinary satisfaction count already rejects the track.
        self.assertIn(int(modes[1]), (40, 41))
        self.assertIn(int(modes[4]), (45, 46))
        # Fractional errors are caught by satisfaction while whole-winding
        # errors are caught by the mode comparison; neither replaces the other.
        self.assertEqual((satisfied == total).tolist(),
                         [True, False, True, True, False, True])
        self.assertEqual((modes[[0, 2, 3, 5]] - BASE_WINDING).tolist(),
                         [0, 1, 2, 23])

    def test_default_chunked_result_remains_a_pair_and_matches_direct(self):
        tracks = [_track_on(BASE_WINDING), _track_on(BASE_WINDING + 1)]

        chunked = get_track_satisfied_counts_in_chunks(
            _IdentityTransform(), torch.tensor(DR_PER_WINDING), tracks,
            METRICS_CONFIG, chunk_size=1,
        )
        _, direct_satisfied, direct_total, _, _ = get_track_satisfied_counts(
            _IdentityTransform(), torch.tensor(DR_PER_WINDING), tracks,
            METRICS_CONFIG,
        )

        self.assertEqual(len(chunked), 2)
        torch.testing.assert_close(chunked[0], direct_satisfied)
        torch.testing.assert_close(chunked[1], direct_total)

    def test_opt_in_empty_result_is_three_int64_tensors(self):
        result = get_track_satisfied_counts_in_chunks(
            _IdentityTransform(), torch.tensor(DR_PER_WINDING), [],
            METRICS_CONFIG, return_mode_windings=True,
        )

        self.assertEqual(len(result), 3)
        for tensor in result:
            self.assertEqual(tensor.dtype, torch.int64)
            self.assertEqual(tensor.numel(), 0)


if __name__ == '__main__':
    unittest.main()
