"""Focused native crossing-kernel regression tests without Torch."""

import unittest

import numpy as np

try:
    from vc_spiral import track_crossings
except ImportError:
    track_crossings = None


def _line_track(track, length=9):
    points = np.zeros((length + 1, 3), dtype=np.float32)
    points[:, 0] = track * 32
    points[:, 2] = np.arange(length + 1, dtype=np.float32)
    return points


@unittest.skipUnless(track_crossings is not None,
                     'vc_spiral.track_crossings is not built')
class NativeTrackCrossingsTests(unittest.TestCase):
    def test_repeated_partners_with_multiple_workers_match_serial(self):
        """Stress both direct-table atomic stages in ``resample_tracks``.

        Every row has three entries, and seven rows contend for track zero as
        their partner.  The first atomic loop counts those repeated anchors;
        the second reserves their slots.  Sorting later makes the intended
        result independent of worker scheduling, so a four-worker run must
        exactly match the serial reference.
        """
        track_count = 8
        tracks = [_line_track(track) for track in range(track_count)]
        coordinates = np.concatenate(tracks)
        offsets = np.arange(
            0, len(coordinates) + 1, len(tracks[0]), dtype=np.int64)
        partners = np.zeros((track_count, 3), dtype=np.int32)
        partners[0] = 1
        self_local = np.tile(np.asarray([2, 4, 6], dtype=np.int32),
                             (track_count, 1))
        partner_local = np.tile(np.asarray([3, 5, 7], dtype=np.int32),
                                (track_count, 1))
        kwargs = dict(
            coordinates=coordinates,
            offsets=offsets,
            crossing_partners=partners,
            crossing_self_local=self_local,
            crossing_partner_local=partner_local,
            minimum_spacing=1.0,
            maximum_spacing=2.0,
        )

        serial = track_crossings.resample_tracks(workers=1, **kwargs)
        parallel = track_crossings.resample_tracks(workers=4, **kwargs)

        for key in ('coordinates', 'source_local', 'offsets', 'lengths',
                    'crossing_self_sample', 'crossing_partner_sample'):
            np.testing.assert_array_equal(parallel[key], serial[key])
        self.assertTrue((parallel['crossing_self_sample'] >= 0).all())
        self.assertTrue((parallel['crossing_partner_sample'] >= 0).all())
        self.assertEqual(parallel['minimum_observed_spacing'],
                         serial['minimum_observed_spacing'])
        self.assertEqual(parallel['maximum_observed_spacing'],
                         serial['maximum_observed_spacing'])
        self.assertEqual(parallel['undersized_anchor_gaps'],
                         serial['undersized_anchor_gaps'])


if __name__ == '__main__':
    unittest.main()
