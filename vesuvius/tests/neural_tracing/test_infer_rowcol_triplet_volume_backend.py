from types import SimpleNamespace

import numpy as np

from vesuvius.neural_tracing.inference import infer_rowcol_triplet_wraps as inference


class _FakeArray:
    shape = (5, 6, 7)
    chunks = (2, 3, 4)
    dtype = np.dtype("uint8")

    def __getitem__(self, key):
        return np.arange(8, dtype=np.uint8).reshape(2, 2, 2)[key]


def test_zarr_backend_is_used_when_native_binding_is_unavailable(monkeypatch):
    calls = []

    def fake_open_zarr(path, scale, config):
        calls.append((path, scale, config))
        return _FakeArray()

    monkeypatch.setattr(inference, "vc", None)
    monkeypatch.setattr(inference, "open_zarr", fake_open_zarr)

    volume, level = inference._open_vc_volume_level(
        "https://example.test/volume.zarr/",
        volume_scale=2,
        cache_dir="cache",
        chunk_cache_gb=3.0,
        retry_seconds=17.0,
    )

    assert level == 2
    assert volume.backend_name == "zarr"
    assert volume.shape == (5, 6, 7)
    assert volume.chunks == (2, 3, 4)
    assert volume.dtype == np.dtype("uint8")
    assert calls == [
        (
            "https://example.test/volume.zarr/",
            2,
            {
                "volume_cache_dir": "cache",
                "volume_cache_retry_seconds": 17.0,
            },
        )
    ]


def test_native_backend_remains_preferred_when_available(monkeypatch):
    class FakeVolume:
        dtype = np.dtype("uint16")

        def set_cache_budget(self, value):
            self.cache_budget = value

        def shape_at(self, level):
            assert level == 1
            return (9, 10, 11)

        def chunk_shape(self, level):
            assert level == 1
            return (3, 4, 5)

        def has_scale_level(self, level):
            return level == 1

    opened = FakeVolume()
    fake_volume_type = SimpleNamespace(open_url=lambda path, cache_root: opened)
    monkeypatch.setattr(inference, "vc", SimpleNamespace(Volume=fake_volume_type))

    volume, level = inference._open_vc_volume_level(
        "https://example.test/volume.zarr/",
        volume_scale=1,
        cache_dir="cache",
        chunk_cache_gb=0.25,
    )

    assert level == 1
    assert volume.backend_name == "vc"
    assert volume.shape == (9, 10, 11)
    assert opened.cache_budget == int(0.25 * (1024**3))
