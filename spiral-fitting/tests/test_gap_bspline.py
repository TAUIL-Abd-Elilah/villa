"""Independent numerical contracts for the optional gap B-spline."""

import math

import numpy as np
import pytest
import scipy.ndimage
import torch
import torch.nn.functional as F

from config import (BACKFILLABLE_CONFIG_DEFAULTS, CHECKPOINT_MODEL_SHAPE_KEYS,
                    Config, durable_config)
from gap_interpolation import sample_gap_logits_bspline
from spline_basis import cubic_bspline_weights
from transforms import GapExpanderParams, GapExpandingTransform, SpiralAndTransform


@pytest.fixture(autouse=True)
def _fixed_seed():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(713)
        yield


def _params(nz=5):
    return GapExpanderParams(24, 0., nz * 24., 8, 16.).double()


def _transform(params, **kwargs):
    return GapExpandingTransform(
        params, torch.tensor(16., dtype=torch.float64), 0., params.num_z * 24.,
        gap_expander_lr_scale=.3, interpolation='bspline', **kwargs)


def _pin(field):
    return torch.cat((torch.zeros_like(field[:, :1]), field[:, 1:]), dim=1)


def test_matches_independent_scipy_coefficients_including_ghost_and_borders():
    generator = torch.Generator().manual_seed(71)
    field = _pin(torch.randn(5, 13, generator=generator, dtype=torch.float64))
    points = torch.rand(2, 400, generator=generator, dtype=torch.float64) * 1.8 - .4
    actual = sample_gap_logits_bspline(field, points[0], points[1])
    # SciPy evaluates the spline from coefficients (prefilter=False). Pad
    # explicitly so its boundary machinery cannot supply the odd ghost for us.
    coeff = np.pad(field.numpy(), ((2, 2), (2, 2)), mode='edge')
    coeff[:, 1] = -coeff[:, 3]
    coords = np.clip(points.numpy(), 0, 1) * np.array([[4], [12]]) + 2
    expected = scipy.ndimage.map_coordinates(
        coeff, coords, order=3, prefilter=False, mode='nearest')
    torch.testing.assert_close(actual, torch.from_numpy(expected), rtol=1e-12, atol=1e-12)


def test_basis_partition_support_and_cardinal_values():
    f = torch.linspace(0, 1, 33, dtype=torch.float64)
    weights = cubic_bspline_weights(f)
    assert (weights >= 0).all()
    torch.testing.assert_close(weights.sum(0), torch.ones_like(f), atol=1e-15, rtol=0)
    torch.testing.assert_close(
        weights[:, 0], torch.tensor([1/6, 2/3, 1/6, 0], dtype=torch.float64))
    impulse = torch.zeros(9, 11, dtype=torch.float64)
    impulse[4, 5] = 1
    z = torch.full((9,), .5, dtype=torch.float64)
    s = torch.arange(1, 10, dtype=torch.float64) / 10
    expected = torch.tensor([0, 0, 0, 1/9, 4/9, 1/9, 0, 0, 0], dtype=torch.float64)
    torch.testing.assert_close(sample_gap_logits_bspline(impulse, z, s), expected)


def test_affine_reproduction_interior_and_singleton_z():
    zz, ss = torch.meshgrid(torch.arange(7, dtype=torch.float64),
                           torch.arange(13, dtype=torch.float64), indexing='ij')
    z = torch.tensor([.2, .43, .8], dtype=torch.float64)
    s = torch.tensor([.2, .71, .82], dtype=torch.float64)
    actual = sample_gap_logits_bspline(2 * zz + 3 * ss - 1, z, s)
    torch.testing.assert_close(actual, 2 * z * 6 + 3 * s * 12 - 1)
    field = _pin(torch.full((1, 13), 2., dtype=torch.float64))
    out = sample_gap_logits_bspline(field, torch.tensor([-4., .5, 8.]), s)
    torch.testing.assert_close(out, torch.full_like(out, 2.))


def test_inner_origin_is_zero_with_zero_coefficient_sensitivity():
    raw = torch.randn(5, 11, dtype=torch.float64, requires_grad=True)
    z = torch.tensor([-.3, 0, .13, .5, 1., 1.4], dtype=torch.float64)
    out = sample_gap_logits_bspline(_pin(raw), z, torch.zeros_like(z))
    torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)
    grad, = torch.autograd.grad(out.sum(), raw)
    torch.testing.assert_close(grad, torch.zeros_like(grad), atol=0, rtol=0)


def test_coefficients_and_coordinates_pass_gradcheck():
    field = torch.randn(4, 9, dtype=torch.float64, requires_grad=True)
    z = torch.tensor([.13, .72, -.2], dtype=torch.float64, requires_grad=True)
    s = torch.tensor([.03, .43, .82], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(sample_gap_logits_bspline, (field, z, s))


@pytest.mark.parametrize('axis', ['z', 'spiral'])
def test_value_first_and_second_derivatives_are_continuous_at_interior_knots(axis):
    field = _pin(torch.randn(6, 12, generator=torch.Generator().manual_seed(8), dtype=torch.float64))
    def derivatives(offset):
        t = torch.tensor(2 + offset, dtype=torch.float64, requires_grad=True)
        z, s = (t / 5, t.new_tensor(.57)) if axis == 'z' else (t.new_tensor(.57), t / 11)
        y = sample_gap_logits_bspline(field, z, s)
        d, = torch.autograd.grad(y, t, create_graph=True)
        dd, = torch.autograd.grad(d, t)
        return torch.stack((y.detach(), d.detach(), dd.detach()))
    torch.testing.assert_close(derivatives(-1e-7), derivatives(1e-7), atol=3e-6, rtol=0)


def test_default_is_bitwise_legacy_bilinear_for_values_and_gradients():
    params = _params()
    with torch.no_grad():
        params.logits.normal_(std=.01)
    default = GapExpandingTransform(params, torch.tensor(16.), 0., 120., .3)
    theta = torch.tensor([.01, 1.1, 5.9], dtype=torch.float64, requires_grad=True)
    z = torch.tensor([-20., 61., 200.], dtype=torch.float64, requires_grad=True)
    actual = default.get_logits_by_winding(theta, z)
    idx = params.winding_first_logit_idx
    coords = torch.lerp(idx[:-1], idx[1:], (theta / (2 * torch.pi))[..., None])
    grid = torch.stack((coords / idx[-1] * 2 - 1,
                        (z / 120. * 2 - 1)[..., None].expand_as(coords)), -1)[None]
    expected = F.grid_sample((_pin(params.logits[0, 0]) * .3)[None, None], grid,
                             align_corners=True, mode='bilinear', padding_mode='border')[0, 0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    a = torch.autograd.grad(actual.sum(), (params.logits, theta, z), retain_graph=True)
    b = torch.autograd.grad(expected.sum(), (params.logits, theta, z))
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=0, rtol=0)


def test_spiral_seam_origin_gap_and_ordered_radii():
    params = _params()
    with torch.no_grad():
        params.logits.normal_(std=.02)
    transform = _transform(params)
    theta = torch.tensor([0., 2 * math.pi], dtype=torch.float64)
    z = torch.full_like(theta, 43.)
    logits = transform.get_logits_by_winding(theta, z)
    assert logits[0, 0] == 0
    torch.testing.assert_close(logits[0, 1:], logits[1, :-1], atol=0, rtol=0)
    radii = transform.get_transformed_winding_radii(theta, z)
    torch.testing.assert_close(radii[0, 1:], radii[1, :-1], atol=2e-13, rtol=0)
    assert (radii.diff(dim=-1) >= 1.).all()
    assert radii[0, 1] == 16.


@pytest.mark.parametrize('truncate_frac', [None, 0., .3, 1.])
def test_identity_and_nontrivial_forward_inverse_at_seam_and_z_borders(truncate_frac):
    params = _params(nz=1)
    theta = torch.tensor([1e-7, 2*math.pi-1e-7, .5, 2.2, 5.9], dtype=torch.float64)
    radius = torch.tensor([.2, 2.1, 3.5, 4.7, 5.2], dtype=torch.float64) * 16
    points = torch.stack((torch.tensor([-10., 0, 12., 24., 40.], dtype=torch.float64),
                          theta.sin() * radius, theta.cos() * radius), -1)
    identity = _transform(params, truncate_frac=truncate_frac)
    torch.testing.assert_close(identity(points), points, atol=3e-14, rtol=0)
    with torch.no_grad():
        params.logits.normal_(std=.004)
    transformed = _transform(params, truncate_frac=truncate_frac)
    output = transformed(points)
    torch.testing.assert_close(transformed.inv(output), points, atol=1e-11, rtol=0)
    assert torch.isfinite(output).all()


def test_native_gap_and_backward_use_spline_and_cache_upgrades(monkeypatch):
    params = _params()
    transform = _transform(params)
    theta = torch.tensor([.9, 2.7], dtype=torch.float64)
    z = torch.tensor([31., 89.], dtype=torch.float64)
    with torch.no_grad():
        transform.get_logits_by_winding(theta, z)
    radii = transform.get_transformed_winding_radii(theta, z)
    winding = torch.tensor([1, 3])
    gaps = transform.get_native_log_gaps(winding, theta, z).exp()
    torch.testing.assert_close(gaps, radii.diff(dim=-1)[torch.arange(2), winding])
    gaps.sum().backward()
    assert torch.isfinite(params.logits.grad).all()
    assert params.logits.grad.abs().sum() > 0
    assert torch.count_nonzero(params.logits.grad[..., 0]) == 0
    monkeypatch.setattr('gap_triton.gap_triton_available', lambda *args: True)
    assert not transform._use_triton(torch.zeros(1, 3), transform.dr_per_winding)


@pytest.mark.parametrize('latent', [-100., 1000.])
def test_extreme_coefficients_keep_positive_finite_gaps(latent):
    params = _params()
    with torch.no_grad():
        params.logits.fill_(latent)
    radii = _transform(params).get_transformed_winding_radii(
        torch.tensor([.2, 3.5, 6.1], dtype=torch.float64),
        torch.tensor([0., 55., 120.], dtype=torch.float64))
    assert torch.isfinite(radii).all()
    assert (radii.diff(dim=-1) >= 1.).all()


def test_inverse_coordinate_derivative_and_parameter_gradients():
    params = _params()
    with torch.no_grad():
        params.logits.normal_(std=.002)
    transform = _transform(params)
    transform.dr_per_winding.requires_grad_()
    points = torch.tensor([[43., 21., 43.], [56., 60., -18.]],
                          dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(transform.inv, (points,))
    grads = torch.autograd.grad(transform.inv(points).square().sum(),
                                (params.logits, transform.dr_per_winding, points))
    for grad in grads:
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0


def test_config_and_model_checkpoint_roundtrip_and_shared_tensor_path(tmp_path):
    key = 'model_gap_expander_interpolation'
    cfg = Config({key: 'bspline', 'model_gap_expander_capacity_windings': 8,
                  'model_gap_expander_num_windings': 8}).as_dict()
    low = torch.tensor([0., -192., -192.])
    high = torch.tensor([192., 192., 192.])
    umbilicus = torch.tensor([[0., 0., 0.], [192., 0., 0.]])
    model = SpiralAndTransform(4, 'rk4', low, high, umbilicus, cfg)
    with torch.no_grad():
        model.gap_expander_params.logits.normal_(std=.004)
    transform = model._get_transform_parts()[0]
    shared = model._get_transform_parts(shared=model.get_shared_transform_tensors())[0]
    theta, z = torch.tensor([.2, 3.4]), torch.tensor([12., 78.])
    expected = transform.get_transformed_winding_radii(theta, z)
    torch.testing.assert_close(shared.get_transformed_winding_radii(theta, z), expected, atol=0, rtol=0)
    path = tmp_path / 'spline.ckpt'
    torch.save({'cfg': durable_config(cfg), 'state': model.state_dict()}, path)
    checkpoint = torch.load(path, weights_only=False)
    restored = SpiralAndTransform(4, 'rk4', low, high, umbilicus,
                                  Config(checkpoint['cfg']).as_dict())
    restored.load_state_dict(checkpoint['state'])
    torch.testing.assert_close(restored._get_transform_parts()[0].get_transformed_winding_radii(theta, z),
                               expected, atol=0, rtol=0)
    assert key in CHECKPOINT_MODEL_SHAPE_KEYS
    assert BACKFILLABLE_CONFIG_DEFAULTS[key] == 'bilinear'
    assert Config().as_dict()[key] == 'bilinear'
    with pytest.raises(ValueError):
        Config({key: 'bicubic'})
