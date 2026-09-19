"""Torch reference sampler for the concatenated spiral gap-logit lattice."""

import torch

from spline_basis import cubic_bspline_weights


def sample_gap_logits_bspline(logits, normalised_z, normalised_spiral):
    """Sample ``[z, spiral]`` control points before the nonlinear gap map.

    Coordinates use the existing align-corners lattice convention in [0, 1].
    The spiral axis concatenates successive turns: both sides of an internal
    turn boundary sample the same lattice coordinate, without separate padding.
    Queries are clamped to the lattice; z and the outer spiral edge replicate
    coefficients. At the inner edge the caller pins column zero to zero and
    the ghost column is -column one. This odd extension keeps the sampled
    origin exactly zero, which the cumulative radii need for theta continuity.
    A one-column lattice has a zero ghost. Singleton z dimensions are valid.

    This eager reference supports coordinate/control-point autograd without
    Triton. It materialises the 4-by-4 stencil; no speed or memory improvement
    over the bilinear kernels is implied.
    """
    nz, ns = logits.shape
    z, s = torch.broadcast_tensors(normalised_z, normalised_spiral)
    z = z.clamp(0., 1.) * (nz - 1)
    s = s.clamp(0., 1.) * (ns - 1)
    zlo, slo = z.floor(), s.floor()
    wz = cubic_bspline_weights(z - zlo)
    ws = cubic_bspline_weights(s - slo)
    offsets = torch.arange(-1, 3, device=logits.device).reshape(
        4, *([1] * z.ndim))
    zi = (zlo.to(torch.long)[None] + offsets).clamp(0, nz - 1)
    si = slo.to(torch.long)[None] + offsets
    # Only the first tap can cross the inner edge after query clamping.
    # Combine its negative ghost weight with column one's positive weight
    # before gathering. At the origin this is exactly zero in value and in
    # coefficient backward, without cancelling scattered floating gradients.
    ghost_weight = torch.where(si[0] < 0, ws[0], torch.zeros_like(ws[0]))
    ws = torch.stack((ws[0] - ghost_weight, ws[1],
                      ws[2] - ghost_weight, ws[3]))
    si = si.abs().clamp(0, ns - 1)
    values = logits[zi[:, None], si[None]]
    return ((values * ws[None]).sum(1) * wz).sum(0)
