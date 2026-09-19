"""Shared uniform cubic B-spline basis for control-point lattices."""

import torch


def cubic_bspline_weights(f):
    """Weights on ``floor(x) - 1 .. floor(x) + 2``, for ``f = frac(x)``.

    Stacks the four nonnegative weights on a new leading axis. These are
    control-point B-spline weights, not interpolating cubic image weights.
    """
    f2 = f * f
    f3 = f2 * f
    return torch.stack([
        (1. - f) ** 3 / 6.,
        (3. * f3 - 6. * f2 + 4.) / 6.,
        (-3. * f3 + 3. * f2 + 3. * f + 1.) / 6.,
        f3 / 6.,
    ])
