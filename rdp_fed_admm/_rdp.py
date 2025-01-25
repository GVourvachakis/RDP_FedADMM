"""Formulas for various DP noise mechanisms."""

import numpy as np
from numpy.random import Generator

from ._types import FArr, MDPNoise
from .utils._etc import call_gettr


def _rdp_gaussian_noise(
    sensitivity: float,
    rdp_params: tuple[float, float],
    size: tuple[int, ...] | int = 1,
    rng: Generator | None = None,
) -> FArr:
    """Noise generator for an RDP Gaussian mechanism."""
    alpha, epsilon = rdp_params

    if rng is None:
        rng = np.random.default_rng()

    scale = alpha / (2 * epsilon) ** 2
    scale *= sensitivity**2

    return rng.normal(scale=scale, size=size)


_DP_MECHS: dict[str, MDPNoise] = {
    "rdp_gaussian": _rdp_gaussian_noise,
}


def get_mechanism(mech: str) -> MDPNoise:
    return call_gettr(mech, _DP_MECHS)
