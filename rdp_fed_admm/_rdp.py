"""Formulas for various DP noise mechanisms."""

from logging import getLogger

import numpy as np
from numpy.random import Generator

from ._types import FArr, MDPNoise
from .utils._etc import call_gettr

__all__ = [
    "get_mechanism",
]

log = getLogger(__name__)


def _rdp_gaussian_noise(
    sensitivity: float,
    rdp_params: tuple[float, float] = (4,1),
    size: tuple[int, ...] | int = 1,
    rng: Generator | None = None,
) -> FArr:
    """Noise generator for an RDP Gaussian mechanism."""
    alpha, epsilon = rdp_params

    if rng is None:
        rng = np.random.default_rng()

    scale = alpha / (2 * epsilon) ** 2
    # scale *= sensitivity

    log.info(f"Scale: {scale:.4g}, Sens: {sensitivity:.4g}")

    return rng.normal(scale=scale, size=size)


_DP_MECHS: dict[str, MDPNoise] = {
    "rdp_gaussian": _rdp_gaussian_noise,
}


def get_mechanism(mech: str) -> MDPNoise:
    return call_gettr(mech, _DP_MECHS)
