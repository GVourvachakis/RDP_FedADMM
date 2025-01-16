"""Formulas for various DP noise mechanisms."""

from ._types import MDPNoise
from ._utils import call_gettr


def _rdp_gaussian_noise(
    sensitivity: float,
    rdp_params: tuple[float, float],
) -> float:
    """Noise stdev for an RDP Gaussian mechanism."""
    alpha, epsilon = rdp_params

    scale = alpha / (2 * epsilon) ** 2
    scale *= sensitivity**2

    return scale


_DP_MECHS: dict[str, MDPNoise] = {
    "rdp_gaussian": _rdp_gaussian_noise,
}


def get_mechanism(mech: str) -> MDPNoise:
    return call_gettr(mech, _DP_MECHS)
