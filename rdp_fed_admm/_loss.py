from logging import getLogger

import numpy as np

from ._types import FArr, MLoss
from .utils._etc import call_gettr

log = getLogger(__name__)
__all__ = [
    "get_loss"
]


def _l_mae(X: FArr, Y: FArr) -> float:
    """Mean Absolute Error (L1-norm) loss."""
    assert X.shape == Y.shape
    abs: float = np.abs(X - Y).mean()
    return abs


def _l_mse(X: FArr, Y: FArr) -> float:
    """Mean Squared Error (L2-norm) loss."""
    assert X.shape == Y.shape
    mse: float = ((X - Y)**2).mean()
    return mse


_LOSSES: dict[str, MLoss] = {
    "mae": _l_mae,
    "mse": _l_mse,
}


def get_loss(name: str) -> MLoss:
    return call_gettr(name, _LOSSES)
