from copy import deepcopy
from logging import getLogger

import numpy as np

from rdp_fed_admm._types import FArr, MLoss

log = getLogger(__name__)
__all__ = [
    "get_loss"
]

def _l_mae(X: FArr, Y: FArr) -> float:
    """Mean Absolute Error (L1-norm) loss."""
    assert X.shape == Y.shape
    return np.abs(X - Y).mean()


def _l_mse(X: FArr, Y: FArr) -> float:
    """Mean Squared Error (L2-norm) loss."""
    assert X.shape == Y.shape
    return (X @ Y) / X.shape[0]


_LOSSES: dict[str, MLoss] = {
    "mae": _l_mae,
    "mse": _l_mse,
}

def get_loss(name: str) -> MLoss:
    try:
        loss = deepcopy(_LOSSES[name])
    except KeyError:
        log.error(f"{name} is an invalid nonconformity measure")
        raise ValueError(
            f"{name} is an invalid measure, please pick one of" +
            ", ".join(_LOSSES.keys())
        )

    return loss
