from typing import Any

from .._types import FArr
from ..admm import ADMMClient, ADMMServer

"""Lasso Objective.

Lasso regularization promotes sparsity in the objective by adding the
l_1 norm as an extra term:
    R := l_1 || x ||₁
where l_1 is an adjustable hyperparameter.

The way the optimization problem is formulated under ADMM, the proximal
of this operation amounts to a soft-thresholding operator applied as the
`z`-update.
"""

class LassoADMMClient(ADMMClient):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class LassoADMMServer(ADMMServer):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    @staticmethod
    def soft_threshold(X: FArr, thresh: float) -> FArr: ...
