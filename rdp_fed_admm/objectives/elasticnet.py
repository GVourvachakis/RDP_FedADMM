"""Elastic Net Objective.

The elastic net is a generalized scheme which linearly combines lasso
and ridge regularization as follows:
    R := l_1 || x ||₁ + l_2 || x ||₂²
where l_1 and l_2 are adjustable hyperparameters.

The way the optimization problem is formulated under ADMM, this only
applies to the soft-thresholding operator during the `z`-update.
"""

from typing import Unpack, final, override

from .._types import FArr
from .lasso import (
    LassoADMMClient,
    LassoADMMClientParams,
    LassoADMMServer,
    LassoADMMServerParams,
)

__all__ = [
    "ElasticNetADMMClient",
    "ElasticNetADMMServer",
]


@final
class ElasticNetADMMClient(LassoADMMClient):
    def __init__(self, **kwargs: Unpack[LassoADMMClientParams]) -> None:
        super().__init__(**kwargs)


@final
class ElasticNetADMMServer(LassoADMMServer):
    def __init__(
        self,
        ridge_multiplier: float = 1.,
        **kwargs: Unpack[LassoADMMServerParams],
    ) -> None:
        super().__init__(**kwargs)
        self._ridge_mult = ridge_multiplier

    @override
    def _z_update(self, z: FArr):
        """ElasticNet Proximal Update.

        The elastic net regularization, i.e. the function:
            f(x) := l1 || x ||₁ + l2 || x ||₂²
        has a proximal operator:
            prox(v; l2, f) = prox(v; l1, g) / (1 + 2 ρ l2)
        That is, the proximal operator for the lasso with some
        multiplicative shrinkage.

        References
        ----------
        [1] Boyd 2013; Proximal Algorithms §6.5.3

        """
        shrinkage: float = 1 + 2 * self._penalty_term * self._ridge_mult
        return super()._z_update(z) / shrinkage
