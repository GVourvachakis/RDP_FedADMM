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
            f(x) := || x ||₁ + (γ/2) || x ||₂²
        for γ > 0 (i.e. a linear combination of l1 and l2
        regularization) has a proximal operator:
            prox(v; λ, f) = prox(v; λ, l1) / (1 + λγ)
        That is, the proximal operator for the lasso with some
        multiplicative shrinkage.

        References
        ----------
        [1] Boyd 2013; Proximal Algorithms §6.5.3

        """
        ridge_threshold: float = self._ridge_mult / self._penalty_term
        ridge_threshold /= self._n_clients

        shrinkage: float = 1 + self._step_size * self._ridge_mult

        return super()._z_update(z) / shrinkage
