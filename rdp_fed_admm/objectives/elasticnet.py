from typing import final, override

from .._types import *
from .lasso import LassoADMMClient, LassoADMMServer


@final
class ElasticNetADMMClient(LassoADMMClient):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


@final
class ElasticNetADMMServer(LassoADMMServer):
    def __init__(
        self,
        addr: str,
        port: int,
        max_clients: int,
        seed: int | None = None,
        step_size: float = 0.3,
        ridge_multiplier: float = 1.,
        penalty_term: float = 0.6,
        subset_size: float = 0.7,
    ) -> None:
        super().__init__(
            addr, port, max_clients,
            seed, step_size, penalty_term, subset_size
        )
        self._ridge_mult = ridge_multiplier

    @override
    def _z_update(self, z: FArr):
        """ElasticNet Proximal Update.

        The elastic net regularization, i.e. the function:
            f(x) := || x ||₁ + (γ/2) || x ||₂²
        for γ > 0 (i.e. a linear combination of l1 and l2 regularization) has
        a proximal operator:
            prox(v; λ, f) = prox(v; λ, l1) / (1 + λγ)
        That is, the proximal operator for the lasso with some multiplicative
        shrinkage.

        References
        ----------
        [1] Boyd 2013; Proximal Algorithms §6.5.3

        """
        ridge_threshold: float = self._ridge_mult / self._penalty_term
        ridge_threshold /= self._n_clients

        shrinkage: float = 1 + self._step_size * self._ridge_mult

        return super()._z_update(z) / shrinkage
