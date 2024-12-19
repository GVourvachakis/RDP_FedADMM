from typing import final, override

import numpy as np

from ._types import *
from .admm import ADMMClient, ADMMServer


@final
class LassoADMMClient(ADMMClient):
    def __init__(
        self,
        addr: str,
        port: int,
        seed: int | None = None,
        step_size: float = 0.3,
        penalty_term: float = 0.6,
        clipping_threshold: float = 0.1,
    ) -> None:
        super().__init__(addr, port, seed, step_size, penalty_term)

    @override
    def _x_update(self, X: FArr, Y: FArr, x: FArr, z: FArr, u: FArr) -> FArr:
        assert X is not None and Y is not None
        rho = self._penalty_term

        # TODO: OPTIMIZE
        XtX: FArr = X.T @ X
        XtY: FArr = X.T @ Y
        inv: FArr = np.linalg.inv(XtX + rho * np.ones_like(XtX))

        return inv @ (XtY + rho * (z - u))


@final
class LassoADMMServer(ADMMServer):
    def __init__(
        self,
        addr: str,
        port: int,
        max_clients: int,
        seed: int | None = None,
        step_size: float = 0.3,
        penalty_term: float = 0.6,
        subset_size: float = 0.7,
    ) -> None:
        super().__init__(
            addr, port, max_clients,
            seed, step_size, penalty_term, subset_size
        )

    @staticmethod
    def soft_threshold(x: FArr, threshold: float) -> FArr:
        return np.sign(x) * np.max(np.abs(x) - threshold, 0)

    @override
    def _z_update(self, z: FArr):
        threshold: float = self._step_size / self._penalty_term
        threshold /= self._n_clients

        return self.soft_threshold(z, threshold)
