from typing import final, override

import numpy as np

from .._types import *
from ..admm import ADMMClient, ADMMServer


class LassoADMMClient(ADMMClient):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @override
    def _x_update(self, X: FArr, Y: FArr, x: FArr, z: FArr, u: FArr) -> FArr:
        rho = self._penalty_term

        if self._cache_miss("lhs"):
            XtX: FArr = X.T @ X
            lhs = np.linalg.inv(XtX + rho * np.eye(*XtX.shape))
            self._cache["lhs"] = lhs

        if self._cache_miss("XtY"):
            self._cache["XtY"] = X.T @ Y

        # print("lhs: ", self._cache["lhs"].mean(),
        #       "XtY: ", self._cache["XtY"].mean(),
        #       "PRP: ", rho * (z - u).mean())

        return self._cache["lhs"] @ (self._cache["XtY"] + rho * (z - u))


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
    def soft_threshold(X: FArr, thresh: float) -> FArr:
        return np.clip(X - thresh, 0, None) + np.clip(X + thresh, None, 0)

    @override
    def _z_update(self, z: FArr):
        threshold: float = self._step_size / self._penalty_term
        threshold /= self._n_clients

        return self.soft_threshold(z, threshold)
