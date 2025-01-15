from typing import override

import numpy as np

from .._types import *
from ..admm import ADMMClient, ADMMServer


class LassoADMMClient(ADMMClient):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @override
    def _get_noise_scale(
        self,
        n_data: int,
        params: tuple[float, float],
    ) -> float:
        """Noise var for Gaussian RDP ADMM x update.

        The sensitivity of the `x_update` LSQ minimizer `f` which:
            - Is `L`-Lipschitz
            - Has a step size `λ`
            - Has a proximal penalty term `ρ`
            - Is applied to data X with `|X| = n`
        is found to be [1; §C.2]:
            Δf = 4λLρ/n

        The randomized algorithm `A` which adds iterative Gaussian
        noise `N` to `f`:
            A : f(x) + N(0,σ²)
        is then said to be (α,ε)-RDP iff the scale of the added
        noise satisfies [2]:
            σ² = α Δ²f / (2ε)²
        Under privacy amplification from iteration and the advanced
        composition theorems, we only add for `K` iterations `σ²/K`
        noise in each one.

        Parameters
        ----------
        n_data : int
            The number of data points/examples in the client's
            dataset.
        params : tuple[float, float]
            The RDP (α,ε) parameters.

        Notes
        -----
        - The Lipschitz constant `L` is set here to 1.
        - Noting the advanced composition theorems associated with
          the RDP formulations, `K` iterations each involving a
          noise factor `N(0,σ²)` will add up to a `Kε` RDP-budget.

        References
        ----------
        [1] Cyffers, Bellet and Basu 2023
        [2] Ilya Mironov, 2017

        """
        L = 1
        sensitivity = 4 * self._step_size * L * self._penalty_term
        sensitivity /= n_data

        alpha, epsilon = params

        scale = alpha / (2 * epsilon)**2
        scale *= sensitivity**2

        return scale / self._n_iter

    @override
    def _x_update(self, X: FArr, Y: FArr, x: FArr, z: FArr, u: FArr) -> FArr:
        rho = self._penalty_term

        if self._cache_miss("lhs"):
            XtX: FArr = X.T @ X
            lhs = np.linalg.inv(XtX + rho * np.eye(*XtX.shape))
            self._cache["lhs"] = lhs

        if self._cache_miss("XtY"):
            self._cache["XtY"] = X.T @ Y

        return self._cache["lhs"] @ (self._cache["XtY"] + rho * (z - u))


class LassoADMMServer(ADMMServer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @staticmethod
    def soft_threshold(X: FArr, thresh: float) -> FArr:
        return np.clip(X - thresh, 0, None) + np.clip(X + thresh, None, 0)

    @override
    def _z_update(self, z: FArr):
        threshold: float = self._step_size / self._penalty_term
        threshold /= self._n_clients

        return self.soft_threshold(z, threshold)
