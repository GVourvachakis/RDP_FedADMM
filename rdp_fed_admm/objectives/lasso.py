"""Lasso Objective.

Lasso regularization promotes sparsity in the objective by adding the
l_1 norm as an extra term:
    R := l_1 || x ||₁
where l_1 is an adjustable hyperparameter.

The way the optimization problem is formulated under ADMM, the proximal
of this operation amounts to a soft-thresholding operator applied as the
`z`-update.
"""

from typing import Unpack, override

import numpy as np

from .._admm import ADMMClient, ADMMClientParams, ADMMServer, ADMMServerParams
from .._rdp import get_mechanism
from .._types import FArr

__all__ = [
    "LassoADMMClient",
    "LassoADMMClientParams",
    "LassoADMMServer",
    "LassoADMMServerParams",
]


class LassoADMMClientParams(ADMMClientParams):
    pass


class LassoADMMClient(ADMMClient):
    def __init__(self, **kwargs: Unpack[ADMMClientParams]) -> None:
        super().__init__(**kwargs)

    @override
    def _x_update_sensitivity(self) -> float:
        """Sensitivity of the ADMM x_update operator.

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

        Notes
        -----
        - The Lipschitz constant `L` is set here to 1.

        References
        ----------
        [1] Cyffers, Bellet and Basu 2023
        [2] Ilya Mironov, 2017

        """
        L = 1
        sens = 4 * self._step_size * L * self._penalty_term
        sens /= self._n_data

        return sens

    @override
    def _get_noise(
        self,
        sensitivity: float,
        size: tuple[int, ...] | int = 1,
    ) -> FArr:
        """Noise var for Gaussian RDP ADMM x update.

        Under privacy amplification from iteration and the advanced
        composition theorems, we only add for `K` iterations `σ²/K`
        noise in each one.

        Parameters
        ----------
        sensitivity : float
            The l-p sensitivity of the function to which the
            randomization mechanism will be applied. Whether `p` is 1
            or 2 depends also depends on the mechanism.
        size : tuple[int, ...] | int = 1
            The shape of the generated array. Passed to numpy's rng.

        Notes
        -----
        - Noting the advanced composition theorems associated with
          the RDP formulations, `K` iterations each involving a
          noise factor `N(0,σ²/K)` will add up to a `ε` RDP-budget.

        References
        ----------
        [1] Cyffers, Bellet and Basu 2023
        [2] Ilya Mironov, 2017

        """
        mech = get_mechanism(self._dp_mechanism)
        return mech(sensitivity, self._dp_params, size)

    @override
    def _x_update(self, X: FArr, Y: FArr, x: FArr, z: FArr, u: FArr) -> FArr:
        rho = self._penalty_term

        if self._cache_miss("lhs"):
            XtX: FArr = X.T @ X
            nxn: tuple[int, ...] = XtX.shape
            I = np.eye(N=nxn[0], M=nxn[1])
            lhs = np.linalg.inv(XtX + rho * I)
            self._cache["lhs"] = lhs

        if self._cache_miss("XtY"):
            self._cache["XtY"] = X.T @ Y

        return self._cache["lhs"] @ (self._cache["XtY"] + rho * (z - u))


class LassoADMMServerParams(ADMMServerParams):
    pass


class LassoADMMServer(ADMMServer):
    def __init__(self, **kwargs: Unpack[ADMMServerParams]) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def soft_threshold(X: FArr, thresh: float) -> FArr:
        return np.clip(X - thresh, 0, None) + np.clip(X + thresh, None, 0)

    @override
    def _z_update(self, z: FArr):
        threshold: float = self._step_size / self._penalty_term
        threshold /= self._n_clients

        return self.soft_threshold(z, threshold)
