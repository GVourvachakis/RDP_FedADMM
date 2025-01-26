"""Foundational ADMM Classes.

Contains implementations for a generic client and server containing
pluggable methods for the `x` and `z` updates respectively. These
modules don't work _per se_ but are intended to be overridden with
classes which implement the aforementioned methods.

This module focuses mostly on communication, messaging and aggregation
logic.
"""

from logging import getLogger
from random import sample
from select import select
from socket import socket
from typing import Literal, Required, Self, TypedDict

import numpy as np
from numpy.random import Generator

from ._net import Client, Server
from ._types import FArr

log = getLogger(__name__)

__all__ = [
    "ADMMClient",
    "ADMMClientParams",
    "ADMMServer",
    "ADMMServerParams",
]


class _ADMMBase:
    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        seed: int | None = None,
        step_size: float = 0.3,
        penalty_term: float = 10,
    ) -> None:
        self.rng: Generator = np.random.default_rng(seed)
        self._step_size: float = step_size
        self._penalty_term: float = penalty_term
        self._coeffs: FArr | None

    def predict(self, X: FArr) -> FArr:
        if self._coeffs is None:
            raise RuntimeError("Not fitted")

        return X @ self._coeffs


class ADMMClientParams(TypedDict, total=False):
    addr: str
    port: int
    n_iter: int
    seed: int | None
    step_size: float
    clipping_threshold: float
    cache_factorizations: bool
    dp_mechanism: Literal["rdp_gaussian"]
    dp_params: tuple[float, float]


class ADMMClient(_ADMMBase, Client):
    def __init__(
        self,
        addr: str = "127.0.0.1",
        port: int = 50000,
        n_iter: int = 150,
        seed: int | None = None,
        step_size: float = 0.3,
        clipping_threshold: float = 0.1,
        cache_factorizations: bool = True,
        dp_mechanism: Literal["rdp_gaussian"] = "rdp_gaussian",
        dp_params: tuple[float, float] = (1, 0.01),
    ) -> None:
        _ADMMBase.__init__(self, seed, step_size)
        Client.__init__(self, addr, port)

        self._n_iter: int = n_iter
        self._n_data: int

        self._step: float = step_size
        self._clip_thresh: float = clipping_threshold
        self._X: FArr | None = None
        self._Y: FArr | None = None
        self._use_cache: bool = cache_factorizations
        self._cache: dict[str, FArr] = {}
        self._coeffs: FArr | None

        self._do_dp: bool
        self._dp_mechanism: str = dp_mechanism
        if dp_params[1] > 0:
            self._do_dp = True
            self._dp_params: tuple[float, float] = dp_params
        else:
            self._do_dp = False
            log.warning(f"DP Disabled: Received {dp_params[1]} <= 0.")

    @staticmethod
    def _clip(v: FArr, thresh: float) -> FArr:
        scale: float = np.min(np.array([thresh, np.linalg.norm(v)]))
        return v * scale

    def _x_update_sensitivity(self) -> float:
        raise NotImplementedError("This is meant to be overridden")

    def _get_noise(
        self,
        _sensitivity: float,
        _size: tuple[int, ...] | int = 1
    ) -> FArr:
        raise NotImplementedError("This is meant to be overridden")

    def _cache_miss(self, key: str) -> bool:
        return key not in self._cache or not self._use_cache

    def _x_update(
        self, _X: FArr, _Y: FArr, _z: FArr, _u: FArr,
    ) -> FArr:
        raise NotImplementedError("This is meant to be overridden")

    def fit(
        self,
        X: FArr,
        Y: FArr,
    ) -> Self:
        assert X.shape[0] == len(Y)
        self._n_data = X.shape[0]
        dim_weights: int = X.shape[1]
        x: FArr = np.zeros(dim_weights)
        z: FArr = np.zeros_like(x)
        u: FArr = np.zeros_like(x)
        du: FArr = np.zeros_like(u)

        if self._do_dp:
            n_data_noise = self._get_noise(1 / self._n_data)
        else:
            n_data_noise = 0

        self.send_array(np.array([self._n_data + n_data_noise]))

        for _ in range(self._n_iter):
            log.debug("Waiting for z")

            try:
                z = self.recv_array()
            except ConnectionResetError:
                self._coeffs = x
                log.info(self._coeffs)
                raise

            x = self._x_update(X, Y, z, u)
            # rsd = self._clip(x - z, self._clip_thresh)  # MAE triples
            rsd = x - z

            if self._do_dp:
                x_upd_noise = self._get_noise(
                    self._x_update_sensitivity(),
                    dim_weights,
                ) / 2
            else:
                x_upd_noise = 0

            du = 2 * self._step * (rsd + x_upd_noise)

            self.send_array(du)
            u += du

        return self


class ADMMServerParams(TypedDict, total=False):
    max_clients: Required[int]
    addr: str
    port: int
    n_iter: int
    seed: int | None
    step_size: float
    subset_size: float
    coeffs_full: bool
    coeff_history: bool
    weighted_aggregation: bool


class ADMMServer(_ADMMBase, Server):
    def __init__(
        self,
        max_clients: int,
        addr: str = "127.0.0.1",
        port: int = 50000,
        n_iter: int = 150,
        seed: int | None = None,
        step_size: float = 0.3,
        subset_size: float = 0.7,
        coeffs_full: bool = False,
        coeff_history: bool = False,
        weighted_aggregation: bool = False,
    ) -> None:
        _ADMMBase.__init__(self, seed, step_size)
        Server.__init__(self, addr, port, max_clients)
        self._n_iter: int = n_iter
        self._subset_size: float = subset_size
        self._n_clients: int

        self._coeffs: FArr | None
        self._keep_coeffs_full: bool = coeffs_full
        self._coeffs_full: FArr
        self._keep_coeff_hist: bool = coeff_history
        self._coeff_hist: FArr

        self._wagg: bool = weighted_aggregation

    @property
    def coeffs(self) -> FArr:
        if self._keep_coeffs_full:
            return self._coeffs_full
        if self._coeffs is None:
            raise RuntimeError("Not fitted")
        return self._coeffs

    @property
    def coeff_hist(self) -> FArr:
        if not self._keep_coeff_hist:
            raise RuntimeError("History has been disabled")
        return self._coeff_hist

    def _z_update(self, _z: FArr) -> FArr:
        raise NotImplementedError("This is meant to be overridden")

    def fit(
        self,
        n_features: int,
    ) -> Self:
        dim_weights = n_features
        du: FArr = np.zeros(dim_weights)
        z: FArr = np.zeros_like(du)

        self.activate_server()
        self._n_clients = len(self._client_conn)

        if self._keep_coeffs_full:
            self._coeffs_full = np.zeros((
                self._n_clients + 1, n_features
            ))
        if self._keep_coeff_hist:
            self._coeff_hist = np.zeros((
                self._n_clients + 1, self._n_iter, n_features
            ))

        conns = list(self._client_conn.values())
        n_data_client: list[float] = [self.recv_array(fd)[0] for fd in conns]
        n_data_client = [i / max(n_data_client) for i in n_data_client]

        subs_size = max(1, int(self._n_clients * self._subset_size))

        rfds: list[socket]
        for iter in range(self._n_iter):
            subset = sample(conns, subs_size)
            nsubs = len(subset)

            for fd in subset:
                self.send_array(z, fd)

            while subset:

                rfds, _, efds = select(subset, [], subset)

                if efds:
                    raise RuntimeError

                for fd in rfds:
                    idx = conns.index(fd)
                    u = self.recv_array(fd)

                    if self._keep_coeffs_full:
                        self._coeffs_full[idx + 1, :] = z
                    if self._keep_coeff_hist:
                        self._coeff_hist[idx + 1, iter, :] = u

                    if self._wagg:
                        du += n_data_client[idx] * u
                    else:
                        du += u

                    subset.remove(fd)

            du /= nsubs
            z = self._z_update(du)

            if self._keep_coeffs_full:
                self._coeffs_full[0, :] = z
            if self._keep_coeff_hist:
                self._coeff_hist[0, iter, :] = z

        self._coeffs = z
        log.info(self.coeffs)

        return self
