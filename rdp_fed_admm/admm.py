from logging import info
from random import sample
from select import select
from typing import Self, cast, override

import numpy as np
from numpy.random import Generator

from ._net import *
from ._types import *


class _ADMMBase:
    def __init__(
        self,
        seed: int | None = None,
        step_size: float = 0.3,
        penalty_term: float = 0.6,
    ) -> None:
        self.rng: Generator = np.random.default_rng(seed)
        self._step_size: float = step_size
        self._penalty_term: float = penalty_term
        self._coeffs: FArr | None = None

    def predict(self, X: FArr) -> FArr:
        if self._coeffs is None:
            raise RuntimeError("Not fitted")

        return X @ self._coeffs


class ADMMClient(_ADMMBase, Client):
    def __init__(
        self,
        addr: str,
        port: int,
        seed: int | None = None,
        step_size: float = 0.3,
        penalty_term: float = 0.6,
        clipping_threshold: float = 0.1,
        sensitivity_momentum: float = 0.93,
        cache_factorizations: bool = True,
    ) -> None:
        _ADMMBase.__init__(self, seed, step_size, penalty_term)
        Client.__init__(self, addr, port)

        self._coeffs: FArr | None
        self._step: float = step_size
        self._clip_thresh: float = clipping_threshold
        self._momentum: float = sensitivity_momentum
        self._use_cache: bool = cache_factorizations
        self._cache: dict[str, FArr] = {}

    @staticmethod
    def _clip(v: FArr, thresh: Float) -> FArr:
        scale: Float = np.min(np.array([thresh, np.linalg.norm(v)]))
        return v * scale

    @staticmethod
    def _edma(x: FArr, s: FArr, alpha: float) -> FArr:
        """Exponentially decaying moving average.

        Returns s' := αs + (1-α)x
        """
        return alpha * s + (1-alpha) * x

    def _cache_miss(self, key: str) -> bool:
        return key not in self._cache or not self._use_cache

    def _x_update(self, X: FArr, Y: Vec, x: FArr, z: FArr, u: FArr) -> FArr:
        raise NotImplementedError("This is meant to be overridden")

    def fit(
        self,
        X: FArr,
        Y: Vec,
        n_iter: int = 100
    ) -> Self:
        n_features = X.shape[1]
        x: FArr = np.zeros(n_features)
        z: FArr = np.zeros_like(x)
        u: FArr = np.zeros_like(x)
        du: FArr = np.zeros_like(u)
        sensitivity: FArr = np.zeros_like(x)

        # local dataset size: used for weighting
        self.send_array(np.array(X.shape[0]))

        for _ in range(n_iter):
            info("Waiting for z")

            try:
                z = self.recv_array()
            except ConnectionResetError:
                raise

            x = self._x_update(X, Y, 2 * z - u, u, z)
            rsd = self._clip(x - z, self._clip_thresh)
            sensitivity = self._edma(np.abs(rsd), sensitivity, self._momentum)
            noise = 0.5 * self.rng.random(X.shape[1])
            noise = cast(FArr, noise)
            du = 2 * self._step * (rsd + noise)

            self.send_array(sensitivity)
            self.send_array(du)

            u += du

        self._coeffs = x

        return self


class ADMMServer(_ADMMBase, Server):
    def __init__(
        self,
        addr: str,
        port: int,
        max_clients: int,
        seed: int | None = None,
        step_size: float = 0.3,
        penalty_term: float = 0.6,
        subset_size: float = 0.7,
        elastic_boost: float = 0.5,
    ) -> None:
        _ADMMBase.__init__(self, seed, step_size, penalty_term)
        Server.__init__(self, addr, port, max_clients)
        self._subset_size: float = subset_size
        self._n_clients: int
        self._coeffs: FArr | None
        self._boost: float = elastic_boost

    def _z_update(self, z: FArr) -> FArr:
        raise NotImplementedError("This is meant to be overridden")

    def fit(
        self,
        n_features: int,
        n_iter: int = 100,
    ) -> Self:
        du: FArr = np.zeros(n_features)
        z: FArr = np.zeros_like(du)

        self.activate_server()
        self._n_clients = len(self._client_conn)

        senses = np.zeros((self._n_clients, n_features))

        info("Collecting dataset sizes from clients")
        data_weights = []
        for client in self._client_conn.values():
            data_weights.append(self.recv_array(client))
        data_weights = np.array(data_weights).astype(float)
        data_weights /= data_weights.sum()
        data_weights = cast(FArr, data_weights)

        conns = list(self._client_conn.values())

        subs_size = max(1, int(self._n_clients * self._subset_size))
        for i in range(n_iter):
            subset = sample(conns, subs_size)
            # nsubs = len(subset)

            for fd in subset:
                self.send_array(z, fd)

            while subset:
                rfds, _, efds = select(subset, [], subset)

                if efds:
                    raise RuntimeError

                for fd in rfds:
                    idx = conns.index(fd)
                    weight = data_weights[idx]
                    senses[idx,:] = self.recv_array(fd)
                    senses[idx,:] *= weight
                    du += weight * self.recv_array(fd)
                    subset.remove(fd)

            agg_senses = senses.sum(axis=0)  # agg over clients
            agg_senses /= agg_senses.max()   # normalize
            elastic_weights = 1 + self._boost - agg_senses
            du *= elastic_weights
            # du /= nsubs
            z = self._z_update(du)

        self._coeffs = z

        return self
