from logging import info
from random import sample
from select import select
from typing import Self, cast, override

import numpy as np
from numpy.random import Generator

from ._loss import get_loss
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
        cache_factorizations: bool = True,
        loss: str | None = None,
    ) -> None:
        _ADMMBase.__init__(self, seed, step_size, penalty_term)
        Client.__init__(self, addr, port)

        self._coeffs: FArr | None
        self._step: float = step_size
        self._clip_thresh: float = clipping_threshold
        self._X: FArr | None = None
        self._Y: FArr | None = None
        self._use_cache: bool = cache_factorizations
        self._cache: dict[str, FArr] = {}

        self._loss_func: MLoss | None = None
        self._loss: list[float] = []
        if loss is not None:
            self._loss_func = get_loss(loss)

    @property
    def training_loss(self) -> FArr:
        return np.array(self._loss)

    @staticmethod
    def _clip(v: FArr, thresh: Float) -> FArr:
        scale: Float = np.min(np.array([thresh, np.linalg.norm(v)]))
        return v * scale

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
        dim_weights: int = X.shape[1]
        x: FArr = np.zeros(dim_weights)
        z: FArr = np.zeros_like(x)
        u: FArr = np.zeros_like(x)
        du: FArr = np.zeros_like(u)

        for _ in range(n_iter):
            info("Waiting for z")

            try:
                z = self.recv_array()
            except ConnectionResetError:
                raise

            x = self._x_update(X, Y, 2 * z - u, u, z)
            # rsd = self._clip(x - z, self._clip_thresh)
            rsd = x - z
            # noise = 0.5 * self.rng.random(dim_weights)  # TODO: FIX
            noise = 0
            du = 2 * self._step * (rsd + noise)

            self.send_array(du)

            u += du
            self._coeffs = x

            self._coeffs = x
            print(f"{Y.mean():.2e}, {(X @ x).mean():.2e}, {x.mean():.2e}, {z.mean():.2e}, {u.mean():.2e}")
            if self._loss_func is not None:
                preds = self.predict(X)
                self._loss.append(self._loss_func(Y, preds))

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
        dim_weights = n_features
        du: FArr = np.zeros(dim_weights)
        z: FArr = np.zeros_like(du)

        self.activate_server()
        self._n_clients = len(self._client_conn)

        conns = list(self._client_conn.values())

        subs_size = max(1, int(self._n_clients * self._subset_size))
        for i in range(n_iter):
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
                    du += self.recv_array(fd)
                    subset.remove(fd)

            du /= nsubs
            z = self._z_update(du)

        self._coeffs = z

        return self
