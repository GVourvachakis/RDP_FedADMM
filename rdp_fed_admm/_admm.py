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
    penalty_term: float
    clipping_threshold: float
    cache_factorizations: bool
    dp_mechanism: Literal["rdp_gaussian"]
    dp_params: tuple[float, float]
    elastic_aggregation: bool  # New flag to enable elastic aggregation  

class ADMMClient(_ADMMBase, Client):
    def __init__(
        self,
        addr: str = "127.0.0.1",
        port: int = 50000,
        n_iter: int = 150,
        seed: int | None = None,
        step_size: float = 0.3,
        penalty_term: float = 10,
        clipping_threshold: float = 0.1,
        cache_factorizations: bool = True,
        dp_mechanism: Literal["rdp_gaussian"] = "rdp_gaussian",
        dp_params: tuple[float, float] = (1, 0.01),
        elastic_aggregation: bool = False
    ) -> None:
        _ADMMBase.__init__(self, seed, step_size, penalty_term)
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
        # New flag for elastic aggregation
        self.elastic_aggregation = elastic_aggregation

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
                log.info("SSCoeff: %.4g" % (self._coeffs @ self._coeffs.T))
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
            if self.elastic_aggregation:
                sensitivity = self._x_update_sensitivity()
                self.send_array(du)
                self.send_array(np.array([sensitivity]))
            else:
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
    elastic_aggregation: bool  # New flag to enable elastic aggregation on server
    sens_momentum: float       # EMA momentum for sensitivity
    tau: float                 # Hyperparameter tau for adaptive coefficient
    clip_min: float            # Clipping lower bound for zeta
    clip_max: float            # Clipping upper bound for zeta
    ema_type: Literal["per", "global"]  # Type of EMA to use

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
        elastic_aggregation: bool = False,
        sens_momentum: float = 0.9,
        tau: float = 0.1,
        clip_min: float = 0.8,
        clip_max: float = 1.2,
        ema_type: Literal["per", "global"] = "per",
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

        # New elastic aggregation attributes
        self.elastic_aggregation = elastic_aggregation
        self.sens_momentum = sens_momentum
        self.tau = tau
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.ema_type = ema_type
        self.sens_ema: FArr | None = None  # will be initialized in fit() when n_features is known
    
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

    def _z_update(self, du: FArr) -> FArr:
        raise NotImplementedError("This is meant to be overridden")

    def fit(
        self,
        n_features: int,
    ) -> Self:
        dim_weights = n_features
        du: FArr = np.zeros(dim_weights)
        z: FArr = np.zeros_like(du)
        self._coeffs = z.copy()

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
        # If using elastic aggregation, prepare lists to store updates and sensitivities.
        for iter in range(self._n_iter):
            subset = sample(conns, subs_size)
            updates_list = []   # list of (update, weight)
            sensitivities_list = []  # list of (sensitivity, weight)
            total_weight = 0.0

            for fd in subset:
                self.send_array(z, fd)

            while subset:

                rfds, _, efds = select(subset, [], subset)

                if efds:
                    raise RuntimeError

                for fd in rfds:
                    idx = conns.index(fd)
                    weight = n_data_client[idx] if self._wagg else 1.0
                    if self.elastic_aggregation:
                        update = self.recv_array(fd)
                        sens = self.recv_array(fd)  # sensitivity sent as a one‐element array
                        updates_list.append((update, weight))
                        sensitivities_list.append((sens, weight))
                    else:
                        u = self.recv_array(fd)
                        du += weight * u
                    subset.remove(fd)
                    # if self._keep_coeffs_full:
                    #     self._coeffs_full[idx + 1, :] = z
                    # if self._keep_coeff_hist:
                    #     self._coeff_hist[idx + 1, iter, :] = u

            if self.elastic_aggregation:
                # Compute weighted averages
                total_weight = sum(w for (_, w) in updates_list)
                weighted_update = sum(u * w for (u, w) in updates_list) / total_weight
                aggregated_sens = sum(s[0] * w for (s, w) in sensitivities_list) / total_weight
                # Initialize sens_ema on first iteration
                if self.sens_ema is None:
                    self.sens_ema = np.zeros_like(weighted_update)
                # Update EMA (per coordinate if ema_type=="per")
                if self.ema_type == "per":
                    self.sens_ema = self.sens_momentum * self.sens_ema + (1 - self.sens_momentum) * weighted_update * 0 + (1 - self.sens_momentum) * aggregated_sens  # note: here we use aggregated sensitivity broadcast to all coordinates
                else:
                    # For global, use a scalar EMA
                    self.sens_ema = self.sens_momentum * np.array([self.sens_ema]) + (1 - self.sens_momentum) * aggregated_sens
                max_ema = np.max(self.sens_ema) if np.max(self.sens_ema) > 0 else 1.0
                zeta = 1 + self.tau - (self.sens_ema / max_ema)
                zeta = np.clip(zeta, self.clip_min, self.clip_max)
                # Elastic update
                du = weighted_update * zeta
            else:
                du /= len(subset) if len(subset) > 0 else 1
            z = self._z_update(du)
            self._coeffs = z.copy()
            if self._keep_coeffs_full:
                self._coeffs_full[0, :] = z
            if self._keep_coeff_hist:
                self._coeff_hist[0, iter, :] = z
        log.info("SSCoeff: %.4g" % (self.coeffs @ self.coeffs.T))
        return self