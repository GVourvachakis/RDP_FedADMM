from typing import Self

from ._net import Client, Server
from ._types import FArr

"""Foundational ADMM Classes.

Contains implementations for a generic client and server containing
pluggable methods for the `x` and `z` updates respectively. These
modules don't work _per se_ but are intended to be overridden with
classes which implement the aforementioned methods.

This module focuses mostly on communication, messaging and aggregation
logic.
"""

class _ADMMBase:
    def __init__(
        self,
        seed: int | None = ...,
        step_size: float = ...,
        penalty_term: float = ...,
    ) -> None: ...
    def predict(self, X: FArr) -> FArr: ...

class ADMMClient(_ADMMBase, Client):
    def __init__(
        self,
        addr: str,
        port: int,
        n_iter: int,
        seed: int | None = ...,
        step_size: float = ...,
        penalty_term: float = ...,
        clipping_threshold: float = ...,
        cache_factorizations: bool = ...,
        dp_params: tuple[float, float] = ...,
        dp_mechanism: str = ...,
    ) -> None: ...
    def fit(self, X: FArr, Y: FArr) -> Self: ...

class ADMMServer(_ADMMBase, Server):
    def __init__(
        self,
        addr: str,
        port: int,
        max_clients: int,
        n_iter: int,
        seed: int | None = ...,
        step_size: float = ...,
        penalty_term: float = ...,
        subset_size: float = ...,
        coeffs_full: bool = ...,
        coeff_history: bool = ...,
    ) -> None: ...
    @property
    def coeffs(self) -> FArr: ...
    @property
    def coeff_hist(self) -> FArr: ...
    def fit(self, n_features: int) -> Self: ...
