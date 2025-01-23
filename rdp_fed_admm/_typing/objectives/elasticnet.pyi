from typing import Any, final

from .lasso import LassoADMMClient, LassoADMMServer

"""Elastic Net Objective.

The elastic net is a generalized scheme which linearly combines lasso
and ridge regularization as follows:
    R := l_1 || x ||₁ + l_2 || x ||₂²
where l_1 and l_2 are adjustable hyperparameters.

The way the optimization problem is formulated under ADMM, this only
applies to the soft-thresholding operator during the `z`-update.
"""

@final
class ElasticNetADMMClient(LassoADMMClient):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

@final
class ElasticNetADMMServer(LassoADMMServer):
    def __init__(
        self, *args: Any, ridge_multiplier: float = ..., **kwargs: Any
    ) -> None: ...
