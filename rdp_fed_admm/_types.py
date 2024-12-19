from typing import Any, Literal

from numpy import dtype, floating, ndarray
from numpy.typing import NDArray

type FArr = NDArray[floating[Any]]
type Float = floating[Any] | float
type NpFloat = dtype[floating[Any]]
type Vec = ndarray[tuple[int, Literal[1]], NpFloat]
