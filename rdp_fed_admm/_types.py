from typing import Any, Callable, Literal

from numpy import dtype, float64, floating, ndarray
from numpy.typing import NDArray

type FArr = NDArray[floating[Any]]
type Float64 = dtype[float64]
type F64Arr = ndarray[tuple[int, ...], Float64]
type F64Vec = ndarray[tuple[int, Literal[1]], Float64]

type MLoss = Callable[[FArr, FArr], float]
type MDPNoise = Callable[[float, tuple[float, float]], float]
