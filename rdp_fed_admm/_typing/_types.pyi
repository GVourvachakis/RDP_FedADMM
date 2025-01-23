from typing import Any, Callable, Literal

from numpy import dtype, float64, floating, ndarray
from numpy.typing import NDArray

FArr = NDArray[floating[Any]]
Float64 = dtype[float64]
F64Arr = ndarray[tuple[int, ...], Float64]
F64Vec = ndarray[tuple[int, Literal[1]], Float64]
MLoss = Callable[[FArr, FArr], float]
MDPNoise = Callable[[float, tuple[float, float]], float]
