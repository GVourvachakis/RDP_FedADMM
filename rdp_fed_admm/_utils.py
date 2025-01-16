"""Helper functions and tools."""

from copy import deepcopy
from logging import getLogger

log = getLogger(__name__)

__all__ = [
    "call_gettr"
]

def call_gettr[T](key: str, hmap: dict[str, T]) -> T:
    try:
        method = deepcopy(hmap[key])
    except KeyError as e:
        log.error(f"{e} invalid")
        raise ValueError(
            f"{e} is invalid, please pick one of" +
            ", ".join(hmap.keys())
        )

    return method
