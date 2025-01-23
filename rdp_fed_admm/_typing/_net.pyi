from collections.abc import Callable
from contextlib import ContextDecorator
from logging import Logger
from socket import SocketType
from types import TracebackType
from typing import Any, Self, override

from ._types import FArr

__all__ = ["Client", "Server"]
log: Logger = ...

class _NetworkBase(ContextDecorator):
    SIZLEN: int = ...
    RDSTEP: int = ...
    def __init__(
        self, addr: str, port: int, sock: Callable[[Any], SocketType]
    ) -> None: ...
    def __enter__(self) -> Self: ...
    def send_array(self, arr: FArr, conn: SocketType | None = ...) -> None: ...
    def recv_array(self, conn: SocketType | None = ...) -> FArr: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool: ...

class Client(_NetworkBase):
    def __init__(self, addr: str, port: int) -> None: ...

class Server(_NetworkBase):
    def __init__(
        self, addr: str, port: int, max_clients: int = ...
    ) -> None: ...
    def activate_server(self) -> None:
        ...
    @override
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool: ...
