import io
import socket
from collections.abc import Callable
from contextlib import ContextDecorator
from logging import getLogger
from socket import SocketType
from types import TracebackType
from typing import cast, override

import numpy as np

from ._types import *

__all__ = [
    "Client",
    "Server",
]

log = getLogger(__name__)

class _NetworkBase(ContextDecorator):
    SIZLEN: int = 4
    RDSTEP: int = 4096

    def __init__(
        self,
        addr: str,
        port: int,
        sock: Callable[[Any], SocketType],
    ) -> None:
        super().__init__()
        self._socket: SocketType = sock((addr, port))


    def __enter__(self):
        return self

    def send_array(self, arr: FArr, conn: SocketType | None = None) -> None:
        conn = self._socket if not conn else conn
        stream: io.BytesIO = io.BytesIO()
        data: bytes = b""
        data_size: int = 0

        np.save(stream, arr)
        assert stream.seek(0) == 0
        data = stream.read()
        data_size = len(data)
        data = data_size.to_bytes(self.SIZLEN, signed=False) + data

        if not data_size:
            return

        log.debug(" ".join((
            f"Sending {cast(tuple[int,...], arr.shape)} array",
            f"to {conn.getpeername()}:",
            f"({data_size} bytes)",
        )))
        conn.sendall(data)

        stream.flush()
        stream.close()

    def recv_array(self, conn: SocketType | None = None) -> FArr:
        conn = self._socket if not conn else conn
        stream: io.BytesIO = io.BytesIO()
        data_size: int = 0
        bytes_written: int = 0

        # buf is empty when the sockets is disconnected
        buf = conn.recv(self.RDSTEP).strip()
        if not buf:
            raise ConnectionResetError("Peer closed connection")

        # strip header and parse data size, then get data
        # until data_size bytes have been written
        data_size = int.from_bytes(buf[:self.SIZLEN])
        buf = buf[self.SIZLEN:]
        bytes_written += stream.write(buf)
        while bytes_written < data_size:
            buf = conn.recv(self.RDSTEP).strip()
            bytes_written += stream.write(buf)

        assert stream.seek(0) == 0
        arr: FArr = np.load(stream)
        log.debug(" ".join((
            f"Received {cast(tuple[int,...], arr.shape)} array",
            f"from {conn.getpeername()}:",
            f"({data_size} bytes)",
        )))

        return arr

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        _msg: str = f"{self.__class__.__name__} shutting down"
        ret: bool = False

        if exc_type and exc_value:
            if exc_type is KeyboardInterrupt:
                _msg += ": Keyboard Interrupt"
                ret = True
            if exc_type is ConnectionResetError:
                _msg += ": " + ', '.join(exc_value.args)
                ret = True

        log.warning(_msg)
        self._socket.shutdown(socket.SHUT_RDWR)
        self._socket.close()

        return ret


class Client(_NetworkBase):
    def __init__(
        self,
        addr: str,
        port: int,
    ):
        log.info(f"Registered client for {addr}:{port}")
        super().__init__(addr, port, socket.create_connection)
        log.info(f"Listening on %s:%d" % self._socket.getsockname())


class Server(_NetworkBase):
    def __init__(
        self,
        addr: str,
        port: int,
        max_clients: int = 5,
    ) -> None:
        log.info(f"Registered server for {addr}:{port} (max {max_clients} clients)")
        super().__init__(addr, port, socket.create_server)
        self._client_conn: dict[Any, Any] = {}
        self._max_clients: int = max_clients

    def activate_server(self):
        self._socket.listen(self._max_clients)
        while len(self._client_conn) < self._max_clients:
            conn, addr = self._socket.accept()
            log.info(f"Accepted connection from {addr[0]}:{addr[1]}")
            self._client_conn[addr] = conn

    @override
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        ret = super().__exit__(exc_type, exc_value, traceback)

        for peer, conn in self._client_conn.items():
            log.info("Closing %s:%d", *peer)
            conn.shutdown(socket.SHUT_RDWR)
            conn.close()

        return ret
