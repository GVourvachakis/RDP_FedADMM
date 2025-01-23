from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from dataclasses import dataclass
from io import BufferedReader, FileIO
from logging import Logger

__all__ = ["Args", "parser"]

@dataclass
class Args(Namespace):
    address: str
    port: int
    verbose: bool
    n_iter: int
    dataset: BufferedReader
    tgt_idx: int
    client_id: int
    n_feat: int
    epsilon: float
    n_client: int
    log_file: str
    hist_file: FileIO | None
    coeffs_full: bool
    coeff_file: FileIO | None
    func: Callable[..., None]
    ...

def is_positive_float(val: str) -> float: ...
def run_client(args: Args, log: Logger) -> None: ...
def run_server(args: Args, log: Logger) -> None: ...

parser: ArgumentParser = ...
