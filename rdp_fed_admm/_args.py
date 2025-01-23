from argparse import (
    ArgumentParser,
    FileType,
    Namespace,
    RawDescriptionHelpFormatter,
)
from collections.abc import Callable
from dataclasses import dataclass
from io import BufferedReader, FileIO
from logging import Logger
from typing import cast

import numpy as np

from .objectives.elasticnet import ElasticNetADMMClient, ElasticNetADMMServer

__all__ = [
    "Args",
    "parser",
]


@dataclass
class Args(Namespace):
    address: str
    port: int
    n_iter: int
    dataset: BufferedReader
    tgt_idx: int
    client_id: int
    n_feat: int
    verbose: bool
    log_file: str
    hist_file: FileIO | None
    coeffs_full: bool
    coeff_file: FileIO | None
    func: Callable[..., None]


def run_client(args: Args, log: Logger):
    log.info(f"Registering client for {args.address}:{args.port}")
    data = np.genfromtxt(args.dataset, delimiter=",", skip_header=1)

    Y = data[:, args.tgt_idx]
    X = np.delete(data, args.tgt_idx, axis=1)

    with ElasticNetADMMClient(
        args.address,
        args.port,
        n_iter=args.n_iter,
    ) as cli:
        _ = cli.fit(X, Y)


def run_server(args: Args, log: Logger):
    log.info(f"Registering server for {args.address}:{args.port}")
    with ElasticNetADMMServer(
        args.address,
        args.port,
        max_clients=2,
        n_iter=args.n_iter,
        coeffs_full=args.coeffs_full,
        coeff_history=args.hist_file is not None,
    ) as srv:
        log.info(f"Listening on {args.address}:{args.port}")
        _ = srv.fit(args.n_feat)

        if args.coeff_file is not None:
            fname = cast(str, args.coeff_file.name)
            arity = "full" if args.coeffs_full else "server"
            log.info(f"Saving {arity} coeffs to {fname}")
            ary = srv.coeffs
            hdr = f"Coeffs: {ary.shape} m: {ary.mean():.3f}, s: {ary.std():.3f}"
            np.savetxt(args.coeff_file, ary, delimiter=",", header=hdr)

        if args.hist_file is not None:
            fname = cast(str, args.hist_file.name)
            log.info(f"Saving coeff history to {fname}")
            np.save(args.hist_file, srv.coeff_hist)


parser = ArgumentParser(
    description="Rényi-DP Federated ADMM Optimization",
    epilog=__doc__,
    formatter_class=RawDescriptionHelpFormatter,
)

_ = parser.add_argument(
    "-v", "--verbose",
    help="increase verbosity",
    action="store_true",
)

_ = parser.add_argument(
    "-l", "--log-file",
    help="log file",
    type=str,
    default=None,
)

_ = parser.add_argument(
    "-n", "--n-iter",
    help="number of optimization rounds before stopping",
    type=int,
    default=25,
)

_connection = parser.add_argument_group(description="connection")
_ = _connection.add_argument(
    "-a", "--address",
    default="127.0.0.1",
    help="address to {listen,connect} to when running as a {server, client}",
)

_ = _connection.add_argument(
    "-p", "--port",
    type=int,
    default=50000,
    help="port to {listen,connect} to when running as a {server, client}",
)

subparsers = parser.add_subparsers(required=True, help="instance type")

parser_cli = subparsers.add_parser("client", help="ADMM Client")
parser_cli.set_defaults(func=run_client)
_ = parser_cli.add_argument(
    "dataset",
    help="file containing the dateset (*.csv.xz)",
    type=FileType('rb'),
)
_ = parser_cli.add_argument(
    "-t", "--tgt-idx",
    help="index of the target column in the dataset",
    type=int,
)
_ = parser_cli.add_argument(
    "-i", "--client-id",
    help="a number which separates one client from another",
    type=int,
    required=True,
)

parser_srv = subparsers.add_parser("server", help="ADMM Server")
parser_srv.set_defaults(func=run_server)
_ = parser_srv.add_argument(
    "-f", "--n-feat",
    help="number of features in the dataset",
    type=int,
    default=None,
)
_ = parser_srv.add_argument(
    "--coeffs-full",
    help="Return the coefficients of the connected" +
        "clients as well as the server's.",
    type=bool,
    default=False
)
_ = parser_srv.add_argument(
    "--coeff-file",
    help="coeff for server and clients (default: disabled)",
    type=FileType("wb", 0),
    default=None,
)
_ = parser_srv.add_argument(
    "--hist-file",
    help="coeff history file (default: disabled)",
    type=FileType("wb", 0),
    default=None,
)
