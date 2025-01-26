from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    FileType,
    Namespace,
)
from collections.abc import Callable
from dataclasses import dataclass
from io import BufferedReader, FileIO
from logging import Logger
from typing import Literal, cast

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
    verbose: int
    n_iter: int
    dataset: BufferedReader
    tgt_idx: int
    client_id: int
    n_feat: int
    epsilon: float
    dp_mechanism: Literal["rdp_gaussian"]
    n_client: int
    log_file: str
    hist_file: FileIO | None
    coeffs_full: bool
    coeff_file: FileIO | None
    ridge_multiplier: float
    lasso_multiplier: float
    weighted_aggregation: bool
    func: Callable[..., None]


def run_client(args: Args, log: Logger):
    log.info(f"Registering client for {args.address}:{args.port}")
    data = np.genfromtxt(args.dataset, delimiter=",", skip_header=1)

    Y = data[:, args.tgt_idx]
    X = np.delete(data, args.tgt_idx, axis=1)

    with ElasticNetADMMClient(
        addr=args.address,
        port=args.port,
        n_iter=args.n_iter,
        dp_params=(1, args.epsilon),
        dp_mechanism=args.dp_mechanism,
    ) as cli:
        cli.fit(X, Y)


def run_server(args: Args, log: Logger):
    log.info(f"Registering server for {args.address}:{args.port}")
    with ElasticNetADMMServer(
        ridge_multiplier=args.ridge_multiplier,
        lasso_multiplier=args.lasso_multiplier,
        addr=args.address,
        port=args.port,
        max_clients=args.n_client,
        n_iter=args.n_iter,
        coeffs_full=args.coeffs_full,
        coeff_history=args.hist_file is not None,
        weighted_aggregation=args.weighted_aggregation,
    ) as srv:
        log.info(f"Listening on {args.address}:{args.port}")
        srv.fit(args.n_feat)

        if args.coeff_file is not None:
            fname = cast(str, args.coeff_file.name)
            arity = "full" if args.coeffs_full else "server"
            log.info(f"Saving {arity} coeffs to {fname}")

            # NOTE: atleast_2d forces savetxt to export coeffs
            # in one line making the file easier to manipulate
            coeffs = np.atleast_2d(srv.coeffs)
            np.savetxt(args.coeff_file, coeffs, delimiter=",")

        if args.hist_file is not None:
            fname = cast(str, args.hist_file.name)
            log.info(f"Saving coeff history to {fname}")
            np.save(args.hist_file, srv.coeff_hist)


parser = ArgumentParser(
    description="Rényi-DP Federated ADMM Optimization",
    epilog=__doc__,
    formatter_class=ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "-v", "--verbose",
    help="increase verbosity",
    action="count",
    default=0,
)

parser.add_argument(
    "-l", "--log-file",
    help="log file",
    type=str,
    default=None,
)

parser.add_argument(
    "-n", "--n-iter",
    help="number of optimization rounds before stopping",
    type=int,
    default=150,
)

_connection = parser.add_argument_group(description="connection")
_connection.add_argument(
    "-a", "--address",
    default="127.0.0.1",
    help="address to {listen,connect} to when running as a {server, client}",
)

_connection.add_argument(
    "-p", "--port",
    type=int,
    default=50000,
    help="port to {listen,connect} to when running as a {server, client}",
)

subparsers = parser.add_subparsers(required=True, help="instance type")

parser_cli = subparsers.add_parser(
    "client",
    help="ADMM Client",
    formatter_class=ArgumentDefaultsHelpFormatter
)
parser_cli.set_defaults(func=run_client)
parser_cli.add_argument(
    "dataset",
    help="file containing the dateset (*.csv.xz)",
    type=FileType('rb'),
)
parser_cli.add_argument(
    "-t", "--tgt-idx",
    help="index of the target column in the dataset",
    type=int,
)
parser_cli.add_argument(
    "-i", "--client-id",
    help="a number which separates one client from another",
    type=int,
    required=True,
)
parser_cli.add_argument(
    "-e", "--epsilon",
    help="differential privacy budget epsilon (float)",
    type=float,
    default=0.003,
)
parser_cli.add_argument(
    "--dp-mechanism",
    choices=["rdp_gaussian"],
    help="differential privacy mechanism",
    type=str,
    default="rdp_gaussian",
)

parser_srv = subparsers.add_parser(
    "server",
    help="ADMM Server",
    formatter_class=ArgumentDefaultsHelpFormatter,
)
parser_srv.set_defaults(func=run_server)
parser_srv.add_argument(
    "-c", "--n-client",
    help="number of client connections to expect",
    type=int,
    required=True,
)
parser_srv.add_argument(
    "-f", "--n-feat",
    help="number of features in the dataset",
    type=int,
    required=True,
)
parser_srv.add_argument(
    "--coeffs-full",
    help="Return the coefficients of the connected " +
        "clients as well as the server's.",
    action="store_true",
)
parser_srv.add_argument(
    "--coeff-file",
    help="coeff for server and clients",
    type=FileType("wb", 0),
    default=None,
)
parser_srv.add_argument(
    "--hist-file",
    help="coeff history file",
    type=FileType("wb", 0),
    default=None,
)
parser_srv.add_argument(
    "--lasso-multiplier",
    help="linear lasso (l1) coefficient for elastic net",
    type=float,
    default=1,
)
parser_srv.add_argument(
    "--ridge-multiplier",
    help="linear ridge (l2) coefficient for elastic net",
    type=float,
    default=1,
)
parser_srv.add_argument(
    "--weighted-aggregation",
    help="weigh client parameters by size of their dataset",
    action="store_true",
)
