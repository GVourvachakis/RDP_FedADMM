from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from dataclasses import dataclass

__all__ = [
    "Args",
    "parser",
]

@dataclass
class Args(Namespace):
    client: bool = False
    server: bool = False
    address: str = "127.0.0.1"
    port: int = 50000
    verbose: bool = False


parser = ArgumentParser(
    description="Rényi-DP Federated ADMM Optimization",
    epilog=__doc__,
    formatter_class=RawDescriptionHelpFormatter,
)

_ = parser.add_argument(
    "-v", "--verbose",
    help="Increase verbosity",
    action="store_true",
)

_ = parser.add_argument(
    "-l", "--log-file",
    help="Log file",
    type=str,
    default=None,
)

_ = parser.add_argument(
    "-d", "--dataset",
    help="CSV File containing the dateset",
    type=str,
)

_ = parser.add_argument(
    "-t", "--target",
    help="Index of the target column in the dataset",
    type=int,
)

_instance_type = parser.add_mutually_exclusive_group(required=True)
_ = _instance_type.add_argument(
    "-c", "--client",
    help="Run as a client instance",
    action="store_true",
)
_ = _instance_type.add_argument(
    "-s", "--server",
    help="Run as a server instance",
    action="store_true",
)

_connection = parser.add_argument_group(description="connection")
_ = _connection.add_argument(
    "-a", "--address",
    default="127.0.0.1",
    help="Address to {listen,connect} to when running as a {server, client}",
)

_ = _connection.add_argument(
    "-p", "--port",
    type=int,
    default=50000,
    help="Port to {listen,connect} to when running as a {server, client}",
)
