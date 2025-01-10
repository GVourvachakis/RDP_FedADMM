import logging
from typing import cast

import numpy as np

from ._parser import Args, parser
from .objectives.elasticnet import LassoADMMClient, LassoADMMServer


def rdp_fed_admm():
    args = Args()
    args = parser.parse_args(namespace=args)

    logger = logging.getLogger(__name__)

    fdata: str = args.dataset
    n_features: int = args.features
    target_col: int = args.target
    n_iter: int = args.n_iter

    data = np.genfromtxt(fdata, delimiter=",", skip_header=1)
    Y = data[:, target_col]
    X = np.delete(data, target_col, axis=1)

    logging.basicConfig(format="[%(levelname)s] %(asctime)s %(message)s",
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO if args.verbose else logging.WARNING,
                        filename=cast(str | None,  args.log_file))

    if args.client:
        logger.info(f"Registering client for {args.address}:{args.port}")
        with LassoADMMClient(args.address, args.port) as cli:
            _ = cli.fit(X, Y, n_iter)
    elif args.server:
        logger.info(f"Registering server for {args.address}:{args.port}")
        with LassoADMMServer(args.address, args.port, max_clients=1) as srv:
            logger.info(f"Listening on {args.address}:{args.port}")
            preds = srv.fit(n_features, n_iter).predict(X)
            logger.warning("MAE: {:.3f}".format(np.abs(preds - Y).mean()))
    else:
        raise RuntimeError("Undefined behaviour")


if __name__ == "__main__":
    rdp_fed_admm()
