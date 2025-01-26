import logging
from typing import cast

from ._args import Args, parser


def rdp_fed_admm() -> None:
    args = cast(Args, parser.parse_args())
    log = logging.getLogger(__name__)

    logging.basicConfig(format="[%(levelname)s] %(asctime)s %(message)s",
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=min(10, 30 - 10 * args.verbose),
                        filename=cast(str | None, args.log_file))

    args.func(args, log)


if __name__ == "__main__":
    rdp_fed_admm()
