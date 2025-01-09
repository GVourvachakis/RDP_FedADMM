#! /usr/bin/env python

"""Non i.i.d. data splitting for regression datasets.

Reads a csv file containing a dataset and a number of splits
and generates a series of subsets according to some
non-i.i.d-ness metric.
"""

import math
import pathlib
from argparse import ArgumentParser
from collections import defaultdict
from random import sample

import numpy as np
from numpy.typing import ArrayLike

parser = ArgumentParser(
    description="Split a dataset in a biased way",
    epilog=__doc__,
)

_ = parser.add_argument(
    "dataset",
    help="CSV File containing the dateset",
    type=str,
)

_ = parser.add_argument(
    "-t", "--target",
    help="Index of the target column in the dataset",
    type=int,
    required=True,
)

_ = parser.add_argument(
    "-s", "--splits",
    help="How many subsets to create from the original",
    type=int,
    default=2,
)

_ = parser.add_argument(
    "-b", "--bias",
    help="Bias factor (float between 0 and 1)",
    type=float,
    default=0.5,
)


def _create_dir(path: str) -> None:
    p = pathlib.Path(path)
    if not p.exists():
        p.mkdir()


if __name__ == "__main__":
    args = parser.parse_args()
    rng = np.random.default_rng(42)

    bias = args.bias
    assert 0 <= bias <= 1

    NBINS = 10
    NSPLITS = args.splits
    data = np.genfromtxt(args.dataset, delimiter=",", skip_header=1)
    assert 0 <= args.target <= data.shape[1] - 1
    Y = data[:,args.target]

    bins = np.linspace(Y.min(), Y.max(), NBINS+1)[1:]
    inds = bins.searchsorted(Y)

    type ArrayLike = np.ndarray[tuple[int, int], np.dtype[np.float64]]
    hmap: dict[str, list[ArrayLike]] = defaultdict(list[ArrayLike])
    for i, ind in enumerate(inds):
        hmap[ind].append(data[i])
    splits = defaultdict(list[np.array])

    K = max((1, math.floor(bias * len(hmap))))
    for idx_split in range(NSPLITS):
        sub_inds = sample(inds.tolist(), K)

        for ind in sub_inds:
            samples = hmap[ind]
            n = math.floor(len(samples) / NSPLITS / bias)

            for _ in range(n):
                sub_sample = hmap[ind].pop()
                splits[idx_split].append(sub_sample)

    _create_dir("./splits")
    for k, v in splits.items():
        v = np.array(v)
        hdr = f"No {k}: mean: {v.mean()}, std: {v.std()}"
        np.savetxt(f"./splits/{k}.csv", v, delimiter=",", header=hdr)
