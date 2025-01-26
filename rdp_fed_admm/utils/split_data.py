#! /usr/bin/env python

"""Non i.i.d. data splitting for regression datasets.

Reads a csv file containing a dataset and a number of splits
and generates a series of subsets according to some
non-i.i.d-ness metric.
"""

import math
import random
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

type F64Arr = np.ndarray[tuple[int, ...], np.dtype[np.float64]]


@dataclass
class _Args(Namespace):
    dataset: str
    directory: str
    target: int
    splits: int
    bias: float
    test_size: float
    randomize_sizes: bool
    no_normalize: bool


_parser = ArgumentParser(
    description="Split a dataset in a biased way",
    epilog=__doc__,
)

_parser.add_argument(
    "dataset",
    help="csv File containing the dateset",
    type=str,
)

_parser.add_argument(
    "-d", "--directory",
    help="where to export the split data",
    type=str,
    default="./splits",
)

_parser.add_argument(
    "-t", "--target",
    help="index of the target column in the dataset",
    type=int,
    required=True,
)

_parser.add_argument(
    "-s", "--splits",
    help="how many subsets to create from the original",
    type=int,
    default=2,
)

_parser.add_argument(
    "-b", "--bias",
    help="bias factor (float between 0 and 1)",
    type=float,
    default=0.5,
)

_parser.add_argument(
    "--test-size",
    help="size of the test set relative to the dataset",
    type=float,
    default=0.15,
)

_parser.add_argument(
    "--randomize-sizes",
    help="randomize the sizes of each split (within reason)",
    type=bool,
    default=True,
)

_parser.add_argument(
    "--no-normalize",
    help="don't perform normalization of the dataset",
    action="store_true",
)


def _create_dir(path: str) -> None:
    p = Path(path)
    if not p.exists():
        p.mkdir()


def export_data(ary: F64Arr, name: str | int, dir: str) -> None:
    """Export a 1d or 2d array as a csv file."""
    hdr = f"No {name}: mean: {ary.mean()}, std: {ary.std()}"
    np.savetxt(f"{dir}/{name}.csv", ary, delimiter=",", header=hdr)


def export_splits(hmap: dict[int, F64Arr], dir: str) -> None:
    """Export split dict to separate csv files."""
    for k, v in hmap.items():
        export_data(v, k, dir)


def _arr_rng_pop(ary: F64Arr, end: int, n_points: int) -> tuple[F64Arr, F64Arr]:
    """Extract random subrange from array.

    Parameters
    ----------
    ary : np.ndarray[tuple[int, ...], np.dtype[np.float[Any]]]
        The array to sample from.
    end : int
        The end of the subrange such that the subrange is (0, end].
    n_points : int
        The number of points to sample from (0, end] ∈ ary.

    Returns
    -------
    (view, ary) : tuple[F64Arr, F64Arr]
        view: The sampled points from ary[0:end,:].
        ary: The input array with the sampled points deleted.

    """
    inds = random.sample(range(end), n_points)
    view = ary[inds]
    ary = np.delete(ary, inds, axis=0)

    return view, ary


def train_test_split(
    data: F64Arr,
    test_size: float,
) -> tuple[F64Arr, F64Arr]:
    """Split dataset in two unequal portions.

    This function will uniformly sample `test_size*n_data` points
    from the `data` array, remove then and return two arrays:
        - left: the sampled points.
        - right: the original array without the sampled points.

    Parameters
    ----------
    data : np.ndarray[tuple[int, ...], np.dtype[np.float[Any]]]
        The array containing the dataset.
    test_size : float between 0 and 1
        The relative ratio of the split.

    Returns
    -------
    left_split, right_split : tuple[F64Arr, F64Arr]

    """
    assert 0 < test_size < 1
    n_data = data.shape[0]
    n_test = math.floor(n_data * test_size)

    return _arr_rng_pop(data, n_data, n_test)


def niid_reg_split(
    data: F64Arr,
    tgt_idx: int,
    n_split: int,
    bias: float,
    randomize_sizes: bool = True,
) -> dict[int, F64Arr]:
    """Non-IID array splitting.

    Given an input dataset `data`. The algorithm:
        1. Sorts `data` by target values _ascending_.
        2. With `n` the number of data points and `n_split` the number
           of required splits, the initial split size is:
             s_split := n / n_split
        3. Selects a random subrange of `data` [0,s] where s ~ N lies in
           the range [s_split/2, s_split].
        4. Pops `n_mode` randomly sampled (modal) points from data[0,s].
        5. Pops `n_rsd` randomly sampled (residual) points from data.
        6. Concatenates the popped points to a dict slot.
    Items (2)-(6) are repeated `n_split` times in total.

    Parameters
    ----------
    data : np.ndarray
        The input dataset containing both features and target.
    tgt_idx : int
        The index of the column containing the target values.
    n_split : int
        The number of times the dataset should be split.
    bias : float
        A number ∈ [0, 1] which weighs sampling from the modal or
        residual points. A value of 0.5 samples fairly and will
        remove bias from the splits.
    randomize_sizes : bool
        Whether to randomize the sizes of each split on top of
        introducing bias to the mode. A value of `False` ensures the
        splits be equal in size.

    Returns
    -------
    hmap : dict[int, F64Arr]
        A dict containing `n_split` biased subsets of `data`.

    """
    n, m = data.shape[0], data.shape[1]
    assert tgt_idx < m

    # sort data by target values
    inds = data[:, tgt_idx].argsort()
    data = data[inds]

    hmap: dict[int, np.ndarray[tuple[int, ...], np.dtype[np.float64]]] = {}
    for i in range(n_split):
        rand = (1 + random.random()) / 2 if randomize_sizes else 1
        s_split = math.floor(n / n_split * rand)
        n_mode = math.floor(bias * s_split)
        n_rsd = s_split - n_mode

        samples: list[F64Arr] = [np.array([])] * 2
        samples[0], data = _arr_rng_pop(data, s_split, n_mode)
        samples[1], data = _arr_rng_pop(data, data.shape[0], n_rsd)

        hmap[i] = np.concatenate(samples)

        print(f"[{i}] n_data: {s_split}\tn_mode: {n_mode}\tn_rsd: {n_rsd}")

    return hmap


if __name__ == "__main__":
    args: _Args = cast(_Args, _parser.parse_args())

    assert 0 <= args.bias <= 1

    random.seed(42)

    data: F64Arr
    test_set: F64Arr
    hmap: dict[int, F64Arr]

    data = np.genfromtxt(args.dataset, delimiter=",", skip_header=1)

    if not args.no_normalize:
        data /= np.linalg.norm(data, axis=0)

    _create_dir(args.directory)
    export_data(data, "full", args.directory)

    test_set, data = train_test_split(data, args.test_size)
    hmap = niid_reg_split(data, 8, args.splits, args.bias,
                          args.randomize_sizes)

    export_data(test_set, "test", args.directory)
    export_data(data, "train", args.directory)
    export_splits(hmap, args.directory)
