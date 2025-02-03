# Differentially Private Federated ADMM

This repo contains a prototype implementation of a Rényi-Differentially-Private
(somewhat) resource aware Federated ADMM algorithm built on solely using `numpy`
and `sockets`.

This was developed _in part_ as a project assignment for UoC's CS573
_Optimization Methods_.

# Building

> [!NOTE]
> All commands assume to be run from inside the project directory. Ensure you
> are in the right place.

This project contains a `flake.nix`, thus nix users may simply switch to the
project directory and run:
```
$ nix run
```
to build the project and _execute the demo_ or:
```
$ nix build
```
to _just build_. For everyone else, this project depends _solely on
[numpy](https://numpy.org)_ and uses [hatch](https://github.com/pypa/hatch) as a
build tool so install both and run:
```
$ hatch build
```
This should create the project wheel in the `./dist` directory which can be
installed normally with `pip install`.

# Running

> [!INFORMATION]
> TLDR; `python -mrdp_fed_admm.rdp_fed_admm --help`

Once built and installed, the demo client can be run either directly or through
python:
```
$ python -mrdp_fed_admm.rdp_fed_admm
```
The cli supports two modes `client` and `server`. To start a **client** run:
```
$ ./rdp-fed-admm client -t target_idx -i client_idx data.csv
```
Replace `data.csv` with a csv file of your dataset which _must include the
target_, `target_idx` with the index of the target column in the dataset and
`client_idx` with a numeric id you wish to give to the client.

To start a **server** run:
```
$ ./rdp-fed-admm server -c num_clients -f num_features
```
Replace `num_clients` with the desired size of the client pool and
`num_features` with the number of features in the dataset.

> [!NOTE]
> The server doesn't require data

Once the clients are connected, optimization should start and if everything
worked properly, nothing should happen. If you want to export the coefficients
of the server for predictions or evaluation see `--coeff-file` and
`--coeff-hist`.

# Contents of this Repo

The project is built with OOP principles (as much as one can do this in python)
and thus contains pluggable classes for:
- A simple `sockets` based network layer `net.py` incorporating a _client_ and a
  _server_ which communicate via the `recv_array()` method which can send
  `numpy` arrays back and forth.

- A base ADMM library `admm.py` which builds upon `net.py` and specialises its
  classes for ADMM by adding common hyperparameters, caching and other required
  features.

- A series of objective libraries (under `objectives/`) which specialize ADMM to
  specific optimization objectives.

- A configurable cli demo which allows spawning a client or a server.

Currently, the publicly exposed modules of `rdp-fed-admm` are only the
objectives:
- The lasso objective `lasso.py`.
- The elastic net objective `elasticnet.py` which slightly modifies the
  aggregation step of lasso.

The cli program is registered as the main program of the project and should be
made available with standard install methods.
