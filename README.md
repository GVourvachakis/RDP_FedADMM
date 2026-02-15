# EP-FedADMM: Elastic Private Federated ADMM with Rényi Differential Privacy

> **Heterogeneity in Private Federated Learning: Elastic ADMM, RDP Mechanisms, and Non-IID Partitioning**

A research prototype implementing **EP-FedADMM** (Elastic Private Federated ADMM) — a framework for privacy-preserving federated optimization under heterogeneous, non-IID client data distributions. Built entirely on `numpy` and `sockets`, this system integrates Rényi Differential Privacy (RDP) guarantees, weighted aggregation for client-drift mitigation, and a statistical framework for non-IID data synthesis and analysis.

Originally developed as a project for UoC's [**CS573 Optimization Methods**](https://www.csd.uoc.gr/CSD/index.jsp?content=pg_courses_catalog&lang=en&area_id=33&course=152), this work has since grown into a self-contained research contribution spanning federated learning theory, differential privacy, and distributed optimization.

---

## Overview

Federated learning must simultaneously balance privacy, utility, and robustness to heterogeneous data — a challenge this work addresses through three integrated contributions:

**1. EP-FedADMM Algorithm**
A Resource-aware Federated ADMM formulation for Elastic Net regression that incorporates weighted aggregation to mitigate client drift in non-IID scenarios, with pluggable noise injection mechanisms for formal privacy guarantees.

**2. Non-IID Statistical Framework**
A systematic procedure for synthesizing and analyzing non-IID client data distributions, grounded in theoretical concentration bounds for randomized partitioning, HSIC-based independence quantification, and inter-/intra-partition variance analysis.

**3. Unified RDP Privacy Analysis**
Formal Rényi Differential Privacy guarantees for both truncated and untruncated Gaussian and Laplacian noise mechanisms, including RDP-to-(ε,δ)-DP conversion bounds and equivalence regimes for noise mechanisms.

Empirical evaluations on synthetic and real-world datasets demonstrate that under strict privacy constraints, the approach achieves a mean absolute error (MAE) of **0.017**, outperforming heuristic baselines in robustness — at an expected utility cost relative to non-private models.

---

## Repository Structure

```
rdp_fed_admm/
├── _net.py              # Sockets-based network layer (client & server)
├── _admm.py             # Base ADMM library (hyperparameters, caching, coordination)
├── rdp_fed_admm.py      # CLI entry point and demo
└── objectives/
    ├── lasso.py         # LASSO objective
    └── elasticnet.py    # Elastic Net objective (modified aggregation step)
```

The project is designed with OOP principles and pluggable components:

- **`_net.py`** — A lightweight `sockets`-based network layer with `client` and `server` classes communicating via `recv_array()`, which efficiently serializes and transfers `numpy` arrays over the wire.
- **`_admm.py`** — Specializes the network layer for ADMM by introducing common hyperparameters (ρ, penalty terms), caching of intermediate updates, and synchronization primitives.
- **`objectives/`** — Pluggable objective modules. Currently exposes:
  - `lasso.py` — Standard LASSO regression objective.
  - `elasticnet.py` — Elastic Net objective with modified weighted aggregation to reduce client drift.
- **`rdp_fed_admm.py`** — CLI demo for spawning clients and servers in a federated topology.

---

## Building

> [!NOTE]
> All commands assume you are running from inside the project directory.

### Nix (recommended)

```bash
# Build and run the demo
nix run

# Build only
nix build
```

### Standard (pip + hatch)

This project depends solely on [`numpy`](https://numpy.org) and uses [`hatch`](https://github.com/pypa/hatch) as its build tool.

```bash
pip install hatch
hatch build
pip install dist/*.whl
```

---

## Running

```
python -m rdp_fed_admm.rdp_fed_admm --help
```

The CLI supports two modes: `client` and `server`.

### Starting a Server

```bash
./rdp-fed-admm server -c <num_clients> -f <num_features>
```

| Flag | Description |
|------|-------------|
| `-c` | Number of clients expected in the pool |
| `-f` | Number of input features in the dataset |

The server coordinates global model aggregation and does not require local data.

### Starting a Client

```bash
./rdp-fed-admm client -t <target_idx> -i <client_idx> data.csv
```

| Flag | Description |
|------|-------------|
| `-t` | Column index of the target variable in the CSV |
| `-i` | Numeric ID to assign to this client |
| `data.csv` | Local dataset, including the target column |

Once all clients are connected, federated optimization begins automatically. To export the server's learned coefficients for evaluation or downstream use, see the `--coeff-file` and `--coeff-hist` flags.

---

## Theoretical Foundations

The project provides formal theoretical analysis across three areas, detailed in the report's appendices ([`RDP_FedADMM.pdf`](RDP_FedADMM.pdf)):

- **Privacy Guarantees**: RDP accounting for Gaussian and Laplacian mechanisms (truncated and untruncated), with conversion to (ε,δ)-DP via tight bounds.
- **Non-IID Analysis**: Concentration inequalities under randomized data splitting; HSIC-based quantification of partition independence; inter-/intra-partition variance metrics.
- **Convergence & Drift**: Weighted ADMM aggregation analysis for heterogeneous client distributions; elastic net regularization properties under federated constraints.

---

## Authors & Contributions

This project is a joint work by:

**Georgios S. Vourvachakis** ([@GVourvachakis](https://github.com/GVourvachakis))
Primarily responsible for the theoretical framework (RDP analysis, concentration bounds, non-IID synthesis), high-level algorithmic implementation, benchmarking, prototyping, and literature comparison.

**Alexandros C. Traios** (GitLab: [csdp1353](https://gitlab-csd.datacenter.uoc.gr/csdp1353))
Primarily responsible for the socket-based network layer, low-level backbone implementation, object-oriented modular architecture, and the overall software pipeline.

Both authors contributed equally to the conceptualization, algorithmic design, and CS573 course presentation.

---

## License

This project is licensed under the **GNU General Public License v3.0**. See [`LICENSE`](LICENSE) for details.

---

## Citation

If you use this work in your research, please cite it as:

```bibtex
@misc{vourvachakis2025epfedadmm,
  title     = {Heterogeneity in Private Federated Learning: Elastic ADMM, RDP Mechanisms, and Non-IID Partitioning},
  author    = {Vourvachakis, Georgios and Traios, Alexandros C.},
  year      = {2025},
  url       = {https://github.com/GVourvachakis/RDP_FedADMM}
}
```
