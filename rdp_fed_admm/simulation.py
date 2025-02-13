import datetime, os, csv
import numpy as np
import matplotlib.pyplot as plt
from time import time
from .synthetic import generate_synthetic_ols_data, partition_data

from ._admm import ADMMServer, ADMMClient

from ._rdp import get_mechanism

# Define simple OLS ADMM client and server subclasses for simulation
class OLSADMMClient(ADMMClient):
    def __init__(self, **kwargs):
        # Enable elastic aggregation if requested.
        super().__init__(**kwargs)

    def _x_update(self, X: np.ndarray, Y: np.ndarray, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        # Closed-form update for consensus OLS:
        #   x_new = (X^T X + ρ I)^{-1} (X^T Y + ρ (z - u))
        rho = self._penalty_term
        lhs = np.linalg.inv(X.T @ X + rho * np.eye(X.shape[1]))
        return lhs @ (X.T @ Y + rho * (z - u))

    def _x_update_sensitivity(self) -> float:
        # Sensitivity estimate (per analysis): Δf = 4 * step_size / n_data
        return 4 * self._step_size / self._n_data

    def _get_noise(self, sensitivity: float, size: tuple[int, ...] | int = 1) -> np.ndarray:
        """
        Introduce DP noise using the RDP Gaussian mechanism.
        """
        mechanism = get_mechanism(self._dp_mechanism)
        return mechanism(sensitivity, self._dp_params, size)

class OLSADMMServer(ADMMServer):
    def __init__(self, gamma: float = 0.05, **kwargs):
        # Add gamma parameter for the global update
        super().__init__(**kwargs)
        self.gamma = gamma

    def _z_update(self, du: np.ndarray) -> np.ndarray:
        # Global update: new z = current z + gamma * du
        if self._coeffs is None:
            current_z = np.zeros_like(du)
        else:
            current_z = self._coeffs
        return current_z + self.gamma * du

def simulate_fed_admm(clients_data, test_data, server_params, client_params, n_features: int):
    server = OLSADMMServer(
        n_iter=server_params["n_iter"],
        elastic_aggregation=server_params.get("elastic_aggregation", False),
        sens_momentum=server_params.get("sens_momentum", 0.9),
        tau=server_params.get("tau", 0.1),
        clip_min=server_params.get("clip_min", 0.8),
        clip_max=server_params.get("clip_max", 1.2),
        ema_type=server_params.get("ema_type", "per"),
        step_size=server_params.get("step_size", 0.3),
        subset_size=server_params.get("subset_size", 0.7),
        max_clients=len(clients_data)
    )
    # For simulation, bypass socket connections by storing client objects in a list
    num_clients = len(clients_data)
    clients = []
    for cd in clients_data:
        # Set elastic_aggregation flag from client_params
        client = OLSADMMClient(n_iter=client_params["n_iter"],
                               step_size=client_params.get("step_size", 0.3),
                               penalty_term=client_params.get("penalty_term", 10),
                               dp_mechanism=client_params.get("dp_noise", "rdp_gaussian"),
                               dp_params=client_params.get("dp_params", (4.0, 1.0)),
                               elastic_aggregation=client_params.get("elastic_aggregation", False),
                               seed=client_params.get("seed", None))
        # For simulation, we store the local data in the client object
        client._X = cd["X"]
        client._Y = cd["Y"]
        clients.append(client)
    # Initialize each client's dual variable u and state x
    client_states = [{"x": np.zeros(n_features), "u": np.zeros(n_features)} for _ in range(num_clients)]
    # Initialize global model z on server
    z = np.zeros(n_features)
    server._coeffs = z.copy()
    # For evaluation, unpack test data
    test_X, test_Y, _ = test_data

    mae_history = []

    # Main communication rounds
    n_iter = server_params["n_iter"]
    for round in range(n_iter):
        updates = []
        sens_list = []
        weights = []
        # Each client computes local update
        for i, client in enumerate(clients):
            X_local = client._X
            Y_local = client._Y
            state = client_states[i]
            # Ensure that _n_data is set
            client._n_data = X_local.shape[0]
            # Compute new x for client
            x_new = client._x_update(X_local, Y_local, z, state["u"])
            rsd = x_new - z
            sens = client._x_update_sensitivity()
            noise = client._get_noise(sens, size=rsd.shape)
            du = 2 * client._step_size * (rsd + noise)
            state["u"] = state["u"] + du
            updates.append(du)
            sens_list.append(sens)
            weights.append(X_local.shape[0])
        # Weighted aggregation:
        weights = np.array(weights, dtype=np.float32)
        norm_weights = weights / weights.sum()
        weighted_update = np.sum([u * w for u, w in zip(updates, norm_weights)], axis=0)
        if server.elastic_aggregation:
            # For simplicity, use scalar sensitivities: weighted average sensitivity:
            aggregated_sens = sum(s * w for s, w in zip(sens_list, norm_weights))
            if server.sens_ema is None:
                server.sens_ema = np.zeros(n_features)
            # Here, we broadcast aggregated_sens to all coordinates:
            aggregated_sens_vec = np.full(n_features, aggregated_sens)
            server.sens_ema = server.sens_momentum * server.sens_ema + (1 - server.sens_momentum) * aggregated_sens_vec
            max_ema = np.max(server.sens_ema) if np.max(server.sens_ema) > 0 else 1.0
            zeta = 1 + server.tau - (server.sens_ema / max_ema)
            zeta = np.clip(zeta, server.clip_min, server.clip_max)
            elastic_update = zeta * weighted_update
            agg_update = elastic_update
        else:
            agg_update = weighted_update
        # Server update:
        z = server._z_update(agg_update)
        server._coeffs = z.copy()
        # Evaluate MAE on test set:
        preds = test_X @ z
        mae = np.mean(np.abs(preds - test_Y))
        mae_history.append(mae)
        print(f"Round {round+1}/{n_iter} | MAE: {mae:.4f}")

    return z, mae_history


def hyperparameter_search():
    # Create a timestamped directory to store results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"hyperparameter_search_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Define grid of hyperparameters:
    rhos = [1, 5]             # penalty_term (rho)
    lams = [0.05, 0.3]           # step_size (lam)
    taus = [0.1, 0.2]             # tau for elastic aggregation
    gammas = [0.05, 0.1]          # gamma for server update
    sens_momenta = [0.75, 0.9]    # sensitivity EMA momentum
    ema_types = ["per", "global"]  # EMA type
    update_directions = ["plus", "minus"]  # update direction (for legend only)

    results = {}

    # Hyperparameter grid for server parameters:
    results = {}
    n_iter = 100
    for rho in rhos:
        for lam in lams:
            for tau in taus:
                for gamma in gammas:
                    for sens_mom in sens_momenta:
                        for ema in ema_types:
                            for upd in update_directions:
                                server_params = {
                                    "n_iter": n_iter,
                                    "step_size": lam,
                                    "subset_size": 1.0, # full participation
                                    "elastic_aggregation": True,
                                    "sens_momentum": sens_mom,
                                    "tau": tau,
                                    "clip_min": 0.8,
                                    "clip_max": 1.2,
                                    "ema_type": ema,
                                }
                                client_params = {
                                    "n_iter": n_iter,
                                    "step_size": lam,
                                    "penalty_term": rho,
                                    "dp_noise" : "rdp_gaussian",
                                    "dp_params" : (1./137., 0.5), # unit variance
                                    "elastic_aggregation": True,
                                    "seed": 42,
                                }
                                label = (f"rho={rho}, lam={lam}, tau={tau}, gamma={gamma}, "
                                         f"ema={ema}, upd={upd}, sens_mom={sens_mom}")
                                print(f"Running experiment with {label}")
                                # Generate training and test data
                                X_train, Y_train, x_true = generate_synthetic_ols_data(10000, 20, noise_std=0.1)
                                X_test, Y_test, _ = generate_synthetic_ols_data(int(10000 * 0.2), 20, noise_std=0.1)
                                clients_data = partition_data(X_train, Y_train, num_clients=5)
                                z_final, mae_history = simulate_fed_admm(
                                    clients_data,
                                    (X_test, Y_test, x_true),
                                    server_params,
                                    client_params,
                                    n_features=20
                                )
                                results[label] = mae_history
    # Save results to CSV:
    with open(f"{results_dir}/results.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["config"] + [f"round_{i+1}" for i in range(n_iter)]
        writer.writerow(header)
        for label, history in results.items():
            writer.writerow([label] + history)
    
    window_size = 100
    tolerance = 1e-3 # at least 1e-3 and above to be noticable
    decreasing_results = {label: history for label, history in results.items()
                          if is_relatively_decreasing(history, window_size, tolerance)}
    
    # Plot and save
    plt.figure(figsize=(15, 10))
    for label, history in decreasing_results.items():
        rounds = list(range(1, len(history) + 1))
        plt.plot(rounds, history, label=label, linewidth=1, alpha=0.7)
    plt.xlabel("Communication Round")
    plt.ylabel("MAE")
    plt.title(f"MAE vs Rounds (Window = {window_size}, Tol = {tolerance})")
    plt.legend(fontsize=6, loc="upper right", ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/decreasing_trend_mae_vs_rounds.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    print(f"Results saved to directory: {results_dir}")
    return results

# Filter configurations with an overall decreasing trend
def is_relatively_decreasing(mae_history, window_size=10, tolerance=1e-4):
    if len(mae_history) < window_size:
        return False
    for i in range(len(mae_history) - window_size + 1):
        window = mae_history[i:i + window_size]
        if all(window[j] <= window[j-1] + tolerance for j in range(1, len(window))):
            return True
    return False
    
if __name__ == "__main__":
    np.random.seed(42)
    start_time = time()
    hyperparameter_search()
    print(f"Total runtime: {time() - start_time:.2f} seconds")