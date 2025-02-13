import numpy as np

def generate_synthetic_ols_data(n_samples: int, d: int, noise_std: float = 0.1):
    """
    Generate synthetic data for an OLS problem:
       Y = X @ x_true + noise.
    Returns:
       X: [n_samples, d] matrix
       Y: [n_samples,] vector
       x_true: ground-truth parameter vector of length d.
    """
    X = np.random.randn(n_samples, d).astype(np.float32)
    x_true = np.random.randn(d).astype(np.float32)
    noise = noise_std * np.random.randn(n_samples).astype(np.float32)
    Y = X @ x_true + noise
    return X, Y, x_true

def partition_data(X: np.ndarray, Y: np.ndarray, num_clients: int):
    """
    Partition the dataset (X, Y) among num_clients.
    Returns a list of dictionaries with keys 'X' and 'Y'.
    """
    n = X.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    partitions = np.array_split(indices, num_clients)
    client_partitions = [{"X": X[part], "Y": Y[part]} for part in partitions]
    return client_partitions