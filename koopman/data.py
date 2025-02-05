import numpy as np
from scipy.special import ellipj, ellipk
from jax import numpy as jnp
from typing import Tuple, List


def add_channels(X: np.ndarray) -> np.ndarray:
    if len(X.shape) == 2:
        return X.reshape(X.shape[0], 1, X.shape[1], 1)
    elif len(X.shape) == 3:
        return X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
    else:
        raise ValueError("Unexpected data dimensions")


def get_train_loader(train_sequences: List[np.ndarray], batch_size: int):
    num_samples = train_sequences[0].shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    for start_idx in range(0, num_samples, batch_size):
        batch_idx = indices[start_idx : start_idx + batch_size]
        yield [np.array(seq[batch_idx]) for seq in train_sequences]


def pendulum_dynamics(time: np.ndarray, theta0: float) -> np.ndarray:
    S = np.sin(0.5 * theta0)
    K_S = ellipk(S**2)
    omega_0 = np.sqrt(9.81)
    sn, cn, dn, _ = ellipj(K_S - omega_0 * time, S**2)
    theta = 2.0 * np.arcsin(S * sn)
    d_sn_du = cn * dn
    d_sn_dt = -omega_0 * d_sn_du
    d_theta_dt = 2.0 * S * d_sn_dt / np.sqrt(1.0 - (S * sn) ** 2)
    return np.stack([theta, d_theta_dt], axis=1)


def pendulum_data(
    noise: float = 0.0, theta: float = 2.4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    np.random.seed(1)
    time_series = np.arange(0, 2200 * 0.1, 0.1)
    data = pendulum_dynamics(time_series, theta)
    data = data.T
    clean_data = data.copy()
    data += np.random.standard_normal(data.shape) * noise
    rotation_matrix = np.random.standard_normal((64, 2))
    rotation_matrix, _ = np.linalg.qr(rotation_matrix)
    data = data.T @ rotation_matrix.T
    clean_data = clean_data.T @ rotation_matrix.T
    data = 2 * (data - np.min(data)) / np.ptp(data) - 1
    clean_data = 2 * (clean_data - np.min(clean_data)) / np.ptp(clean_data) - 1
    train_size = 600
    X_train = data[:train_size]
    X_test = data[train_size:]
    X_train_clean = clean_data[:train_size]
    X_test_clean = clean_data[train_size:]
    return X_train, X_test, X_train_clean, X_test_clean, 64, 1
