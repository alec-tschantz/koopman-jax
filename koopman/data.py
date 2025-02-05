from typing import Tuple, List


import numpy as np
from scipy.special import ellipj, ellipk


def pendulum_dynamics(t: np.ndarray, theta0: float) -> np.ndarray:
    s = np.sin(0.5 * theta0)
    k_s = ellipk(s**2)
    omega_0 = np.sqrt(9.81)
    sn, cn, dn, _ = ellipj(k_s - omega_0 * t, s**2)
    theta = 2.0 * np.arcsin(s * sn)
    d_sn_du = cn * dn
    d_sn_dt = -omega_0 * d_sn_du
    d_theta_dt = 2.0 * s * d_sn_dt / np.sqrt(1.0 - (s * sn) ** 2)
    return np.stack([theta, d_theta_dt], axis=1)


def pendulum_data(
    noise: float = 0.0, theta: float = 2.4, train_size: int = 600
) -> Tuple[np.ndarray, np.ndarray]:
    t_series = np.arange(0, 2200 * 0.1, 0.1)
    data = pendulum_dynamics(t_series, theta)
    data = 2 * (data - np.min(data)) / np.ptp(data) - 1
    x_train, x_test = data[:train_size], data[train_size:]
    return x_train, x_test


def train_loader(data: np.ndarray, num_steps: int, batch_size: int):
    train_sequences = [
        data[i:] if i == 0 else data[:-i] for i in range(num_steps, -1, -1)
    ]
    num_samples = train_sequences[0].shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    for start in range(0, num_samples, batch_size):
        batch_idx = indices[start : start + batch_size]
        yield [seq[batch_idx] for seq in train_sequences]
