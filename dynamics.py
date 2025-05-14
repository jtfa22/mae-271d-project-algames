# MAE 271D Project
# ALGAMES implementation

import numpy as np
from scipy import linalg

# assuming double integrator model


# constant matrices:
def get_linear_dynamics(n, m, dt):
    """dynamics for one player at one timestep: x_k+1 = A @ x_k + B @ u_k"""
    A = np.eye(n) + np.diag([dt, dt], k=int(n / 2))
    B = np.vstack((np.eye(m) * dt**2 / 2, (np.eye(m) * dt)))
    return A, B


def get_player_dynamics(N, n, m, dt):
    """dynamics for one player: A_eq @ X_v + B_eq @ U_v = E_eq @ x0"""
    A, B = get_linear_dynamics(n, m, dt)
    a_block = linalg.block_diag(*([-A] * (N - 1)))
    A_eq = np.eye(N * n) + np.pad(a_block, [(n, 0), (0, n)])
    B_eq = linalg.block_diag(*([-B] * N))
    E_eq = np.vstack((A, np.zeros(((N - 1) * n, n))))
    return A_eq, B_eq, E_eq


def get_system_dynamics(M, N, n, m, dt):
    """system dynamics for all players: A_sys @ X + B_sys @ U = E_sys @ x0"""
    A_eq, B_eq, E_eq = get_player_dynamics(N, n, m, dt)
    A_sys = linalg.block_diag(*([A_eq] * M))
    B_sys = linalg.block_diag(*([B_eq] * M))
    E_sys = np.vstack(([E_eq] * M))
    return A_sys, B_sys, E_sys


# optimal control formulation
def D(X, U, A_sys, B_sys, E_sys, x0):
    """equality constraint: A_sys @ X + B_sys @ U - E_sys @ x0 = 0"""
    return A_sys @ X + B_sys @ U - E_sys @ x0


def grad_D(X, U, A_sys, B_sys, E_sys):
    """gradient wrt X, U"""
    return np.hstack((A_sys, B_sys))
