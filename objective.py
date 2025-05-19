# MAE 271D Project
# ALGAMES implementation

import numpy as np
from scipy import linalg


# quadratic cost of player
def J_v(X, U, u_v, Q, Qf, R, M, N, n, m, xf):
    """objective function of player v (index)"""
    cost = 0

    # trajectory states
    for k in range(0, N - 2):
        for v in range(M):
            ind = (v * N + k) * n
            xk = X[ind : ind + n]
            cost += 0.5 * (xk - xf).T @ Q @ (xk - xf)

    # final state
    k = N - 1
    ind = (v * N + k) * n
    xk = X[ind : ind + n]
    cost += 0.5 * (xk - xf).T @ Qf @ (xk - xf)

    # player control input
    for k in range(0, N - 1):
        ind = (u_v * N + k) * m
        uk = U[ind : ind + m]
        cost += 0.5 * uk.T @ R @ uk

    return cost


def grad_J_v(X, U, u_v, Q, Qf, R, M, N, n, m, list_xf):
    """gradient wrt X, U"""
    # wrt x
    xf_sys = np.concatenate([np.tile(xf, N) for xf in list_xf])
    Q_sys = linalg.block_diag(*(([Q] * (N - 1) + [Qf]) * M))
    J_x = (X - xf_sys).T @ Q_sys

    # wrt u
    ind = u_v * N * m
    U_v = np.zeros(np.shape(U))
    U_v[ind : ind + N * m] = U[ind : ind + N * m]
    R_sys = linalg.block_diag(*([R] * M * N))
    J_u = U_v.T @ R_sys

    return np.hstack((J_x, J_u))


def hess_J_v(X, U, mu, u_v, Q, Qf, R, M, N, n, m, list_xf):
    """hessian wrt X, U, mu"""
    len_xu = len(X) + len(U)

    # wrt x
    Q_sys = linalg.block_diag(*(([Q] * (N - 1) + [Qf]) * M))
    J_x = np.vstack((Q_sys, np.zeros((len(U), len(X)))))

    # wrt u
    R_sys = linalg.block_diag(*([R] * M * N))
    J_u = np.vstack((np.zeros((len(X), len(U))), R_sys))

    # wrt mu
    J_mu = np.zeros((len_xu, len(mu)))

    return np.hstack((J_x, J_u, J_mu))
