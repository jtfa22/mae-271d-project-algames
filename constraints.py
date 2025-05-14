# MAE 271D Project
# ALGAMES implementation

import numpy as np
from scipy import linalg


# wall constraint ineqauality (assume one y wall only)
def get_linear_wall_y(y, r, n):
    """Wall inequality for a single player at one timestep: c_wall_ineq * x_k - d_wall_ineq <= 0"""
    ind_y = 1  # index corresponding to y position in state
    c_wall_ineq = np.zeros(n)
    c_wall_ineq[ind_y] = 1
    d_wall_ineq = y - r
    return c_wall_ineq, d_wall_ineq


def get_player_wall_y(y, r, N, n):
    """Wall inequality for one player: C_wall_ineq * X_v - D_wall_ineq <= 0"""
    c_wall_ineq, d_wall_ineq = get_linear_wall_y(y, r, n)
    C_wall_ineq = linalg.block_diag(*([c_wall_ineq] * N))
    D_wall_ineq = np.vstack(([d_wall_ineq] * N))
    return C_wall_ineq, D_wall_ineq


def get_system_wall_y(y, r, M, N, n):
    """Wall inequality for all players: C_wall_sys * X - D_wall_sys <= 0"""
    C_wall_ineq, D_wall_ineq = get_player_wall_y(y, r, N, n)
    C_wall_sys = linalg.block_diag(*([C_wall_ineq] * M))
    D_wall_sys = np.vstack(([D_wall_ineq] * M))
    return C_wall_sys, D_wall_sys


def C_wall(X, U, C_wall_sys, D_wall_sys):
    """wall inequality constraint: C_wall_sys @ X - D_wall_sys <= 0"""
    return C_wall_sys @ X - D_wall_sys


def grad_C_wall(X, U, C_wall_sys, D_wall_sys):
    """gradient wrt X, U"""
    return np.hstack((C_wall_sys, np.zeros((np.shape(C_wall_sys)[0], len(U)))))


# control input inequality constraint
def get_linear_input_bound(m, max_x=5, max_y=5):
    """Control inequality for a single player at one timestep: f_ineq * x_k - g_ineq <= 0"""
    ind_x = 0  # index corresponding to x position in control
    ind_y = 1
    f_ineq = np.zeros((4, m))
    f_ineq[0, ind_x] = 1
    f_ineq[1, ind_x] = -1
    f_ineq[2, ind_y] = 1
    f_ineq[3, ind_y] = -1
    g_ineq = np.hstack((np.ones((2, 1)) * max_x, np.ones((2, 1)) * max_y))
    return f_ineq, g_ineq


def get_player_input_bound(N, m, max_x=5, max_y=5):
    """Control input inequality for one player: F_ineq * X_v - G_ineq <= 0"""
    f_ineq, g_ineq = get_linear_input_bound(m, max_x, max_y)
    F_ineq = linalg.block_diag(*([f_ineq] * N))
    G_ineq = np.vstack(([g_ineq] * N))
    return F_ineq, G_ineq


def get_system_input_bound(M, N, m, max_x=5, max_y=5):
    """Control input inequality for all players: F_sys * X - G_sys <= 0"""
    F_ineq, G_ineq = get_player_input_bound(N, m, max_x, max_y)
    F_sys = linalg.block_diag(*([F_ineq] * M))
    G_sys = np.vstack(([G_ineq] * M))
    return F_sys, G_sys


def C_input(X, U, F_sys, G_sys):
    """control input inequality constraint: F_sys @ U - G_sys <= 0"""
    return F_sys @ U - G_sys


def grad_C_input(X, U, F_sys, G_sys):
    """gradient wrt X, U"""
    return np.hstack((np.zeros((np.shape(F_sys)[0], len(X))), F_sys))


# collision avoidance inequality constraint
def get_system_cola(M, N, n):
    """Collision avoidance inequality for all players:
    r - (C_cola_k_v1_v2 * X).T @ (C_cola_k_v1_v2 * X) <= 0
    each C_cola matrix is formulated per timestep k, per players v1, v2"""
    pos = np.hstack(
        (np.eye(2), np.zeros((2, 2)))
    )  # matrix to select position out of state

    list_cola = []
    for k in range(N):  # timestep
        for v1 in range(M):  # player 1
            for v2 in range(v1 + 1, M):  # player 2
                c_block = [np.zeros((2, n))] * N * M
                ind1 = v1 * N + k
                ind2 = v2 * N + k
                c_block[ind1] = pos
                c_block[ind2] = -1 * pos
                C_cola = np.hstack(c_block)
                list_cola.append(C_cola)
    return list_cola


def C_cola(X, U, r, list_cola):
    """collision avoidance inequality constraint:
    r - (C_cola_k_v1_v2 @ X).T @ (C_cola_k_v1_v2 @ X) <= 0"""
    C_k_v1_v2 = [r - (C_k @ X).T @ (C_k @ X) for C_k in list_cola]
    return np.array(C_k_v1_v2)


def grad_C_cola(X, U, r, list_cola):
    """gradient wrt X, U"""
    C_x = np.vstack([-X.T @ C_k.T @ C_k for C_k in list_cola])
    C_u = np.zeros((np.shape(C_x)[0], len(U)))
    return np.hstack((C_x, C_u))


# all constraints combined
def C(X, U, C_wall_sys, D_wall_sys, F_sys, G_sys, r, list_cola):
    c_wall = C_wall(X, U, C_wall_sys, D_wall_sys)
    c_input = C_input(X, U, F_sys, G_sys)
    c_cola = C_cola(X, U, r, list_cola)
    return np.vstack((c_wall, c_input, c_cola))


def grad_C(X, U, C_wall_sys, D_wall_sys, F_sys, G_sys, r, list_cola):
    c_wall = grad_C_wall(X, U, C_wall_sys, D_wall_sys)
    c_input = grad_C_input(X, U, F_sys, G_sys)
    c_cola = grad_C_cola(X, U, r, list_cola)
    return np.vstack((c_wall, c_input, c_cola))
