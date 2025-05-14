# MAE 271D Project
# ALGAMES implementation

import constraints
import dynamics
import numpy as np
import objective


# penalty matrix
def Irho(C, lam, rho):
    """Form penalty diagonal matrix"""
    irho_vec = [rho if C[k] < 0 and lam[k] == 0 else 0 for k in range(len(C))]
    return np.diag(irho_vec)


def grad_penalty(X, U, Irho, C_wall_sys, D_wall_sys, F_sys, G_sys, r, list_cola):
    """gradient of 1/2 * C.T @ Irho @ C wrt X, U"""
    irho_vec = np.diag(Irho)

    # wall
    size_wall = np.shape(C_wall_sys)[0]
    Irho1 = np.diag(irho_vec[:size_wall])
    c_wall_x = X.T @ C_wall_sys.T @ Irho1 @ C_wall_sys
    c_wall_u = np.zeros(len(U))
    c_wall = np.hstack((c_wall_x, c_wall_u))

    # input
    size_input = np.shape(F_sys)[0]
    Irho2 = np.diag(irho_vec[size_wall : size_wall + size_input])
    c_input_x = np.zeros(len(X))
    c_input_u = U.T @ F_sys.T @ Irho2 @ F_sys
    c_input = np.hstack((c_input_x, c_input_u))

    # cola
    size_cola = len(list_cola)
    assert size_wall + size_input + size_cola == len(irho_vec)
    irho3_vec = irho_vec[size_wall + size_input :]
    c_cola_x = np.array(
        [
            2 * -X.T @ C_k.T @ C_k * rho_k * (r - (C_k @ X).T @ (C_k @ X))
            for C_k, rho_k in zip(list_cola, irho3_vec)
        ]
    )
    c_cola_u = np.zeros(len(U))
    c_cola = np.hstack((c_cola_x, c_cola_u))

    # add all
    return c_wall + c_input + c_cola


# augmented lagrangian
def L_v(mu_v, lam, Irho, J_v, D, C):
    """Augmented Lagrangian of one player"""
    return J_v + mu_v.T @ D + lam.T @ C + 0.5 * C.T @ Irho @ C


def grad_L_v(mu_v, lam, grad_J_v, grad_D, grad_C, grad_penalty):
    """gradient of Augmented Lagrangian of one player wrt X, U"""
    return grad_J_v + mu_v.T @ grad_D + lam.T @ grad_C + grad_penalty


def grad_aug_lagrangian(
    y,
    M,
    N,
    n,
    m,
    lam,
    rho,
    Q,
    Qf,
    R,
    x0,
    xf,
    A_sys,
    B_sys,
    E_sys,
    C_wall_sys,
    D_wall_sys,
    F_sys,
    G_sys,
    r,
    list_cola,
):
    """gradient of Augmented Lagrangian of all players wrt X, U"""
    # split y into X, U, mu
    X, U, mu = np.split(y, [M * N * n, M * N * (n + m)])

    # calculate current constraints
    C = constraints.C(X, U, C_wall_sys, D_wall_sys, F_sys, G_sys, r, list_cola)

    # compute new penalty matrix
    Irho = Irho(C, lam, rho)

    # compute derivative of augmented lagrangian
    list_grad_J_v = [
        objective.grad_J_v(X, U, u_v, Q, Qf, R, M, N, n, m, xf) for u_v in range(M)
    ]
    grad_D = dynamics.grad_D(X, U, A_sys, B_sys, E_sys)
    grad_C = constraints.grad_C(
        X, U, C_wall_sys, D_wall_sys, F_sys, G_sys, r, list_cola
    )
    grad_penalty = grad_penalty(
        X, U, Irho, C_wall_sys, D_wall_sys, F_sys, G_sys, r, list_cola
    )

    # compte gradient of aug lagrangian for each player
    g_players = [
        grad_L_v(mu_v, lam, grad_J_v, grad_D, grad_C, grad_penalty)
        for mu_v, grad_J_v in zip(np.split(mu, M), list_grad_J_v)
    ]
    g_players.append(dynamics.D(X, U, A_sys, B_sys, E_sys, x0))
    return np.hstack(g_players)
