# MAE 271D Project
# ALGAMES implementation

import constraints
import dynamics
import numpy as np
import objective
import penalty
from scipy import linalg


# augmented lagrangian
def L_v(u_v, N, n, mu, lam, Irho, J_v, D, C):
    """Augmented Lagrangian of one player"""
    # get player
    ind = u_v * N * n
    mu_v = mu[ind : ind + N * n]
    D_v = D[ind : ind + N * n]

    return J_v + mu_v.T @ D_v + lam.T @ C + 0.5 * C.T @ Irho @ C


def grad_L_v(u_v, N, n, m, mu, lam, grad_J_v, grad_D, grad_C, grad_penalty):
    """gradient of Augmented Lagrangian of one player wrt X, U"""
    # get player
    ind_x = u_v * N * n
    ind_u = u_v * N * m

    mu_v = np.zeros(np.shape(mu))
    mu_v[ind_x : ind_x + N * n] = mu[ind_x : ind_x + N * n]

    grad_D_v = np.zeros(np.shape(grad_D))
    grad_D_v[ind_x : ind_x + N * n, ind_u : ind_u + N * m] = grad_D[
        ind_x : ind_x + N * n, ind_u : ind_u + N * m
    ]

    return grad_J_v + mu_v.T @ grad_D_v + lam.T @ grad_C + grad_penalty


def hess_L_v(u_v, N, n, m, mu, lam, hess_J_v, hess_D, hess_C, hess_penalty):
    """hessian of Augmented Lagrangian of one player wrt X, U"""
    # ignore correct selection since hess_D is all zero
    mu_v = mu
    hess_D_v = hess_D

    return hess_J_v + mu_v.T @ hess_D_v + lam.T @ hess_C + hess_penalty


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
    list_x0,
    list_xf,
    A_sys,
    B_sys,
    E_sys,
    C_wall_sys,
    D_wall_sys,
    F_sys,
    G_sys,
    r,
    list_cola,
    dynamics_mult,
):
    """gradient of Augmented Lagrangian of all players wrt X, U"""
    # split y into X, U, mu
    X, U, mu = np.split(y, [M * N * n, M * N * (n + m)])

    # calculate current constraints
    C = constraints.C(X, U, C_wall_sys, D_wall_sys, F_sys, G_sys, r, list_cola)

    # compute new penalty matrix
    I_rho = penalty.Irho(C, lam, rho)

    # compute derivative of augmented lagrangian
    list_grad_J_v = [
        objective.grad_J_v(X, U, u_v, Q, Qf, R, M, N, n, m, list_xf) for u_v in range(M)
    ]
    grad_D = dynamics.grad_D(X, U, A_sys, B_sys, E_sys)
    grad_C = constraints.grad_C(
        X, U, C_wall_sys, D_wall_sys, F_sys, G_sys, r, list_cola
    )
    grad_C_penalty = penalty.grad_penalty(
        X, U, I_rho, C_wall_sys, D_wall_sys, F_sys, G_sys, r, list_cola
    )

    # compte gradient of aug lagrangian for each player
    g_players = [
        grad_L_v(u_v, N, n, m, mu, lam, grad_J_v, grad_D, grad_C, grad_C_penalty)
        for u_v, grad_J_v in enumerate(list_grad_J_v)
    ]
    # append dynamics to root solving problem
    x0 = np.hstack(list_x0)
    g_players.append(dynamics.D(X, U, A_sys, B_sys, E_sys, x0) * dynamics_mult)

    return np.concatenate(g_players)


def hess_aug_lagrangian(
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
    list_x0,
    list_xf,
    A_sys,
    B_sys,
    E_sys,
    C_wall_sys,
    D_wall_sys,
    F_sys,
    G_sys,
    r,
    list_cola,
    dynamics_mult,
):
    """hessian of Augmented Lagrangian of all players wrt X, U"""
    # split y into X, U, mu
    X, U, mu = np.split(y, [M * N * n, M * N * (n + m)])

    # calculate current constraints
    C = constraints.C(X, U, C_wall_sys, D_wall_sys, F_sys, G_sys, r, list_cola)

    # compute new penalty matrix
    I_rho = penalty.Irho(C, lam, rho)

    # compute derivative of augmented lagrangian
    list_hess_J_v = [
        objective.hess_J_v(X, U, mu, u_v, Q, Qf, R, M, N, n, m, list_xf)
        for u_v in range(M)
    ]
    hess_D = dynamics.hess_D(X, U, mu, A_sys, B_sys, E_sys)
    hess_C = constraints.hess_C(
        X, U, mu, C_wall_sys, D_wall_sys, F_sys, G_sys, r, list_cola
    )
    hess_C_penalty = penalty.hess_penalty(
        X, U, mu, I_rho, C_wall_sys, D_wall_sys, F_sys, G_sys, r, list_cola
    )

    # compte hessian of aug lagrangian for each player
    g_players = [
        hess_L_v(u_v, N, n, m, mu, lam, hess_J_v, hess_D, hess_C, hess_C_penalty)
        for u_v, hess_J_v in enumerate(list_hess_J_v)
    ]
    # append dynamics to root solving problem
    x0 = np.hstack(list_x0)
    grad_D = dynamics.grad_D(X, U, A_sys, B_sys, E_sys) * dynamics_mult
    grad_D = np.hstack((grad_D, np.zeros((len(X), len(mu)))))
    g_players.append(grad_D)

    return np.vstack(g_players)
