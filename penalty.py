# MAE 271D Project
# ALGAMES implementation

import numpy as np
from scipy import linalg


# penalty matrix
def Irho(C, lam, rho):
    """Form penalty diagonal matrix"""
    irho_vec = [0 if C[k] < 0 and lam[k] == 0 else rho for k in range(len(C))]
    return np.diag(irho_vec)


def grad_penalty(X, U, Irho, C_wall_sys, D_wall_sys, F_sys, G_sys, r, list_cola):
    """gradient of 1/2 * C.T @ Irho @ C wrt X, U"""
    irho_vec = np.diag(Irho)

    # wall
    size_wall = np.shape(C_wall_sys)[0]
    Irho1 = np.diag(irho_vec[:size_wall])
    c_wall_x = (C_wall_sys @ X - D_wall_sys).T @ Irho1 @ C_wall_sys
    c_wall_u = np.zeros(len(U))
    c_wall = np.hstack((c_wall_x, c_wall_u))

    # input
    size_input = np.shape(F_sys)[0]
    Irho2 = np.diag(irho_vec[size_wall : size_wall + size_input])
    c_input_x = np.zeros(len(X))
    c_input_u = (F_sys @ U - G_sys).T @ Irho2 @ F_sys
    c_input = np.hstack((c_input_x, c_input_u))

    # cola
    size_cola = len(list_cola)
    assert size_wall + size_input + size_cola == len(irho_vec)
    irho3_vec = irho_vec[size_wall + size_input :]
    c_cola_x = np.sum(
        [
            2 * -X.T @ C_k.T @ C_k * rho_k * (r**2 - (C_k @ X).T @ (C_k @ X))
            for C_k, rho_k in zip(list_cola, irho3_vec)
        ],
        axis=0,
    )
    c_cola_u = np.zeros(len(U))
    c_cola = np.hstack((c_cola_x, c_cola_u))

    # add all
    return c_wall + c_input + c_cola


def hess_penalty(X, U, mu, Irho, C_wall_sys, D_wall_sys, F_sys, G_sys, r, list_cola):
    """Hessian of 1/2 * C.T @ Irho @ C wrt X, U, mu"""
    len_xu = len(X) + len(U)
    len_y = len(X) + len(U) + len(mu)
    irho_vec = np.diag(Irho)

    # wall
    size_wall = np.shape(C_wall_sys)[0]
    Irho1 = np.diag(irho_vec[:size_wall])
    c_wall_x = np.vstack(
        (C_wall_sys.T @ Irho1 @ C_wall_sys, np.zeros((len(U), len(X))))
    )
    c_wall_u = np.zeros((len_xu, len(U)))
    c_wall_mu = np.zeros((len_xu, len(mu)))
    c_wall = np.hstack((c_wall_x, c_wall_u, c_wall_mu))

    # input
    size_input = np.shape(F_sys)[0]
    Irho2 = np.diag(irho_vec[size_wall : size_wall + size_input])
    c_input_x = np.zeros((len_xu, len(X)))
    c_input_u = np.vstack((np.zeros((len(X), len(U))), F_sys.T @ Irho2 @ F_sys))
    c_input_mu = np.zeros((len_xu, len(mu)))
    c_input = np.hstack((c_input_x, c_input_u, c_input_mu))

    # cola
    size_cola = len(list_cola)
    assert size_wall + size_input + size_cola == len(irho_vec)
    irho3_vec = irho_vec[size_wall + size_input :]
    c_cola_x = np.sum(
        [
            np.vstack(
                (
                    -2 * C_k.T @ C_k * rho_k * (r**2 - (C_k @ X).T @ (C_k @ X)),
                    np.zeros((len(U), len(X))),
                )
            )
            for C_k, rho_k in zip(list_cola, irho3_vec)
        ],
        axis=0,
    )
    c_cola_u = np.zeros((len_xu, len(U)))
    c_cola_mu = np.zeros((len_xu, len(mu)))
    c_cola = np.hstack((c_cola_x, c_cola_u, c_cola_mu))

    # add all
    return c_wall + c_input + c_cola
