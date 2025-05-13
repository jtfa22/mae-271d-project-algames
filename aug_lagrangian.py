# MAE 271D Project
# ALGAMES implementation

import numpy as np
from scipy import linalg


# augmented lagrangian
def L_v(X,U, mu_v, lam, Irho, J_v, D, C):
    """Augmented Lagrangian"""
    return J_v + mu_v.T @ D + lam.T @ C + 0.5 * C.T @ Irho @ C

def grad_L_v(X,U, mu_v, lam, Irho, grad_J_v, grad_D, grad_C):
    # TODO missing penalty calcs
    pass