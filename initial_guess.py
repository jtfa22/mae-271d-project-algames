import numpy as np
from dynamics import get_linear_dynamics


# default guess is no control inputs
def generate(x0, M, N, n, m, dt):
    # x0: list of (n,) numpy arrays
    # M: num players
    # N: horizon
    # n: state size
    # m: control input size

    # integrate separately for each player, assuming
    # zero input as initial guess and all players have
    # identical dynamics
    X = np.empty((0,))
    A = get_linear_dynamics(n,m,dt)[0]
    for i in range(M):
        x = x0[i]
        for j in range(N):
            x = A@x
            X = np.hstack((X, x))
            
    U = np.zeros((M*N*m,)) # guess zero input
    mu = np.ones((M*N*n,)) # guess multipliers of 1

    y0 = np.hstack((X,U,mu))
    return y0, X, U, mu