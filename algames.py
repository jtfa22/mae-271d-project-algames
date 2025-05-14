# MAE 271D Project
# ALGAMES implementation

import numpy as np
from scipy import linalg, optimize

from . import aug_lagrangian, constraints, dynamics, objective


# function that runs algames solver once
# will call in each iteration of MPC loop
def ALGAMES(
    M,  # number players
    N,  # horizon
    dt,  # timestep
    r,  # collision avoidance radius
    x0,  # list of initial states
    xf,  # list of target states
    Q,  # running cost matrix
    Qf,  # terminal cost matrix
    R,  # control cost matrix
    rho,  # constraint penalty value
    gamma,  # constraint penalty schedule
    eps,  # convergence tolerance
    constraint_wall_y,  # y value of horizontal wall
    constraint_u_x_max,  # control input x bound
    constraint_u_y_max,  # control input y bound
):
    # double integrator model
    n = 4  # state size (x, y, v_x, v_y)
    m = 2  # control input size (a_x, a_y)

    # define dynamics matrices
    A_sys, B_sys, E_sys = dynamics.get_system_dynamics(M, N, n, m, dt)

    # define constraints matrices
    C_wall_sys, D_wall_sys = constraints.get_system_wall_y(
        constraint_wall_y, r, M, N, n
    )
    F_sys, G_sys = constraints.get_system_input_bound(
        M, N, m, constraint_u_x_max, constraint_u_y_max
    )
    list_cola = constraints.get_system_cola(M, N, n)

    # create initial guess
    y0 = ...  # TODO [X, U, mu]
    lam = ...

    # ALGAMES loop - until y converge
    y = y0
    while 0.0 > eps:  # TODO
        # solve G
        al_args = (
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
        )

        # currently uses derivative free method
        # TODO derive the 2nd derivative of Lagrangian
        sol = optimize.root(aug_lagrangian.grad_aug_lagrangian, y, args=al_args)
        y = sol.x

        # split y into X, U, mu
        X, U, mu = np.split(y, [M * N * n, M * N * (n + m)])

        #  calculate new constraints
        C = constraints.C(X, U, C_wall_sys, D_wall_sys, F_sys, G_sys, r, list_cola)

        # dual ascent penalty update

    # return trajectory
