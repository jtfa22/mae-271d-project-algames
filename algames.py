# MAE 271D Project
# ALGAMES implementation

import aug_lagrangian
import constraints
import dynamics
import initial_guess
import numpy as np
import objective
import update_weights
from scipy import linalg, optimize


# function that runs algames solver once
# will call in each iteration of MPC loop
def ALGAMES(
    M,  # number players
    N,  # horizon
    dt,  # timestep
    r,  # collision avoidance radius
    list_x0,  # list of initial states of all players
    list_xf,  # list of target states of all players
    Q,  # running cost matrix
    Qf,  # terminal cost matrix
    R,  # control cost matrix
    rho,  # constraint penalty value
    gamma,  # constraint penalty schedule
    eps,  # convergence tolerance
    constraint_wall_y,  # y value of horizontal wall
    constraint_u_x_max,  # control input x bound
    constraint_u_y_max,  # control input y bound
    max_iter,  # maximum number of iterations
    dynamics_mult=1000,  # multiplier on dynamics in root finding problem
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
    y0, X_guess, U_guess, mu_guess = initial_guess.generate(
        list_x0, M, N, n, m, dt
    )  # [X, U, mu]
    C = constraints.C(
        X_guess, U_guess, C_wall_sys, D_wall_sys, F_sys, G_sys, r, list_cola
    )

    lam = np.zeros(len(C))

    # ALGAMES loop - until y converge
    y = y0
    yprev = y + 2 * eps * np.ones(y.shape)

    iter = 0
    while (abs(yprev - y) > eps).any() and iter < max_iter:  # TODO
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
        )

        yprev = y

        # use solver
        sol = optimize.root(
            aug_lagrangian.grad_aug_lagrangian,
            y,
            method="lm",
            jac=aug_lagrangian.hess_aug_lagrangian,
            args=al_args,
        )
        y = sol.x

        # split y into X, U, mu
        X, U, mu = np.split(y, [M * N * n, M * N * (n + m)])

        #  calculate new constraints
        C = constraints.C(X, U, C_wall_sys, D_wall_sys, F_sys, G_sys, r, list_cola)

        # debug print
        print(f"{iter} y_max={round(max(y),3)} rho={rho} C_vio={round(max(C),3)}")

        # dual ascent penalty update
        # current implementation only has ineq constraints
        lam = update_weights.dual_ascent_update(lam, rho, C, len(C))
        rho = update_weights.increasing_schedule_update(rho, gamma)

        iter += 1

    # return trajectory
    return X, U
