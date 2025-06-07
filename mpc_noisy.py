#MAE271D Project
# MPC

from algames import ALGAMES
import numpy as np


def MPC_noisy(    
    M,  # number players
    N,  # horizon
    n,  # state size
    m,  # control input size
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
    mean,       # mean
    sigma,    # 
    max_iter=10,  # maximum number of iterations
    dynamics_mult=1000,  # multiplier on dynamics in root finding problem
):

        U_mpc = np.zeros((M,m))
        # store initial conditions
        X_mpc = np.reshape(np.array(list_x0), (M,n))

        # store all trajectories over horizon
        X_hist = np.empty((M*N*n))

        for i in range (N):

                X, U = ALGAMES(
                        M,  # number players
                        N,  # horizon
                        dt,  # timestep
                        r,  # collision avoidance radius
                        list_x0,  # list of initial states
                        list_xf,  # list of target states
                        Q,  # running cost matrix
                        Qf,  # terminal cost matrix
                        R,  # control cost matrix
                        rho,  # constraint penalty value
                        gamma,  # constraint penalty schedule
                        eps,  # convergence tolerance
                        constraint_wall_y,  # y value of horizontal wall
                        constraint_u_x_max,  # control input x bound
                        constraint_u_y_max,  # control input y bound
                        max_iter,
                        dynamics_mult,
                )
                
                # # generate noise to perturb state
                # noise = np.random.normal( mean, sigma, n)

                x_players = np.split(X, M)

                # pull out first state X for next iteration's initial condition
                list_x0_k1 = []
                for i, x in enumerate(x_players):
                        x1 = x[0:n]
                        list_x0_k1.append(x1)    
                        # generate noise to perturb state
                        noise = np.random.normal( mean, sigma, n)
                        #TODO: should check here that the perturbation doesnt cause a cola violation
                # update initial conditions of next iteration to the state at the first step of this trajectory
                list_x0 = list_x0_k1 + noise

                # store trajectory
                X_mpc = np.hstack((X_mpc, np.array(list_x0_k1)))

                # pull out initial input U
                list_u = []
                u_players = np.split(U, M)
                for i, u in enumerate(u_players):
                        u1 = u[0:m]
                        list_u.append(u1)

                # store control inputs
                U_mpc = np.hstack((U_mpc, np.array(list_u)))

                X_hist = np.vstack((X_hist, X))

        # return mpc trajectory
        return X_mpc, U_mpc, X_hist