# MAE 271D Project
# ALGAMES implementation

import numpy as np
from scipy import linalg

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
):
    # double integrator model
    n = 4  # state size (x, y, v_x, v_y)
    m = 2  # control input size (a_x, a_y)

    # create initial guess

    # define dynamics

    # define constraints

    # define derivative of augmented lagrangian

    # ALGAMES loop

    # solve G

    # dual ascent penalty update

    # return trajectory
