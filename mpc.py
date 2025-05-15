#MAE271D Project
# MPC

from algames import ALGAMES
from dynamics import get_linear_dynamics
import initial_guess

from scipy import linalg
import numpy as np
import matplotlib as plt


# M players
M = 2

# N horizon length
N = 10

# n state size
n = 4  # (x, y, v_x, v_y)

# m control input size
m = 2  # (a_x, a_y)

# simulation length
sim_length = 10

# timestep
dt = 0.1 

# simulation steps
L = int(sim_length/dt)

# collision avoidance radius
r = 0.5

# running cost matrix
Q = np.eye(n)

# terminal cost matrix
Qf = np.eye(n)

# control cost matrix
R = np.eye(m)

# get linear dynamics
A, B = get_linear_dynamics(n, m, dt)

# initial state
x0 = [np.zeros(n)]*n
y0, X, U, mu = initial_guess.generate(x0, M, N, n, m, dt)
list_x0 = [np.zeros((n,))]*M
list_xf = [np.array([3, 3, 0.2, 0.2])]*M
rho = 1
gamma = 1
eps = 1e-5
constraint_wall_y = 6
constraint_u_x_max = 5
constraint_u_y_max = 5


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
        )

print(X)
print(U)
print("done")








# x = np.zeros((n, N+1))
# u = np.zeros((m, N))
# #x[0] = list_x0

# q = 0
# #for q in range(L):
# x_cur = x[q]

# # new initial state = state from applying u in previous timestep
# list_x0 = x_cur

# # find u
# X, U = ALGAMES(
#     M,  # number players
#     N,  # horizon
#     dt,  # timestep
#     r,  # collision avoidance radius
#     list_x0,  # list of initial states
#     list_xf,  # list of target states
#     Q,  # running cost matrix
#     Qf,  # terminal cost matrix
#     R,  # control cost matrix
#     rho,  # constraint penalty value
#     gamma,  # constraint penalty schedule
#     eps,  # convergence tolerance
#     constraint_wall_y,  # y value of horizontal wall
#     constraint_u_x_max,  # control input x bound
#     constraint_u_y_max,  # control input y bound
#     )
# # pull out first u
# U_1 = U[0]

# # update state using first control input u 
# # x[k+1] = A x[k] + B u[k]
# x_k_1 = A @ x_cur + B @ U_1

# #    x[q+1] = x_k_1



# # solve optimal control problem to get optimal ut
# # simulate - apply first control input 



# # plot car trajectory
# # plt.plot()
# # plt.plot(x[:,0])
# # plt.plot(x[:,1])
# # plt.xlabel("X")
# # plt.ylabel("Y")
# # plt.legend()
# # plt.show()

# # plot control input
