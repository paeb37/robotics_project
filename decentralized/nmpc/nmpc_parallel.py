"""
Collision avoidance using Nonlinear Model-Predictive Control

author: Ashwin Bose (atb033@github.com)
"""

from utils.multi_robot_plot import plot_robot_and_obstacles
from utils.create_obstacles import create_obstacles
import numpy as np
from scipy.optimize import minimize, Bounds
import time
from numba import cuda
import math


SIM_TIME = 20.
TIMESTEP = 0.1
NUMBER_OF_TIMESTEPS = int(SIM_TIME/TIMESTEP)
ROBOT_RADIUS = 2
VMAX = 10
VMIN = 0.2

# collision cost parameters
Qc = 5.
kappa = 4.

# nmpc parameters
HORIZON_LENGTH = int(4)
NMPC_TIMESTEP = 0.3
upper_bound = [(1/np.sqrt(2)) * VMAX] * HORIZON_LENGTH * 2
lower_bound = [-(1/np.sqrt(2)) * VMAX] * HORIZON_LENGTH * 2


def simulate(filename):
    print("In here uiefdjkn")
    obstacles = create_obstacles(SIM_TIME, NUMBER_OF_TIMESTEPS)

    start = np.array([50, 50])
    p_desired = np.array([5, 5])

    robot_state = start
    robot_state_history = np.empty((4, NUMBER_OF_TIMESTEPS))

    for i in range(NUMBER_OF_TIMESTEPS):
        # Predict the obstacles' position in future
        obstacle_predictions = np.array(predict_obstacle_positions(obstacles[:, i, :]))
        xref = compute_xref(robot_state, p_desired, HORIZON_LENGTH, NMPC_TIMESTEP)

        # Compute velocity using NMPC
        vel, velocity_profile = compute_velocity(robot_state, obstacle_predictions, xref)
        robot_state = update_state(robot_state, vel, TIMESTEP)
        robot_state_history[:2, i] = robot_state

    # Plot results
    plot_robot_and_obstacles(
        robot_state_history, obstacles, ROBOT_RADIUS, NUMBER_OF_TIMESTEPS, SIM_TIME, filename
    )

def compute_velocity(robot_state, obstacle_predictions, xref):
    """
    Computes control velocity of the copter
    """
    # u0 = np.array([0] * 2 * HORIZON_LENGTH)
    u0 = np.random.rand(2*HORIZON_LENGTH)
    def cost_fn(u): return total_cost(
        u, robot_state, obstacle_predictions, xref)

    bounds = Bounds(lower_bound, upper_bound)

    res = minimize(cost_fn, u0, method='SLSQP', bounds=bounds)
    velocity = res.x[:2]
    return velocity, res.x


def compute_xref(start, goal, number_of_steps, timestep):
    dir_vec = (goal - start)
    norm = np.linalg.norm(dir_vec)
    if norm < 0.1:
        new_goal = start
    else:
        dir_vec = dir_vec / norm
        new_goal = start + dir_vec * VMAX * timestep * number_of_steps
    return np.linspace(start, new_goal, number_of_steps).reshape((2*number_of_steps))


def total_cost(u, robot_state, obstacle_predictions, xref):
    """
    Total cost function incorporating tracking cost and collision cost,
    updated for CUDA.
    """
    x_robot = update_state(robot_state, u, NMPC_TIMESTEP)
    c1 = tracking_cost(x_robot, xref)
    
    # Launch GPU collision cost computation
    num_obstacles = obstacle_predictions.shape[0]
    threads_per_block = (16, 16)
    blocks_per_grid = (
        math.ceil(HORIZON_LENGTH / threads_per_block[0]),
        math.ceil(num_obstacles / threads_per_block[1]),
    )
    
    total_cost_array = cuda.to_device(np.zeros(1, dtype=np.float64))  # Device array to store the result

    # Call the CUDA kernel
    total_collision_cost[blocks_per_grid, threads_per_block](x_robot, obstacle_predictions, total_cost_array)

    # Retrieve the result from the device
    c2 = total_cost_array.copy_to_host()[0]

    # Combine tracking and collision costs
    total = c1 + c2
    return total


def tracking_cost(x, xref):
    return np.linalg.norm(x-xref)


@cuda.jit
def total_collision_cost(robot, obstacles, total_cost_array):
    """
    Parallelized computation of total collision cost between a robot and obstacles using CUDA.
    """
    i, j = cuda.grid(2)  # Get 2D grid indices
    HORIZON_LENGTH, num_obstacles = robot.shape[0] // 2, obstacles.shape[0]

    if i < HORIZON_LENGTH and j < num_obstacles:
        rob = robot[2 * i: 2 * i + 2]
        obs = obstacles[j, 2 * i: 2 * i + 2]
        cost = collision_cost(rob, obs)
        cuda.atomic.add(total_cost_array, 0, cost)

@cuda.jit(device=True)
def collision_cost(x0, x1):
    """
    Cost of collision between two robot states for CUDA kernel.
    """
    d = math.sqrt((x0[0] - x1[0]) ** 2 + (x0[1] - x1[1]) ** 2)  # Avoid np.linalg.norm
    return Qc / (1 + math.exp(kappa * (d - 2 * ROBOT_RADIUS)))


def predict_obstacle_positions(obstacles):
    obstacle_predictions = []
    for i in range(np.shape(obstacles)[1]):
        obstacle = obstacles[:, i]
        obstacle_position = obstacle[:2]
        obstacle_vel = obstacle[2:]
        u = np.vstack([np.eye(2)] * HORIZON_LENGTH) @ obstacle_vel
        obstacle_prediction = update_state(obstacle_position, u, NMPC_TIMESTEP)
        obstacle_predictions.append(obstacle_prediction)
    return obstacle_predictions


def update_state(x0, u, timestep):
    """
    Computes the states of the system after applying a sequence of control signals u on
    initial state x0
    """
    N = int(len(u) / 2)
    lower_triangular_ones_matrix = np.tril(np.ones((N, N)))
    kron = np.kron(lower_triangular_ones_matrix, np.eye(2))

    new_state = np.vstack([np.eye(2)] * int(N)) @ x0 + kron @ u * timestep

    return new_state
