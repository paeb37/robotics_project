"""
Collision avoidance using Nonlinear Model-Predictive Control

author: Ashwin Bose (atb033@github.com)
"""
from utils.multi_robot_plot import plot_robot_and_obstacles
from utils.create_obstacles import create_obstacles
import numpy as np
import numpy as np
from numba import cuda
import math
from scipy.optimize import minimize, Bounds
import time

SIM_TIME = 8.
TIMESTEP = 0.1
NUMBER_OF_TIMESTEPS = int(SIM_TIME/TIMESTEP)
ROBOT_RADIUS = 0.5
VMAX = 2
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
    obstacles = create_obstacles(SIM_TIME, NUMBER_OF_TIMESTEPS)

    start = np.array([5, 5])
    p_desired = np.array([5, 5])  # Changed to a different goal point

    robot_state = start
    robot_state_history = np.zeros((NUMBER_OF_TIMESTEPS, 2))

    start_time = time.time()

    for i in range(NUMBER_OF_TIMESTEPS):
        # predict the obstacles' position in future
        obstacle_predictions = predict_obstacle_positions(obstacles[:, i, :])
        xref = compute_xref(robot_state, p_desired,
                            HORIZON_LENGTH, NMPC_TIMESTEP)
        
        # compute velocity using nmpc
        vel, velocity_profile = compute_velocity(
            robot_state, obstacle_predictions, xref)
        
        # Update robot state directly with the velocity
        robot_state = robot_state + vel * TIMESTEP
        robot_state_history[i] = robot_state

    end_time = time.time()
    sim_time = end_time - start_time
    print(f"Simulation time: {sim_time:.2f} seconds")

    plot_robot_and_obstacles(
        robot_state_history.T, obstacles, ROBOT_RADIUS, NUMBER_OF_TIMESTEPS, SIM_TIME, filename)


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
    x_robot = update_state(robot_state, u, NMPC_TIMESTEP)
    c1 = tracking_cost(x_robot, xref)
    c2 = total_collision_cost(x_robot, obstacle_predictions)
    total = c1 + c2
    return total


def tracking_cost(x, xref):
    return np.linalg.norm(x-xref)


def total_collision_cost(robot, obstacles):
    total_cost = 0
    for i in range(HORIZON_LENGTH):
        for j in range(len(obstacles)):
            obstacle = obstacles[j]
            rob = robot[2 * i: 2 * i + 2]
            obs = obstacle[2 * i: 2 * i + 2]
            total_cost += collision_cost(rob, obs)
    return total_cost


def collision_cost(x0, x1):
    """
    Cost of collision between two robot_state
    """
    d = np.linalg.norm(x0 - x1)
    cost = Qc / (1 + np.exp(kappa * (d - 2*ROBOT_RADIUS)))
    return cost

@cuda.jit
def predict_obstacle_kernel(d_obstacles, d_obstacle_predictions):
    # Get thread index
    i = cuda.grid(1)
    
    # Ensure thread is within array bounds
    if i < d_obstacles.shape[1]:
        # Extract obstacle data for this thread
        obstacle_x = d_obstacles[0, i]
        obstacle_y = d_obstacles[1, i]
        vel_x = d_obstacles[2, i]
        vel_y = d_obstacles[3, i]
        
        # Predict obstacle positions
        for j in range(HORIZON_LENGTH):
            # Update position using simple linear motion model
            obstacle_x += vel_x * NMPC_TIMESTEP
            obstacle_y += vel_y * NMPC_TIMESTEP
            
            # Store prediction
            d_obstacle_predictions[i, j, 0] = obstacle_x
            d_obstacle_predictions[i, j, 1] = obstacle_y

def predict_obstacle_positions(obstacles):
    """
    Parallelize obstacle position prediction using CUDA
    
    :param obstacles: Input obstacle array (shape: [4, num_obstacles])
    :return: Array of predicted obstacle positions
    """
    obstacles = np.ascontiguousarray(obstacles, dtype=np.float32)
    
    threads_per_block = 256
    blocks_per_grid = (obstacles.shape[1] + threads_per_block - 1) // threads_per_block
    
    d_obstacle_predictions = cuda.device_array(
        (obstacles.shape[1], HORIZON_LENGTH, 2), 
        dtype=np.float32
    )
    
    d_obstacles = cuda.to_device(obstacles)
    
    predict_obstacle_kernel[blocks_per_grid, threads_per_block](
        d_obstacles,
        d_obstacle_predictions
    )
    
    return d_obstacle_predictions.copy_to_host()

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