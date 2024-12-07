"""
Collision avoidance using Nonlinear Model-Predictive Control

author: Ashwin Bose (atb033@github.com)
"""
from utils.multi_robot_plot import plot_robot_and_obstacles
from utils.create_obstacles import create_obstacles
import numpy as np
from numba import cuda
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
    p_desired = np.array([7, 7])

    robot_state = start
    robot_state_history = np.empty((4, NUMBER_OF_TIMESTEPS))

    start_time = time.time()

    for i in range(NUMBER_OF_TIMESTEPS):
        # predict the obstacles' position in future
        obstacle_predictions = predict_obstacle_positions(obstacles[:, i, :])
        xref = compute_xref(robot_state, p_desired,
                            HORIZON_LENGTH, NMPC_TIMESTEP)
        # compute velocity using nmpc
        vel, velocity_profile = compute_velocity(
            robot_state, obstacle_predictions, xref)
        robot_state = update_state_host(robot_state, vel, TIMESTEP)
        robot_state_history[:2, i] = robot_state

    end_time = time.time()
    sim_time = end_time - start_time
    print(f"Simulation time: {sim_time:.2f} seconds")

    plot_robot_and_obstacles(
        robot_state_history, obstacles, ROBOT_RADIUS, NUMBER_OF_TIMESTEPS, SIM_TIME, filename)


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
    x_robot = update_state_host(robot_state, u, NMPC_TIMESTEP)
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
def update_state_cuda(x0, u, timestep, new_state):
    N = u.shape[0] // 2
    idx = cuda.grid(1)  # Get the thread index
    if idx < N:  # Ensure we don't go out of bounds
        lower_triangular_ones_matrix = cuda.to_device(np.tril(np.ones((N, N))))
        kron = cuda.to_device(np.kron(lower_triangular_ones_matrix, np.eye(2)))

        new_state[idx, :2] = (np.eye(2) @ x0 + kron @ u * timestep)[:2]

def update_state_host(x0, u, timestep):
    N = len(u) // 2
    new_state = np.zeros((HORIZON_LENGTH, 2))  # Adjust shape as needed
    d_x0 = cuda.to_device(x0)
    d_u = cuda.to_device(u)
    d_new_state = cuda.to_device(new_state)

    threads_per_block = 32
    blocks_per_grid = (N + (threads_per_block - 1)) // threads_per_block
    update_state_cuda[blocks_per_grid, threads_per_block](d_x0, d_u, timestep, d_new_state)

    return d_new_state.copy_to_host()  # Copy results back to host

@cuda.jit
def predict_obstacle_positions_cuda(obstacles, obstacle_predictions, NMPC_TIMESTEP):
    idx = cuda.grid(1)  # Get the thread index
    if idx < obstacles.shape[1]:  # Ensure we don't go out of bounds
        obstacle = obstacles[:, idx]
        obstacle_position = obstacle[:2]
        obstacle_vel = obstacle[2:]

        u = np.zeros((HORIZON_LENGTH * 2,))  # Adjust shape as needed
        for i in range(HORIZON_LENGTH):
            u[2 * i: 2 * i + 2] = obstacle_vel
        # u = cuda.to_device(np.vstack([np.eye(2)] * HORIZON_LENGTH) @ obstacle_vel)

        new_state = cuda.device_array((HORIZON_LENGTH, 2))  # Adjust shape as needed
        update_state_cuda[(HORIZON_LENGTH, 1)](obstacle_position, u, NMPC_TIMESTEP, new_state)

        obstacle_predictions[idx] = new_state

def predict_obstacle_positions(obstacles: np.ndarray) -> np.ndarray:
    obstacles = np.ascontiguousarray(obstacles)
    obstacle_predictions = cuda.device_array((HORIZON_LENGTH, obstacles.shape[1], 2))  # Adjust shape as needed
    d_obstacles = cuda.to_device(obstacles)
    threads_per_block = 32
    blocks_per_grid = (obstacles.shape[1] + (threads_per_block - 1)) // threads_per_block
    predict_obstacle_positions_cuda[blocks_per_grid, threads_per_block](d_obstacles, obstacle_predictions, NMPC_TIMESTEP)
    return obstacle_predictions.copy_to_host()  # Copy results back to host
