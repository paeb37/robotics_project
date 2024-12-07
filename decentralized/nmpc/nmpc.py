"""
Integrated multi-robot NMPC with GPU-accelerated obstacle collision cost for obstacles.
Robot-robot collisions remain on CPU, while robot-obstacle collisions are computed in parallel on the GPU.
"""

import random
from utils.multi_robot_plot import plot_robots_and_obstacles
from utils.create_obstacles import create_obstacles
import numpy as np
from scipy.optimize import minimize, Bounds
import time

### GPU INTEGRATION
from numba import cuda
import math

SIM_TIME = 8.0
TIMESTEP = 0.1
NUMBER_OF_TIMESTEPS = int(SIM_TIME / TIMESTEP)
ROBOT_RADIUS = 0.5
VMAX = 2
VMIN = 0.2

# collision cost parameters
Qc = 5.0
kappa = 4.0

# nmpc parameters and constraints
HORIZON_LENGTH = int(4)
NMPC_TIMESTEP = 0.3
upper_bound = [(1 / np.sqrt(2)) * VMAX] * HORIZON_LENGTH * 2
lower_bound = [-(1 / np.sqrt(2)) * VMAX] * HORIZON_LENGTH * 2
NUM_STATES = 2


def simulate(filename):
    num_obstacles = random.randint(1, 7)
    obstacles = create_obstacles(SIM_TIME, NUMBER_OF_TIMESTEPS, num_obstacles)

    # Initialize multiple robots
    num_robots = 3
    robots = []  # robot states
    goals = []  # robot goals

    for _ in range(num_robots):
        start = np.array([np.random.uniform(0, 10), np.random.uniform(0, 10)])
        goal = np.array([np.random.uniform(0, 10), np.random.uniform(0, 10)])
        robots.append(start)
        goals.append(goal)

    robots_state_history = np.empty((NUM_STATES, NUMBER_OF_TIMESTEPS, num_robots))

    for i in range(NUMBER_OF_TIMESTEPS):
        obstacle_predictions = predict_obstacle_positions(obstacles[:, i, :])

        for j, robot in enumerate(robots):
            xref = compute_xref(robot, goals[j], HORIZON_LENGTH, NMPC_TIMESTEP)
            vel = compute_velocity(robot, obstacle_predictions, xref, robots, j)[0]
            robots[j] = update_state(robot, vel, TIMESTEP)
            robots_state_history[:2, i, j] = robots[j]

    plot_robots_and_obstacles(
        robots_state_history,
        obstacles,
        goals,
        ROBOT_RADIUS,
        NUMBER_OF_TIMESTEPS,
        SIM_TIME,
        filename,
    )


def compute_velocity(robot_state, obstacle_predictions, xref, robots, robot_index):
    u0 = np.random.rand(2 * HORIZON_LENGTH)

    def cost_fn(u):
        return total_cost(
            u, robot_state, obstacle_predictions, xref, robots, robot_index
        )

    bounds = Bounds(lower_bound, upper_bound)

    res = minimize(cost_fn, u0, method="SLSQP", bounds=bounds)
    velocity = res.x[:2]
    return velocity, res.x


def compute_xref(start, goal, number_of_steps, timestep):
    dir_vec = goal - start
    norm = np.linalg.norm(dir_vec)
    if norm < 0.1:
        new_goal = start
    else:
        dir_vec = dir_vec / norm
        new_goal = start + dir_vec * VMAX * timestep * number_of_steps
    return np.linspace(start, new_goal, number_of_steps).reshape((2 * number_of_steps))


def total_cost(u, robot_state, obstacle_predictions, xref, robots, robot_index):
    x_robot = update_state(robot_state, u, NMPC_TIMESTEP)
    c1 = tracking_cost(x_robot, xref)

    # GPU integration for robot-obstacle collisions
    # obstacle_predictions is currently a list of arrays, one per obstacle.
    # We must have obstacle_predictions as a np.array of shape: (num_obstacles, 2*HORIZON_LENGTH)
    if len(obstacle_predictions) > 0:
        obstacle_predictions_array = np.vstack(obstacle_predictions)
    else:
        # If no obstacles, collision cost is zero.
        obstacle_predictions_array = np.zeros((0, 2 * HORIZON_LENGTH))

    c2_obstacles = 0.0
    if obstacle_predictions_array.size > 0:
        c2_obstacles = gpu_obstacle_collision_cost(x_robot, obstacle_predictions_array)

    # CPU-based robot-robot collision cost
    c2_robots = robot_robot_collision_cost(x_robot, robots, robot_index)

    total = c1 + c2_obstacles + c2_robots
    return total


def tracking_cost(x, xref):
    return np.linalg.norm(x - xref)


def robot_robot_collision_cost(robot, robots, robot_index):
    """
    Computes collision cost with other robots on CPU.
    This remains unchanged from the original CPU version.
    """
    total_cost_val = 0.0
    for i in range(HORIZON_LENGTH):
        rob = robot[2 * i : 2 * i + 2]
        for j, other_robot in enumerate(robots):
            if j != robot_index:
                other_rob = update_state(
                    other_robot, np.zeros(2 * HORIZON_LENGTH), NMPC_TIMESTEP
                )
                # Just predict other robot's position if needed, here we assume static for demonstration
                # In a more realistic scenario, you'd have their predicted trajectories as well.
                other_rob_pos = other_rob[2 * i : 2 * i + 2]
                r_th = 2.5 * ROBOT_RADIUS
                r_min = 2 * ROBOT_RADIUS
                if len(other_rob_pos) > 0:
                    total_cost_val += robot_collision_cost(
                        rob, other_rob_pos, r_th, r_min
                    )
    return total_cost_val


def robot_collision_cost(robot1, robot2, r_th, r_min):
    d = np.linalg.norm(robot1 - robot2)
    if d > r_th:
        return 0
    else:
        return Qc / (1 + np.exp(kappa * (d - r_th)))


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
    N = int(len(u) / 2)
    lower_triangular_ones_matrix = np.tril(np.ones((N, N)))
    kron = np.kron(lower_triangular_ones_matrix, np.eye(2))
    new_state = np.vstack([np.eye(2)] * int(N)) @ x0 + kron @ u * timestep
    return new_state


### GPU INTEGRATION: Device and Kernel functions


@cuda.jit(device=True)
def gpu_collision_cost(x0, x1, Qc, kappa, robot_radius):
    """
    GPU device function for obstacle collision cost.
    Using a logistic-like function as in the given CUDA code.
    """
    dx = x0[0] - x1[0]
    dy = x0[1] - x1[1]
    d = math.sqrt(dx * dx + dy * dy)
    # Using the logistic function from the single-agent code:
    return Qc / (1.0 + math.exp(kappa * (d - 2.0 * robot_radius)))


@cuda.jit
def gpu_total_obstacle_collision_cost(
    robot, obstacles, Qc, kappa, robot_radius, total_cost_array
):
    """
    GPU kernel to sum obstacle collision costs over horizon steps and obstacles.
    robot: 1D array, length = 2*HORIZON_LENGTH
    obstacles: 2D array, shape = (num_obstacles, 2*HORIZON_LENGTH)
    """
    i, j = cuda.grid(2)
    H = robot.shape[0] // 2
    num_obstacles = obstacles.shape[0]

    if i < H and j < num_obstacles:
        # Extract robot and obstacle position at step i
        rob_x = robot[2 * i]
        rob_y = robot[2 * i + 1]
        obs_x = obstacles[j, 2 * i]
        obs_y = obstacles[j, 2 * i + 1]
        x0 = (rob_x, rob_y)
        x1 = (obs_x, obs_y)

        cost = gpu_collision_cost(x0, x1, Qc, kappa, robot_radius)
        cuda.atomic.add(total_cost_array, 0, cost)


def gpu_obstacle_collision_cost(x_robot, obstacle_predictions):
    """
    Host function to run the GPU kernel and return total obstacle collision cost.
    x_robot: shape (2*HORIZON_LENGTH,)
    obstacle_predictions: shape (num_obstacles, 2*HORIZON_LENGTH)
    """
    H = HORIZON_LENGTH
    num_obstacles = obstacle_predictions.shape[0]

    # Copy data to device
    d_robot = cuda.to_device(x_robot)
    d_obstacles = cuda.to_device(obstacle_predictions)
    d_total_cost = cuda.to_device(np.array([0.0], dtype=np.float64))

    # Threads and blocks
    threads_per_block = (16, 16)
    blocks_per_grid = (
        math.ceil(H / threads_per_block[0]),
        math.ceil(num_obstacles / threads_per_block[1]),
    )

    # Launch kernel
    gpu_total_obstacle_collision_cost[blocks_per_grid, threads_per_block](
        d_robot, d_obstacles, Qc, kappa, ROBOT_RADIUS, d_total_cost
    )

    # Copy result back to host
    c2 = d_total_cost.copy_to_host()[0]
    return c2
