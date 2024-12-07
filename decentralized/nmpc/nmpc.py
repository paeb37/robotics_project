"""
Collision avoidance using Nonlinear Model-Predictive Control
with partial CUDA acceleration for obstacle collision costs.

author: (Based on code by Ashwin Bose and integrated GPU changes)
"""

import random
import numpy as np
from scipy.optimize import minimize, Bounds
import time
from numba import cuda
import math

from utils.multi_robot_plot import plot_robots_and_obstacles
from utils.create_obstacles import create_obstacles

# Simulation parameters
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
HORIZON_LENGTH = 4
NMPC_TIMESTEP = 0.3
upper_bound = [(1 / np.sqrt(2)) * VMAX] * HORIZON_LENGTH * 2
lower_bound = [-(1 / np.sqrt(2)) * VMAX] * HORIZON_LENGTH * 2
NUM_STATES = 2


def simulate(filename):
    # Generate obstacles
    num_obstacles = random.randint(1, 7)
    obstacles = create_obstacles(SIM_TIME, NUMBER_OF_TIMESTEPS, num_obstacles)

    # Initialize multiple robots with their start positions and goals
    num_robots = 3  # Adjust as needed
    robots = []
    goals = []

    # Generate random start and goal positions
    for _ in range(num_robots):
        start = np.array([np.random.uniform(0, 10), np.random.uniform(0, 10)])
        goal = np.array([np.random.uniform(0, 10), np.random.uniform(0, 10)])
        robots.append(start)
        goals.append(goal)

    robots_state_history = np.empty((NUM_STATES, NUMBER_OF_TIMESTEPS, num_robots))

    for i in range(NUMBER_OF_TIMESTEPS):
        # Obtain predicted obstacle positions for the current time step
        # obstacles[:, i, :] shape: (num_obstacles, 4) = [x, y, vx, vy] at time i
        obstacle_predictions = predict_obstacle_positions(obstacles[:, i, :])

        # Compute control for each robot
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
    """
    Compute control velocity using decentralized NMPC.
    """
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
    c2 = total_collision_cost(x_robot, obstacle_predictions, robots, robot_index)
    return c1 + c2


def tracking_cost(x, xref):
    return np.linalg.norm(x - xref)


def total_collision_cost(robot, obstacles, robots, robot_index):
    """
    Compute total collision cost with both obstacles and robots.
    We'll split the computations:
    - GPU: obstacle collision cost
    - CPU: robot-robot collision cost

    `robot` shape: (2 * HORIZON_LENGTH,)
    `obstacles` shape: (num_obstacles, 2 * HORIZON_LENGTH)

    Robot-to-robot collision cost remains CPU-based for simplicity.
    """
    # First compute obstacle collision cost on GPU
    num_obstacles = obstacles.shape[0]

    if num_obstacles > 0:
        # Prepare GPU arrays
        robot_device = cuda.to_device(robot)
        obstacles_device = cuda.to_device(obstacles)
        total_cost_device = cuda.to_device(np.zeros(1, dtype=np.float64))

        # Determine thread/block layout
        threads_per_block = (16, 16)
        blocks_per_grid = (
            math.ceil(HORIZON_LENGTH / threads_per_block[0]),
            math.ceil(num_obstacles / threads_per_block[1]),
        )

        # Launch kernel
        compute_obstacle_collision_cost[blocks_per_grid, threads_per_block](
            robot_device, obstacles_device, total_cost_device
        )

        # Copy result back
        c2_obstacles = total_cost_device.copy_to_host()[0]
    else:
        c2_obstacles = 0.0

    # Now compute robot-robot collision cost on CPU
    c2_robots = 0.0
    for i in range(HORIZON_LENGTH):
        rob = robot[2 * i : 2 * i + 2]
        for j, other_robot in enumerate(robots):
            if j != robot_index:
                # other_robot might be just current position or we need predicted?
                # The current code tries to access other_robot states as if it were
                # also a vector over horizon. But currently other_robot is just [x, y].
                # We must note that we have not computed a predicted trajectory
                # for other robots. In a fully decentralized setting, you might
                # also predict their future states. For simplicity, consider them static at their current positions.
                # If you want to reflect their future positions, you'd need their control inputs too.
                # We'll keep them as currently implemented: this might be a logical discrepancy,
                # but it's how original code was structured.

                # If other_robot is only current position [x, y], we replicate it over horizon:
                other_rob = np.tile(other_robot, HORIZON_LENGTH)
                other_rob = other_rob[2 * i : 2 * i + 2]

                r_th = 2.5 * ROBOT_RADIUS
                r_min = 2 * ROBOT_RADIUS
                c2_robots += robot_collision_cost(rob, other_rob, r_th, r_min)

    return c2_obstacles + c2_robots


def robot_collision_cost(robot1, robot2, r_th, r_min):
    d = np.linalg.norm(robot1 - robot2)
    if d > r_th:
        return 0
    else:
        return Qc / (1 + np.exp(kappa * (d - r_th)))


def predict_obstacle_positions(obstacles_at_t):
    """
    Predict obstacle positions for the next HORIZON_LENGTH steps.
    obstacles_at_t shape: (num_obstacles, 4) = [x, y, vx, vy]
    We'll return a (num_obstacles, 2*HORIZON_LENGTH) array for GPU consumption.
    """
    num_obstacles = obstacles_at_t.shape[0]
    obstacle_predictions = np.zeros((num_obstacles, 2 * HORIZON_LENGTH))
    for j in range(num_obstacles):
        obstacle = obstacles_at_t[j]  # [x, y, vx, vy]
        obstacle_position = obstacle[:2]
        obstacle_vel = obstacle[2:]
        u = np.vstack([np.eye(2)] * HORIZON_LENGTH) @ obstacle_vel
        obstacle_prediction = update_state(obstacle_position, u, NMPC_TIMESTEP)
        obstacle_predictions[j, :] = obstacle_prediction.flatten()
    return obstacle_predictions


def update_state(x0, u, timestep):
    """
    Updates the state given initial state x0 and control inputs u over horizon.
    x0 shape: (2,)
    u shape: (2*N,) where N = HORIZON_LENGTH
    """
    N = int(len(u) / 2)
    lower_triangular_ones_matrix = np.tril(np.ones((N, N)))
    kron = np.kron(lower_triangular_ones_matrix, np.eye(2))
    new_state = np.vstack([np.eye(2)] * N) @ x0 + kron @ u * timestep
    return new_state


# ---------------- GPU Kernels -----------------
@cuda.jit
def compute_obstacle_collision_cost(robot, obstacles, total_cost_array):
    """
    GPU kernel to compute obstacle collision costs in parallel.
    robot: 1D array (2*HORIZON_LENGTH,)
    obstacles: 2D array (num_obstacles, 2*HORIZON_LENGTH)
    total_cost_array: 1D array [single float], for atomic add
    """
    i, j = cuda.grid(2)
    horizon = robot.size // 2  # since robot.size = 2*HORIZON_LENGTH
    num_obstacles = obstacles.shape[0]

    if i < horizon and j < num_obstacles:
        rob_x = robot[2 * i]
        rob_y = robot[2 * i + 1]
        obs_x = obstacles[j, 2 * i]
        obs_y = obstacles[j, 2 * i + 1]

        cost = device_collision_cost(rob_x, rob_y, obs_x, obs_y)
        cuda.atomic.add(total_cost_array, 0, cost)


@cuda.jit(device=True)
def device_collision_cost(r_x, r_y, o_x, o_y):
    """
    Compute obstacle collision cost using the safe-distance quadratic cost.
    This mirrors the original `collision_cost` function.
    """
    d = math.sqrt((r_x - o_x) ** 2 + (r_y - o_y) ** 2)
    safe_distance = 2.5 * ROBOT_RADIUS
    if d > safe_distance:
        return 0.0
    else:
        diff = (safe_distance - d) / safe_distance
        return Qc * diff * diff


if __name__ == "__main__":
    simulate("output.png")
