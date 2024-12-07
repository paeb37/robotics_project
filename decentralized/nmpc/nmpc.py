"""
Collision avoidance using Nonlinear Model-Predictive Control

author: Ashwin Bose (atb033@github.com)
"""

from utils.multi_robot_plot import plot_robots_and_obstacles
from utils.create_obstacles import create_obstacles
import numpy as np
from scipy.optimize import minimize, Bounds
import time

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


def simulate(filename):
    obstacles = create_obstacles(SIM_TIME, NUMBER_OF_TIMESTEPS)

    # Initialize multiple robots with their start positions and goals
    start1, goal1 = np.array([5, 5]), np.array([0, 0])
    start2, goal2 = np.array([0, 0]), np.array([10, 10])
    start3, goal3 = np.array([10, 0]), np.array([0, 10])

    robots = [start1, start2, start3]  # List of robot states
    goals = [goal1, goal2, goal3]  # List of robot goals
    num_robots = len(robots)

    # Modified to store history for multiple robots
    # 4 states: x, y, vx, vy
    robots_state_history = np.empty((num_robots, 4, NUMBER_OF_TIMESTEPS))

    for i in range(NUMBER_OF_TIMESTEPS):
        # Predict the obstacles' position in future
        obstacle_predictions = predict_obstacle_positions(obstacles[:, i, :])

        # For each robot, compute its own velocity using decentralized NMPC
        for j, robot in enumerate(robots):
            xref = compute_xref(robot, goals[j], HORIZON_LENGTH, NMPC_TIMESTEP)
            vel = compute_velocity(robot, obstacle_predictions, xref, robots)[0]
            robots[j] = update_state(robot, vel, TIMESTEP)
            robots_state_history[j, :2, i] = robots[j]

    plot_robots_and_obstacles(
        robots_state_history,
        obstacles,
        ROBOT_RADIUS,
        NUMBER_OF_TIMESTEPS,
        SIM_TIME,
        filename,
    )


def compute_velocity(robot_state, obstacle_predictions, xref, robots):
    """
    Computes control velocity of the robot using decentralized NMPC
    """
    u0 = np.random.rand(2 * HORIZON_LENGTH)

    def cost_fn(u):
        return total_cost(u, robot_state, obstacle_predictions, xref, robots)

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


def total_cost(u, robot_state, obstacle_predictions, xref, robots):
    x_robot = update_state(robot_state, u, NMPC_TIMESTEP)
    c1 = tracking_cost(x_robot, xref)
    c2 = total_collision_cost(x_robot, obstacle_predictions, robots)
    total = c1 + c2
    return total


def tracking_cost(x, xref):
    return np.linalg.norm(x - xref)


def total_collision_cost(robot, obstacles, robots):
    total_cost = 0
    for i in range(HORIZON_LENGTH):
        for j in range(len(obstacles)):
            obstacle = obstacles[j]
            rob = robot[2 * i : 2 * i + 2]
            obs = obstacle[2 * i : 2 * i + 2]
            total_cost += collision_cost(rob, obs)
        # Add collision cost with nearby robots
        # for other_robot in robots:
        #     if not np.array_equal(
        #         other_robot, robot[:2]
        #     ):  # Don't consider self-collision
        #         total_cost += collision_cost(robot[2 * i : 2 * i + 2], other_robot)
    return total_cost


def collision_cost(x0, x1):
    """
    Cost of collision between two robot_state
    """
    d = np.linalg.norm(x0 - x1)
    cost = Qc / (1 + np.exp(kappa * (d - 2 * ROBOT_RADIUS)))
    return cost


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
