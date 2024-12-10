"""
This version of the script was used to generate benchmarking results on the GPU speedup for obstacle prediction and collision avoidance in the NMPC-based decentralized control system. The script uses the `numba.cuda` library to offload computations to the GPU and compare the performance with the CPU-based implementation.
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
import matplotlib.pyplot as plt

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

# Global timing storage
timings = {
    "predict_obstacles_gpu_kernel": [],
    "predict_obstacles_gpu_total": [],
    "predict_obstacles_cpu": [],
    "obstacle_collision_gpu_kernel": [],
    "obstacle_collision_gpu_total": [],
    "obstacle_collision_cpu": [],
}


def simulate(filename):
    num_obstacles = 4
    obstacles = create_obstacles(SIM_TIME, NUMBER_OF_TIMESTEPS, num_obstacles)
    num_robots = 3
    robots = []
    goals = []
    for _ in range(num_robots):
        start = np.array([np.random.uniform(0, 10), np.random.uniform(0, 10)])
        goal = np.array([np.random.uniform(0, 10), np.random.uniform(0, 10)])
        robots.append(start)
        goals.append(goal)
    robots_state_history = np.empty((NUM_STATES, NUMBER_OF_TIMESTEPS, num_robots))

    for i in range(NUMBER_OF_TIMESTEPS):
        # GPU-based predictions
        obstacle_predictions_gpu = predict_obstacle_positions(obstacles[:, i, :])
        obstacle_predictions_gpu = obstacle_predictions_gpu.reshape(
            (obstacle_predictions_gpu.shape[0], 2 * HORIZON_LENGTH)
        )

        # CPU-based predictions for comparison
        obstacle_predictions_cpu = predict_obstacle_positions_cpu(obstacles[:, i, :])
        obstacle_predictions_cpu = obstacle_predictions_cpu.reshape(
            (obstacle_predictions_cpu.shape[0], 2 * HORIZON_LENGTH)
        )

        for j, robot in enumerate(robots):
            xref = compute_xref(robot, goals[j], HORIZON_LENGTH, NMPC_TIMESTEP)
            vel = compute_velocity(robot, obstacle_predictions_gpu, xref, robots, j)[0]
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

    # GPU-based obstacle collision cost
    # We'll also time the GPU obstacle collision computation and compare to CPU.
    start_cpu = time.perf_counter()
    c2_obstacles_cpu = cpu_obstacle_collision_cost(x_robot, obstacle_predictions)
    end_cpu = time.perf_counter()
    timings["obstacle_collision_cpu"].append(end_cpu - start_cpu)

    start_gpu_total = time.perf_counter()
    c2_obstacles_gpu = gpu_obstacle_collision_cost(x_robot, obstacle_predictions)
    end_gpu_total = time.perf_counter()

    # Just timing the kernel itself separately (already done inside gpu_obstacle_collision_cost)
    # We will rely on measured times in gpu_obstacle_collision_cost function.

    # CPU-based robot-robot collision cost
    c2_robots = robot_robot_collision_cost(x_robot, robots, robot_index)

    total = c1 + c2_obstacles_gpu + c2_robots
    return total


def tracking_cost(x, xref):
    return np.linalg.norm(x - xref)


def robot_robot_collision_cost(robot, robots, robot_index):
    total_cost_val = 0.0
    for i in range(HORIZON_LENGTH):
        rob = robot[2 * i : 2 * i + 2]
        for j, other_robot in enumerate(robots):
            if j != robot_index:
                other_rob = update_state(
                    other_robot, np.zeros(2 * HORIZON_LENGTH), NMPC_TIMESTEP
                )
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


@cuda.jit
def predict_obstacle_kernel(d_obstacles, d_obstacle_predictions):
    i = cuda.grid(1)
    if i < d_obstacles.shape[1]:
        obstacle_x = d_obstacles[0, i]
        obstacle_y = d_obstacles[1, i]
        vel_x = d_obstacles[2, i]
        vel_y = d_obstacles[3, i]

        for j in range(HORIZON_LENGTH):
            obstacle_x += vel_x * NMPC_TIMESTEP
            obstacle_y += vel_y * NMPC_TIMESTEP
            d_obstacle_predictions[i, j, 0] = obstacle_x
            d_obstacle_predictions[i, j, 1] = obstacle_y


def predict_obstacle_positions(obstacles):
    obstacles = np.ascontiguousarray(obstacles, dtype=np.float32)

    start_total = time.perf_counter()

    threads_per_block = 256
    blocks_per_grid = (obstacles.shape[1] + threads_per_block - 1) // threads_per_block

    d_obstacle_predictions = cuda.device_array(
        (obstacles.shape[1], HORIZON_LENGTH, 2), dtype=np.float32
    )

    start_h2d = time.perf_counter()
    d_obstacles = cuda.to_device(obstacles)
    cuda.synchronize()
    end_h2d = time.perf_counter()

    # Time kernel
    start_kernel = time.perf_counter()
    predict_obstacle_kernel[blocks_per_grid, threads_per_block](
        d_obstacles, d_obstacle_predictions
    )
    cuda.synchronize()
    end_kernel = time.perf_counter()

    start_d2h = time.perf_counter()
    result = d_obstacle_predictions.copy_to_host()
    cuda.synchronize()
    end_d2h = time.perf_counter()

    end_total = time.perf_counter()

    timings["predict_obstacles_gpu_kernel"].append(end_kernel - start_kernel)
    timings["predict_obstacles_gpu_total"].append(end_total - start_total)

    return result


def predict_obstacle_positions_cpu(obstacles):
    # CPU equivalent to predict obstacle positions
    start = time.perf_counter()
    num_obs = obstacles.shape[1]
    pred = np.zeros((num_obs, HORIZON_LENGTH, 2), dtype=np.float32)
    for i in range(num_obs):
        obstacle_x = obstacles[0, i]
        obstacle_y = obstacles[1, i]
        vel_x = obstacles[2, i]
        vel_y = obstacles[3, i]
        for j in range(HORIZON_LENGTH):
            obstacle_x += vel_x * NMPC_TIMESTEP
            obstacle_y += vel_y * NMPC_TIMESTEP
            pred[i, j, 0] = obstacle_x
            pred[i, j, 1] = obstacle_y
    end = time.perf_counter()
    timings["predict_obstacles_cpu"].append(end - start)
    return pred


def update_state(x0, u, timestep):
    N = int(len(u) / 2)
    lower_triangular_ones_matrix = np.tril(np.ones((N, N)))
    kron = np.kron(lower_triangular_ones_matrix, np.eye(2))
    new_state = np.vstack([np.eye(2)] * int(N)) @ x0 + kron @ u * timestep
    return new_state


### GPU INTEGRATION: Device and Kernel functions


@cuda.jit(device=True)
def gpu_collision_cost(x0, x1, Qc, kappa, robot_radius):
    dx = x0[0] - x1[0]
    dy = x0[1] - x1[1]
    d = math.sqrt(dx * dx + dy * dy)
    return Qc / (1.0 + math.exp(kappa * (d - 2.0 * robot_radius)))


@cuda.jit
def gpu_total_obstacle_collision_cost(
    robot, obstacles, Qc, kappa, robot_radius, total_cost_array
):
    i, j = cuda.grid(2)
    H = robot.shape[0] // 2
    num_obstacles = obstacles.shape[0]

    if i < H and j < num_obstacles:
        rob_x = robot[2 * i]
        rob_y = robot[2 * i + 1]
        obs_x = obstacles[j, 2 * i]
        obs_y = obstacles[j, 2 * i + 1]
        x0 = (rob_x, rob_y)
        x1 = (obs_x, obs_y)

        cost = gpu_collision_cost(x0, x1, Qc, kappa, robot_radius)
        cuda.atomic.add(total_cost_array, 0, cost)


def gpu_obstacle_collision_cost(x_robot, obstacle_predictions):
    # GPU version timing
    start_total = time.perf_counter()

    H = HORIZON_LENGTH
    num_obstacles = obstacle_predictions.shape[0]

    d_robot = cuda.to_device(x_robot)
    d_obstacles = cuda.to_device(obstacle_predictions)
    d_total_cost = cuda.to_device(np.array([0.0], dtype=np.float64))

    threads_per_block = (16, 16)
    blocks_per_grid = (
        math.ceil(H / threads_per_block[0]),
        math.ceil(num_obstacles / threads_per_block[1]),
    )

    # Time the kernel
    cuda.synchronize()
    start_kernel = time.perf_counter()
    gpu_total_obstacle_collision_cost[blocks_per_grid, threads_per_block](
        d_robot, d_obstacles, Qc, kappa, ROBOT_RADIUS, d_total_cost
    )
    cuda.synchronize()
    end_kernel = time.perf_counter()

    c2 = d_total_cost.copy_to_host()
    cuda.synchronize()
    end_total = time.perf_counter()

    timings["obstacle_collision_gpu_kernel"].append(end_kernel - start_kernel)
    timings["obstacle_collision_gpu_total"].append(end_total - start_total)
    # total time already recorded in total_cost function (outer level)
    return c2[0]


def cpu_obstacle_collision_cost(x_robot, obstacle_predictions):
    # CPU equivalent for obstacle collision cost
    # obstacle_predictions: shape (num_obstacles, 2*HORIZON_LENGTH)
    total_cost_val = 0.0
    num_obstacles = obstacle_predictions.shape[0]
    H = HORIZON_LENGTH
    for i in range(H):
        rob_x = x_robot[2 * i]
        rob_y = x_robot[2 * i + 1]
        for j in range(num_obstacles):
            obs_x = obstacle_predictions[j, 2 * i]
            obs_y = obstacle_predictions[j, 2 * i + 1]
            d = np.sqrt((rob_x - obs_x) ** 2 + (rob_y - obs_y) ** 2)
            total_cost_val += Qc / (1.0 + np.exp(kappa * (d - 2.0 * ROBOT_RADIUS)))
    return total_cost_val


def plot_timing_results():
    # Example plotting function
    # Compute average times if needed
    def avg(lst):
        return np.mean(lst) if len(lst) > 0 else 0.0

    avg_predict_gpu_kernel = avg(timings["predict_obstacles_gpu_kernel"])
    avg_predict_gpu_total = avg(timings["predict_obstacles_gpu_total"])
    avg_predict_cpu = avg(timings["predict_obstacles_cpu"])

    avg_collision_gpu_kernel = avg(timings["obstacle_collision_gpu_kernel"])
    avg_collision_gpu_total = avg(timings["obstacle_collision_gpu_total"])
    avg_collision_cpu = avg(timings["obstacle_collision_cpu"])

    # Plot for obstacle prediction
    plt.figure(figsize=(8, 6))
    plt.bar(
        ["CPU", "GPU Kernel", "GPU Total"],
        [avg_predict_cpu, avg_predict_gpu_kernel, avg_predict_gpu_total],
        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
    )
    plt.title("Obstacle Prediction Timing")
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.savefig("obstacle_prediction_timing.png")

    # Plot for obstacle collision
    plt.figure(figsize=(8, 6))
    plt.bar(
        ["CPU", "GPU Kernel", "GPU Total"],
        [avg_collision_cpu, avg_collision_gpu_kernel, avg_collision_gpu_total],
        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
    )
    plt.title("Obstacle Collision Timing")
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.savefig("obstacle_collision_timing.png")

    # Plot speedups for both operations (using CPU as baseline)
    # Speedup = CPU time / GPU total time
    pred_speedup = (
        avg_predict_cpu / avg_predict_gpu_total if avg_predict_gpu_total > 0 else 1.0
    )
    collision_speedup = (
        avg_collision_cpu / avg_collision_gpu_total
        if avg_collision_gpu_total > 0
        else 1.0
    )

    plt.figure(figsize=(8, 6))
    plt.bar(
        ["Obstacle Prediction", "Obstacle Collision"],
        [pred_speedup, collision_speedup],
        color=["#2ca02c", "#2ca02c"],
    )
    plt.title("GPU Speedup Over CPU (Total GPU Time)")
    plt.ylabel("Speedup (CPU/GPU)")
    plt.grid(True)
    plt.savefig("gpu_speedup.png")

    plt.show()
