import numpy as np
from numba import cuda
from scipy.optimize import minimize, Bounds
import time
from utils.multi_robot_plot import plot_robot_and_obstacles
from utils.create_obstacles import create_obstacles

# Constants
SIM_TIME = 8.0
TIMESTEP = 0.1
NUMBER_OF_TIMESTEPS = int(SIM_TIME / TIMESTEP)
ROBOT_RADIUS = 0.5
VMAX = 2
VMIN = 0.2

# NMPC parameters
HORIZON_LENGTH = 10  # Increased horizon length for better prediction
NMPC_TIMESTEP = 0.2  # Reduced timestep for finer control
MAX_VELOCITY = (1 / np.sqrt(2)) * VMAX
VELOCITY_BOUNDS = Bounds([-MAX_VELOCITY] * HORIZON_LENGTH * 2, [MAX_VELOCITY] * HORIZON_LENGTH * 2)

# Cost function parameters
TRACKING_WEIGHT = 1.0
COLLISION_WEIGHT = 10.0
SMOOTHNESS_WEIGHT = 0.1

def simulate(filename):
    obstacles = create_obstacles(SIM_TIME, NUMBER_OF_TIMESTEPS)
    start = np.array([5, 5])
    goal = np.array([10, 6])  # Changed goal position for a longer trajectory

    robot_state = start
    robot_state_history = np.zeros((NUMBER_OF_TIMESTEPS, 2))

    start_time = time.time()

    for i in range(NUMBER_OF_TIMESTEPS):
        obstacle_predictions = predict_obstacle_positions(obstacles[:, i, :])
        xref = compute_xref(robot_state, goal, HORIZON_LENGTH, NMPC_TIMESTEP)
        
        vel, _ = compute_velocity(robot_state, obstacle_predictions, xref)
        
        robot_state = update_state(robot_state, vel, TIMESTEP)
        robot_state_history[i] = robot_state

    end_time = time.time()
    sim_time = end_time - start_time
    print(f"Simulation time: {sim_time:.2f} seconds")

    plot_robot_and_obstacles(robot_state_history.T, obstacles, ROBOT_RADIUS, NUMBER_OF_TIMESTEPS, SIM_TIME, filename)

def compute_velocity(robot_state, obstacle_predictions, xref):
    u0 = np.zeros(2 * HORIZON_LENGTH)  # Initialize with zeros for smoother initial guess
    
    def cost_fn(u):
        return total_cost(u, robot_state, obstacle_predictions, xref)

    res = minimize(cost_fn, u0, method='SLSQP', bounds=VELOCITY_BOUNDS)
    return res.x[:2], res.x

def total_cost(u, robot_state, obstacle_predictions, xref):
    x_robot = update_state(robot_state, u, NMPC_TIMESTEP)
    tracking_cost = np.sum((x_robot - xref)**2)
    collision_cost = compute_collision_cost(x_robot, obstacle_predictions)
    smoothness_cost = np.sum(np.diff(u.reshape(-1, 2), axis=0)**2)
    
    return (TRACKING_WEIGHT * tracking_cost +
            COLLISION_WEIGHT * collision_cost +
            SMOOTHNESS_WEIGHT * smoothness_cost)

def compute_collision_cost(robot_trajectory, obstacle_predictions):
    cost = 0
    for i in range(HORIZON_LENGTH):
        robot_pos = robot_trajectory[2*i:2*i+2]
        for obstacle in obstacle_predictions:
            obstacle_pos = obstacle[i]
            distance = np.linalg.norm(robot_pos - obstacle_pos)
            cost += 1 / (1 + np.exp(10 * (distance - 2*ROBOT_RADIUS)))
    return cost

@cuda.jit
def predict_obstacle_kernel_nonlinear(obstacles, obstacle_predictions, timestep):
    i = cuda.grid(1)
    if i < obstacles.shape[1]:
        x, y, vx, vy = obstacles[:, i]
        
        lower_triangular_ones = cuda.local.array((HORIZON_LENGTH, HORIZON_LENGTH), dtype=np.float32)
        for row in range(HORIZON_LENGTH):
            for col in range(row + 1):
                lower_triangular_ones[row, col] = 1.0
        
        for j in range(HORIZON_LENGTH):
            t = j * timestep
            vx_t = vx * (1 + 0.1 * t)  
            vy_t = vy * (1 + 0.1 * t)
            
            dx = 0
            dy = 0
            for k in range(j + 1):
                dx += lower_triangular_ones[j, k] * vx * (1 + 0.1 * k * timestep) * timestep
                dy += lower_triangular_ones[j, k] * vy * (1 + 0.1 * k * timestep) * timestep
            
            obstacle_predictions[i, j, 0] = x + dx
            obstacle_predictions[i, j, 1] = y + dy

def predict_obstacle_positions(obstacles):
    obstacles = np.ascontiguousarray(obstacles, dtype=np.float32)
    threads_per_block = 256
    blocks_per_grid = (obstacles.shape[1] + threads_per_block - 1) // threads_per_block

    d_obstacle_predictions = cuda.device_array(
        (obstacles.shape[1], HORIZON_LENGTH, 2),
        dtype=np.float32
    )

    d_obstacles = cuda.to_device(obstacles)
    predict_obstacle_kernel_nonlinear[blocks_per_grid, threads_per_block](
        d_obstacles,
        d_obstacle_predictions,
        NMPC_TIMESTEP
    )

    return d_obstacle_predictions.copy_to_host()

def update_state(x0, u, timestep):
    N = len(u) // 2
    lower_triangular_ones_matrix = np.tril(np.ones((N, N)))
    kron = np.kron(lower_triangular_ones_matrix, np.eye(2))
    return np.vstack([np.eye(2)] * N) @ x0 + kron @ u * timestep

def compute_xref(start, goal, number_of_steps, timestep):
    dir_vec = goal - start
    norm = np.linalg.norm(dir_vec)
    if norm < 0.1:
        new_goal = start
    else:
        dir_vec = dir_vec / norm
        new_goal = start + dir_vec * min(VMAX * timestep * number_of_steps, norm)
    return np.linspace(start, new_goal, number_of_steps).flatten()