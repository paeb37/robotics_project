"""
Collision avoidance using Velocity-obstacle method

author: Ashwin Bose (atb033@github.com)
"""

from utils.multi_robot_plot import plot_robot_and_obstacles
from utils.create_obstacles import create_obstacles
from utils.control import compute_desired_velocity
import numpy as np
from scipy.optimize import minimize, Bounds

SIM_TIME = 5.
TIMESTEP = 0.1
NUMBER_OF_TIMESTEPS = int(SIM_TIME/TIMESTEP)
ROBOT_RADIUS = 0.5
VMAX = 2
VMIN = 0.2

# Since initial kkt changes, increased optimization parameters for stronger collision avoidance
MU = 200 # Penalty parameter
Qc = 200 # Collision cost weight
kappa = 15 # Collision cost shape parameter
SAFETY_MARGIN = 2.0 # Safety margin multiplier for collision checking


def simulate(filename):
    obstacles = create_obstacles(SIM_TIME, NUMBER_OF_TIMESTEPS)

    start = np.array([5, 0, 0, 0])
    goal = np.array([5, 10, 0, 0])

    robot_state = start
    robot_state_history = np.empty((4, NUMBER_OF_TIMESTEPS))
    for i in range(NUMBER_OF_TIMESTEPS):
        v_desired = compute_desired_velocity(robot_state, goal, ROBOT_RADIUS, VMAX)
        control_vel = compute_velocity(
            robot_state, obstacles[:, i, :], v_desired)
        robot_state = update_state(robot_state, control_vel)
        robot_state_history[:4, i] = robot_state

    plot_robot_and_obstacles(
        robot_state_history, obstacles, ROBOT_RADIUS, NUMBER_OF_TIMESTEPS, SIM_TIME, filename)


def compute_velocity(robot, obstacles, v_desired):
    """
    Compute optimal velocity using KKT-based optimization
    
    Args:
        robot: Current robot state [x, y, vx, vy]
        obstacles: Array of obstacle states
        v_desired: Desired velocity vector [vx, vy]
    
    Returns:
        Optimal velocity vector that avoids collisions
    """
    u0 = robot[2:]
    
    def total_cost(u, robot, obstacles, v_desired):
        # Tracking cost with reduced weight to prioritize collision avoidance
        tracking_cost = 0.3 * np.linalg.norm(u - v_desired)
        
        collision_cost = 0
        robot_pos = robot[:2]
        
        # Look ahead multiple timesteps
        for t in range(3):  # Check multiple future positions
            future_time = (t + 1) * TIMESTEP
            future_robot_pos = robot_pos + u * future_time
            
            for obstacle in obstacles:
                future_obstacle_pos = obstacle[:2] + obstacle[2:] * future_time
                
                # Vector from robot to obstacle
                relative_pos = future_robot_pos - future_obstacle_pos
                d = np.linalg.norm(relative_pos)
                
                # Required safe distance (sum of radii plus safety margin)
                safe_dist = (2 * ROBOT_RADIUS) * SAFETY_MARGIN
                
                # Exponential collision cost
                collision_cost += Qc / (1 + np.exp(kappa * (d - safe_dist)))
                
                # Additional quadratic penalty for close distances
                if d < safe_dist:
                    collision_cost += Qc * (safe_dist - d)**2
                    
                    # Add directional repulsion
                    if d > 0.1:  # Avoid division by zero
                        repulsion = relative_pos / d
                        collision_cost += Qc * np.dot(u, -repulsion)
        
        # Add velocity smoothing cost
        velocity_smoothing = 0.1 * np.linalg.norm(u - robot[2:])
        
        return tracking_cost + collision_cost + velocity_smoothing

    def kkt_cost(u):
        """
        KKT-based cost function that includes both the objective and constraint penalties
        
        Args:
            u: Control input (velocity) [vx, vy]
        
        Returns:
            Total cost including constraint penalties
        """
        # Calculate base cost (tracking + collision avoidance)
        cost = total_cost(u, robot, obstacles, v_desired)
        
        # Add barrier term for inequality constraints (velocity bounds)
        # MU acts as a penalty parameter that weights the constraint violation
        constraint_violation = np.maximum(0, np.abs(u) - VMAX)
        return cost + MU * np.sum(constraint_violation)

    # Define bounds for velocity optimization
    # Both vx and vy must be within [-VMAX, VMAX]
    bounds = Bounds([-VMAX, -VMAX], [VMAX, VMAX])

    # Solve the optimization problem using SLSQP (Sequential Least SQuares Programming)
    # This method handles nonlinear constraints and is suitable for KKT optimization
    res = minimize(
        kkt_cost,           # Objective function to minimize
        u0,                 # Initial guess
        method='SLSQP',     # Optimization method
        bounds=bounds       # Velocity bounds
    )

    # Return the optimal velocity vector
    return res.x


def update_state(x, v):
    new_state = np.empty((4))
    new_state[:2] = x[:2] + v * TIMESTEP
    new_state[-2:] = v
    return new_state