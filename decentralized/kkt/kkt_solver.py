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
    Get the optimal velocity using KKT optimization
    
    Args:
        robot: Current robot state (positions, velocities)
        obstacles: Just an array of obstacle states
        v_desired: Desired velocity
    
    Returns:
        Optimal velocity vector (that avoids collisions)
    """
    u0 = robot[2:]
    
    def total_cost(u, robot, obstacles, v_desired):
        # This is the tracking cost with reduced weight 
        # It prioritizes collision avoidance
        tracking_cost = 0.3 * np.linalg.norm(u - v_desired)
        
        collision_cost = 0
        robot_pos = robot[:2]
        
        # Look ahead a few timesteps
        for t in range(3):
            future_time = (t + 1) * TIMESTEP
            future_robot_pos = robot_pos + u * future_time
            
            for obstacle in obstacles:
                future_obstacle_pos = obstacle[:2] + obstacle[2:] * future_time
                
                # Robot to obstacle vector
                relative_pos = future_robot_pos - future_obstacle_pos
                d = np.linalg.norm(relative_pos)
                
                # Required safe distance (this is a main change we made)
                safe_dist = (2 * ROBOT_RADIUS) * SAFETY_MARGIN
                
                collision_cost += Qc / (1 + np.exp(kappa * (d - safe_dist)))
                
                # Another quadratic penalty we added (for the edge case of close distances)
                if d < safe_dist:
                    collision_cost += Qc * (safe_dist - d)**2
                    
                    if d > 0.1:
                        repulsion = relative_pos / d
                        collision_cost += Qc * np.dot(u, -repulsion)
        
        # This is an extra velocity cost for smoothing
        velocity_smoothing = 0.1 * np.linalg.norm(u - robot[2:])
        
        return tracking_cost + collision_cost + velocity_smoothing

    def kkt_cost(u):
        """
        KKT-based cost function
        This includes both the objective and constraint penalties
        
        Args:
            u: Control input - this is the velocity
        
        Returns:
            Total cost (including constraint penalties)
        """
        # Calculate base cost (tracking + collision avoidance)
        cost = total_cost(u, robot, obstacles, v_desired)
        
        # This is a new barrier term for inequality constraints (the velocity bounds)
        # MU = penalty parameter, weights the constraint violation
        constraint_violation = np.maximum(0, np.abs(u) - VMAX)
        return cost + MU * np.sum(constraint_violation)

    # Bounds for velocity optimization
    bounds = Bounds([-VMAX, -VMAX], [VMAX, VMAX])

    # SLSQP (Sequential Least SQuares Programming) - uses KKT under the hood
    res = minimize(
        kkt_cost,           
        u0,                 
        method='SLSQP',     
        bounds=bounds
    )

    # the optimal velocity vector
    return res.x


def update_state(x, v):
    new_state = np.empty((4))
    new_state[:2] = x[:2] + v * TIMESTEP
    new_state[-2:] = v
    return new_state