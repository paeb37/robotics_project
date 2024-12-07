"""
Plotting tool for 2D multi-robot system

author: Ashwin Bose (@atb033)
modified by: GitHub Copilot
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import numpy as np


def plot_robots_and_obstacles(
    robots_state_history, obstacles, goals, robot_radius, num_steps, sim_time, filename
):
    num_robots = robots_state_history.shape[2]

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, 10), ylim=(0, 10))
    ax.set_aspect("equal")
    ax.grid()

    # Create a list of line objects for robot trajectories
    lines = [ax.plot([], [], "--r")[0] for _ in range(num_robots)]

    # Create patches for all robots
    robot_patches = []
    # Create patches for all goals and annotate them
    goal_patches = []
    for i in range(num_robots):
        patch = Circle(
            (robots_state_history[0, 0, i], robots_state_history[1, 0, i]),
            robot_radius,
            facecolor="green",
            edgecolor="black",
        )
        robot_patches.append(patch)
        goal_patch = Circle(
            (goals[i][0], goals[i][1]),
            robot_radius,
            facecolor="red",
        )
        goal_patches.append(goal_patch)

    obstacle_list = []
    for obstacle in range(np.shape(obstacles)[2]):
        obstacle = Circle((0, 0), robot_radius, facecolor="aqua", edgecolor="black")
        obstacle_list.append(obstacle)

    def init():
        for patch in robot_patches:
            ax.add_patch(patch)
        for obstacle in obstacle_list:
            ax.add_patch(obstacle)
        for line in lines:
            line.set_data([], [])
        return robot_patches + lines + obstacle_list

    def animate(i):
        for robot_idx, patch in enumerate(robot_patches):
            patch.center = (
                robots_state_history[0, i, robot_idx],
                robots_state_history[1, i, robot_idx],
            )
        for j in range(len(obstacle_list)):
            obstacle_list[j].center = (obstacles[0, i, j], obstacles[1, i, j])
        for robot_idx, line in enumerate(lines):
            line.set_data(
                robots_state_history[0, :i, robot_idx],
                robots_state_history[1, :i, robot_idx],
            )
        return robot_patches + lines + obstacle_list

    init()
    step = sim_time / num_steps
    for i in range(num_steps):
        animate(i)
        plt.pause(step)

    # Save animation
    if not filename:
        return

    ani = animation.FuncAnimation(
        fig,
        animate,
        np.arange(1, num_steps),
        interval=200,
        blit=True,
        init_func=init,
    )

    ani.save(filename, "ffmpeg", fps=60)


def plot_robot(robot, timestep, radius=1, is_obstacle=False):
    if robot is None:
        return
    center = robot[:2, timestep]
    x = center[0]
    y = center[1]
    if is_obstacle:
        circle = plt.Circle((x, y), radius, color="aqua", ec="black")
        plt.plot(
            robot[0, :timestep],
            robot[1, :timestep],
            "--r",
        )
    else:
        circle = plt.Circle((x, y), radius, color="green", ec="black")
        plt.plot(robot[0, :timestep], robot[1, :timestep], "blue")

    plt.gcf().gca().add_artist(circle)
