import numpy as np

# def create_obstacles(sim_time, num_timesteps):
#     # Obstacle 1
#     v = -2
#     p0 = np.array([5, 12])
#     obst = create_robot(p0, v, np.pi/2, sim_time,
#                         num_timesteps).reshape(4, num_timesteps, 1)
#     obstacles = obst
#     # Obstacle 2
#     v = 2
#     p0 = np.array([0, 5])
#     obst = create_robot(p0, v, 0, sim_time, num_timesteps).reshape(
#         4, num_timesteps, 1)
#     obstacles = np.dstack((obstacles, obst))
#     # Obstacle 3
#     v = 2
#     p0 = np.array([10, 10])
#     obst = create_robot(p0, v, -np.pi * 3 / 4, sim_time, num_timesteps).reshape(4,
#                                                                                 num_timesteps, 1)
#     obstacles = np.dstack((obstacles, obst))
#     # Obstacle 4
#     v = 2
#     p0 = np.array([7.5, 2.5])
#     obst = create_robot(p0, v, np.pi * 3 / 4, sim_time, num_timesteps).reshape(4,
#                                                                                num_timesteps, 1)
#     obstacles = np.dstack((obstacles, obst))

#     return obstacles

def create_obstacles(sim_time, num_timesteps, num_obstacles=100):
    obstacles = []

    for i in range(num_obstacles):
        # Generate random or specific values for each obstacle
        v = np.random.uniform(-3, 3)  # Random velocity between -3 and 3
        x = np.random.uniform(0, 100)  # Random initial x position
        y = np.random.uniform(0, 100)  # Random initial y position
        direction = np.random.uniform(-np.pi, np.pi)  # Random direction

        # Create the obstacle using `create_robot` and reshape
        p0 = np.array([x, y])
        obst = create_robot(p0, v, direction, sim_time, num_timesteps).reshape(4, num_timesteps, 1)
        
        # Append or stack the obstacle
        if len(obstacles) == 0:
            obstacles = obst
        else:
            obstacles = np.dstack((obstacles, obst))

    return obstacles


def create_robot(p0, v, theta, sim_time, num_timesteps):
    # Creates obstacles starting at p0 and moving at v in theta direction
    t = np.linspace(0, sim_time, num_timesteps)
    theta = theta * np.ones(np.shape(t))
    vx = v * np.cos(theta)
    vy = v * np.sin(theta)
    v = np.stack([vx, vy])
    p0 = p0.reshape((2, 1))
    p = p0 + np.cumsum(v, axis=1) * (sim_time / num_timesteps)
    p = np.concatenate((p, v))
    return p
