import numpy as np
from numpy.typing import NDArray


def create_obstacles(
    sim_time: float, num_timesteps: int, num_obstacles: int = 4
) -> NDArray:
    # Define bounds for random positions and velocities
    x_bounds = (0, 10)
    y_bounds = (0, 12)
    v_bounds = (-2, 2)

    # Create first obstacle to initialize the array
    p0 = np.array(
        [
            np.random.uniform(x_bounds[0], x_bounds[1]),
            np.random.uniform(y_bounds[0], y_bounds[1]),
        ]
    )
    v = np.random.uniform(v_bounds[0], v_bounds[1])
    theta = np.random.uniform(-np.pi, np.pi)
    obst = create_robot(p0, v, theta, sim_time, num_timesteps).reshape(
        4, num_timesteps, 1
    )
    obstacles = obst

    # Create remaining obstacles
    for _ in range(num_obstacles - 1):
        p0 = np.array(
            [
                np.random.uniform(x_bounds[0], x_bounds[1]),
                np.random.uniform(y_bounds[0], y_bounds[1]),
            ]
        )
        v = np.random.uniform(v_bounds[0], v_bounds[1])
        theta = np.random.uniform(-np.pi, np.pi)
        obst = create_robot(p0, v, theta, sim_time, num_timesteps).reshape(
            4, num_timesteps, 1
        )
        obstacles = np.dstack((obstacles, obst))

    return obstacles


def create_robot(
    p0: NDArray, v: float, theta: float, sim_time: float, num_timesteps: int
) -> NDArray:
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
