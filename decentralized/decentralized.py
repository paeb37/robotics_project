import velocity_obstacle.velocity_obstacle as velocity_obstacle
import nmpc.nmpc_timing as nmpc
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        help="mode of obstacle avoidance; options: velocity_obstacle, or nmpc",
    )
    parser.add_argument(
        "-f", "--filename", help="filename, in case you want to save the animation"
    )

    args = parser.parse_args()
    if args.mode == "velocity_obstacle":
        velocity_obstacle.simulate(args.filename)
    elif args.mode == "nmpc":
        # nmpc.simulate(args.filename)

        # After running multiple simulations:
        for i in range(10):
            nmpc.simulate(f"simulation_{i}.gif")
        nmpc.plot_timing_results()
        #
        # The above plotting function will produce and save 3 plots:
        # 1. obstacle_prediction_timing.png
        # 2. obstacle_collision_timing.png
        # 3. gpu_speedup.png
        #
        # These can be used in the results section of a paper.
    else:
        print("Please enter mode the desired mode: velocity_obstacle or nmpc")
