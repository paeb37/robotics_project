import velocity_obstacle.velocity_obstacle as velocity_obstacle
import kkt.kkt_solver as kkt_solver
import nmpc.nmpc as nmpc
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--mode", help="mode of obstacle avoidance; options: velocity_obstacle, or nmpc")
    parser.add_argument(
        "-f", "--filename", help="filename, in case you want to save the animation")

    args = parser.parse_args()
    if args.mode == "velocity_obstacle": # existing method
        velocity_obstacle.simulate(args.filename)
    elif args.mode == "nmpc":  # existing method
        nmpc.simulate(args.filename)
    elif args.mode == "kkt": # our method
        kkt_solver.simulate(args.filename)
    else:
        print("Please enter mode the desired mode: velocity_obstacle or nmpc")
