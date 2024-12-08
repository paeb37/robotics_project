import velocity_obstacle.velocity_obstacle as velocity_obstacle
import kkt.kkt_solver as kkt_solver
import nmpc.nmpc as nmpc
import argparse
import time
import numpy as np

def run_with_timing(simulate_func, filename):
    start_time = time.time()
    computation_times = []
    
    def timing_wrapper(*args, **kwargs):
        iter_start = time.time()
        result = simulate_func(*args, **kwargs)
        computation_times.append(time.time() - iter_start)
        return result
        
    timing_wrapper.computation_times = computation_times
    
    timing_wrapper(filename)
    total_time = time.time() - start_time
    
    comp_times = np.array(computation_times)
    print(f"\nTiming for {simulate_func.__module__}:")
    print(f"Total Runtime: {total_time:.4f} seconds")
    print(f"Average Iteration Time: {np.mean(comp_times)*1000:.2f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--mode", help="mode of obstacle avoidance; options: velocity_obstacle, or nmpc")
    parser.add_argument(
        "-f", "--filename", help="filename, in case you want to save the animation")

    args = parser.parse_args()
    if args.mode == "velocity_obstacle":
        run_with_timing(velocity_obstacle.simulate, args.filename)
    elif args.mode == "nmpc":
        run_with_timing(nmpc.simulate, args.filename)
    elif args.mode == "kkt":
        run_with_timing(kkt_solver.simulate, args.filename)
    else:
        print("Please enter mode the desired mode: velocity_obstacle or nmpc")
