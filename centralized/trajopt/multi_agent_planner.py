import numpy as np
import yaml
from trajectory_planner import TrajectoryOptimizer
import argparse

class MultiAgentPlanner:
    def __init__(self, map):
        self.map = map
        self.num_agents = len(map["agents"])
        
    def compute_plans(self):
        output = {"schedule": {}}
        
        # Sequential planning
        for i in range(self.num_agents):
            planner = TrajectoryOptimizer(self.map, i)
            
            if planner.compute_plan():
                plan = planner.get_plan()
                output["schedule"].update(plan)
                # Update map with new trajectory as moving obstacle
                self.map["dynamic_obstacles"] = output["schedule"]
            else:
                print(f"Failed to find plan for agent {i}")
                return None
                
        return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("map", help="input file containing map")
    parser.add_argument("output", help="output file with the schedule")
    
    args = parser.parse_args()
    
    with open(args.map, 'r') as map_file:
        map = yaml.load(map_file, Loader=yaml.FullLoader)

    planner = MultiAgentPlanner(map)
    output = planner.compute_plans()
    
    if output:
        with open(args.output, 'w') as output_yaml:
            yaml.safe_dump(output, output_yaml)
    else:
        print("Failed to find complete solution")

if __name__ == "__main__":
    main()