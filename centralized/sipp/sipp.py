"""

SIPP implementation  

author: Ashwin Bose (@atb033)

See the article: DOI: 10.1109/ICRA.2011.5980306

"""

import argparse
import yaml
from math import fabs
from graph_generation import SippGraph, State
from kkt_solver.kkt_core import KKTSolver
from kkt_solver.active_set import ActiveSetSolver
import numpy as np

class SippPlanner(SippGraph):
    def __init__(self, map, agent_id):
        SippGraph.__init__(self, map)
        self.start = tuple(map["agents"][agent_id]["start"])
        self.goal = tuple(map["agents"][agent_id]["goal"])
        self.name = map["agents"][agent_id]["name"]
        self.open = []
        self.kkt_solver = KKTSolver(n_variables=2)  # x,y coordinates

    def get_successors(self, state):
        successors = []
        m_time = 1
        neighbour_list = self.get_valid_neighbours(state.position)

        for neighbour in neighbour_list:
            start_t = state.time + m_time
            end_t = state.interval[1] + m_time
            for i in self.sipp_graph[neighbour].interval_list:
                if i[0] > end_t or i[1] < start_t:
                    continue
                time = max(start_t, i[0]) 
                s = State(neighbour, time, i)
                successors.append(s)
        return successors

    def get_heuristic(self, position):
        return fabs(position[0] - self.goal[0]) + fabs(position[1]-self.goal[1])

    def setup_optimization(self):
        """Setup optimization problem for KKT solver"""
        # Objective: minimize distance to goal
        Q = np.eye(2)  # Positive definite quadratic term
        c = -2 * np.array(self.goal)  # Linear term
        self.kkt_solver.set_objective(Q, c)
        
        # Add collision avoidance constraints
        for obs in self.get_obstacles():
            # Add inequality constraints for obstacle avoidance
            G = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
            h = np.array([obs.x + 0.5, obs.y + 0.5, -(obs.x - 0.5), -(obs.y - 0.5)])
            for i in range(len(G)):
                self.kkt_solver.add_constraint(G[i:i+1], h[i:i+1], 'inequality')
            
    def compute_plan(self):
        """Modified to use KKT solver"""
        self.setup_optimization()
        
        try:
            solution = self.kkt_solver.solve()
            if solution is not None:
                # Convert solution to plan
                self.plan = self.convert_solution_to_plan(solution)
                return 1
        except:
            print("KKT solver failed, falling back to original method")
            return super().compute_plan()
            
        return 0

    def get_plan(self):
        path_list = []

        # first setpoint
        setpoint = self.plan[0]
        temp_dict = {"x":setpoint.position[0], "y":setpoint.position[1], "t":setpoint.time}
        path_list.append(temp_dict)

        for i in range(len(self.plan)-1):
            for j in range(self.plan[i+1].time - self.plan[i].time-1):
                x = self.plan[i].position[0]
                y = self.plan[i].position[1]
                t = self.plan[i].time
                setpoint = self.plan[i]
                temp_dict = {"x":x, "y":y, "t":t+j+1}
                path_list.append(temp_dict)

            setpoint = self.plan[i+1]
            temp_dict = {"x":setpoint.position[0], "y":setpoint.position[1], "t":setpoint.time}
            path_list.append(temp_dict)

        data = {self.name:path_list}
        return data



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("map", help="input file containing map and dynamic obstacles")
    parser.add_argument("output", help="output file with the schedule")
    
    args = parser.parse_args()
    
    with open(args.map, 'r') as map_file:
        try:
            map = yaml.load(map_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)


    output = dict()
    output["schedule"] = dict()

    # compute first plan
    sipp_planner = SippPlanner(map,0)

    if sipp_planner.compute_plan():
        plan = sipp_planner.get_plan()
        output["schedule"].update(plan)
        with open(args.output, 'w') as output_yaml:
            yaml.safe_dump(output, output_yaml)  
    else: 
        print("Plan not found")


if __name__ == "__main__":
    main()