import numpy as np
from scipy.optimize import minimize
import yaml
import argparse

class TrajectoryOptimizer:
    def __init__(self, map, agent_id):
        self.start = np.array(map["agents"][agent_id]["start"])
        self.goal = np.array(map["agents"][agent_id]["goal"])
        self.name = map["agents"][agent_id]["name"]
        self.obstacles = np.array(map["map"]["obstacles"])
        self.other_agents = self._get_other_agents(map, agent_id)
        self.N = 20  # number of waypoints
        self.dt = 1.0  # time step
        
    def _get_other_agents(self, map, agent_id):
        other_agents = []
        for i, agent in enumerate(map["agents"]):
            if i != agent_id:
                other_agents.append({
                    'start': np.array(agent['start']),
                    'goal': np.array(agent['goal'])
                })
        return other_agents

    def objective(self, x):
        """Minimize total distance and control effort"""
        positions = x.reshape(-1, 2)
        
        # Path length cost
        path_length = np.sum(np.linalg.norm(positions[1:] - positions[:-1], axis=1))
        
        # Control effort (acceleration) cost
        accelerations = (positions[2:] - 2*positions[1:-1] + positions[:-2]) / (self.dt**2)
        control_cost = np.sum(np.linalg.norm(accelerations, axis=1))
        
        return path_length + 0.1 * control_cost

    def obstacle_constraints(self, x):
        """Collision avoidance constraints"""
        positions = x.reshape(-1, 2)
        constraints = []
        
        # Static obstacle avoidance
        for obstacle in self.obstacles:
            distances = np.linalg.norm(positions - obstacle, axis=1)
            constraints.extend(distances - 0.5)  # Minimum clearance of 0.5
            
        # Other agents avoidance (simple linear interpolation for their trajectories)
        for agent in self.other_agents:
            for t in range(self.N):
                agent_pos = agent['start'] + (agent['goal'] - agent['start']) * t/self.N
                dist = np.linalg.norm(positions[t] - agent_pos)
                constraints.append(dist - 0.7)  # Minimum separation of 0.7
                
        return np.array(constraints)

    def boundary_constraints(self, x):
        """Start and goal constraints"""
        positions = x.reshape(-1, 2)
        constraints = []
        
        # Start position constraint
        constraints.extend(positions[0] - self.start)
        
        # Goal position constraint
        constraints.extend(positions[-1] - self.goal)
        
        # Velocity constraints (optional)
        velocities = (positions[1:] - positions[:-1]) / self.dt
        max_velocity = 1.0
        constraints.extend(np.linalg.norm(velocities, axis=1) - max_velocity)
        
        return np.array(constraints)

    def compute_plan(self):
        # Initial guess: linear interpolation
        x0 = np.zeros((self.N, 2))
        for i in range(self.N):
            x0[i] = self.start + (self.goal - self.start) * i/(self.N-1)
        
        # Optimization constraints
        constraints = [
            {'type': 'ineq', 'fun': self.obstacle_constraints},
            {'type': 'eq', 'fun': self.boundary_constraints}
        ]
        
        # Solve optimization problem
        result = minimize(
            self.objective,
            x0.flatten(),
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            self.optimal_trajectory = result.x.reshape(-1, 2)
            return True
        return False

    def get_plan(self):
        """Convert optimal trajectory to schedule format"""
        path_list = []
        for t, pos in enumerate(self.optimal_trajectory):
            temp_dict = {
                "x": float(pos[0]),
                "y": float(pos[1]),
                "t": t
            }
            path_list.append(temp_dict)
            
        return {self.name: path_list}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("map", help="input file containing map")
    parser.add_argument("output", help="output file with the schedule")
    
    args = parser.parse_args()
    
    with open(args.map, 'r') as map_file:
        map = yaml.load(map_file, Loader=yaml.FullLoader)

    output = {"schedule": {}}
    
    # Compute trajectories for all agents
    for i in range(len(map["agents"])):
        planner = TrajectoryOptimizer(map, i)
        if planner.compute_plan():
            plan = planner.get_plan()
            output["schedule"].update(plan)
        else:
            print(f"Failed to find plan for agent {i}")
            return

    with open(args.output, 'w') as output_yaml:
        yaml.safe_dump(output, output_yaml)

if __name__ == "__main__":
    main()