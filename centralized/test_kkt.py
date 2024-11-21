import numpy as np
from kkt_solver.kkt_core import KKTSolver
from kkt_solver.active_set import ActiveSetSolver

def test_simple_problem():
    # Simple test problem
    n_vars = 2
    solver = KKTSolver(n_vars)
    
    # Objective: minimize x^2 + y^2
    Q = np.eye(2)
    c = np.zeros(2)
    solver.set_objective(Q, c)
    
    # Constraint: x + y = 1
    A = np.array([[1, 1]])
    b = np.array([1])
    solver.add_constraint(A, b)
    
    solution = solver.solve()
    print("Solution:", solution)
    print("Expected: [0.5, 0.5]")

def test_active_set():
    # Test active set solver
    Q = np.eye(2)
    c = np.zeros(2)
    A = np.array([[1, 1]])
    b = np.array([1])
    G = np.array([[1, 0], [0, 1]])
    h = np.array([0.7, 0.7])
    
    solver = ActiveSetSolver()
    solution = solver.solve(Q, c, A, b, G, h)
    print("Active set solution:", solution)

if __name__ == "__main__":
    print("Testing KKT solver...")
    test_simple_problem()
    print("\nTesting active set solver...")
    test_active_set() 