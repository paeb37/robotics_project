import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

class KKTSolver:
    def __init__(self, n_variables):
        self.n_variables = n_variables
        self.constraints = []
        self.objective = None
        
    def add_constraint(self, A, b, constraint_type='equality'):
        """Add equality or inequality constraint: Ax = b or Ax <= b"""
        self.constraints.append({
            'matrix': A,
            'vector': b,
            'type': constraint_type
        })
    
    def set_objective(self, Q, c):
        """Set quadratic objective: 1/2 x^T Q x + c^T x"""
        self.objective = {
            'Q': Q,
            'c': c
        }
    
    def solve(self):
        """Solve KKT system using direct method"""
        # Construct KKT matrix and vector
        n = self.n_variables
        m = sum(c['matrix'].shape[0] for c in self.constraints)
        
        # Build KKT matrix
        KKT = np.zeros((n + m, n + m))
        KKT[:n, :n] = self.objective['Q']
        
        # Add constraint matrices
        row_idx = n
        for constraint in self.constraints:
            A = constraint['matrix']
            rows = A.shape[0]
            KKT[row_idx:row_idx+rows, :n] = A
            KKT[:n, row_idx:row_idx+rows] = A.T
            row_idx += rows
            
        # Build RHS vector
        rhs = np.zeros(n + m)
        rhs[:n] = -self.objective['c']
        
        # Solve system
        solution = np.linalg.solve(KKT, rhs)
        
        return solution[:n]  # Return only primal variables