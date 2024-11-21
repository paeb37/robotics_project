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
        n = self.n_variables
        
        # Count equality and inequality constraints separately
        n_eq = sum(1 for c in self.constraints if c['type'] == 'equality')
        n_ineq = sum(1 for c in self.constraints if c['type'] == 'inequality')
        m = n_eq + n_ineq
        
        if m == 0:  # No constraints
            return -np.linalg.solve(self.objective['Q'], self.objective['c'])
            
        # Build KKT matrix
        KKT = np.zeros((n + m, n + m))
        KKT[:n, :n] = self.objective['Q']
        
        # Build RHS vector
        rhs = np.zeros(n + m)
        rhs[:n] = -self.objective['c']
        
        # Add constraint matrices
        row_idx = n
        for constraint in self.constraints:
            A = constraint['matrix']
            b = constraint['vector']
            rows = A.shape[0]
            
            KKT[row_idx:row_idx+rows, :n] = A
            KKT[:n, row_idx:row_idx+rows] = A.T
            rhs[row_idx:row_idx+rows] = b
            row_idx += rows
            
        try:
            solution = np.linalg.solve(KKT, rhs)
            return solution[:n]  # Return only primal variables
        except np.linalg.LinAlgError:
            return None