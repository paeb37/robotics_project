import numpy as np

class ActiveSetSolver:
    def __init__(self, max_iter=100):
        self.max_iter = max_iter
        
    def solve(self, Q, c, A, b, G, h):
        """
        Solve QP problem using active set method:
        min 1/2 x^T Q x + c^T x
        s.t. Ax = b
             Gx <= h
        """
        n = Q.shape[0]
        x = np.zeros(n)
        active_set = set()
        
        for iter in range(self.max_iter):
            # Solve equality constrained QP with current active set
            G_active = G[list(active_set)]
            h_active = h[list(active_set)]
            
            # Construct KKT system
            KKT = np.block([
                [Q, A.T, G_active.T],
                [A, np.zeros((A.shape[0], A.shape[0])), np.zeros((A.shape[0], len(active_set)))],
                [G_active, np.zeros((len(active_set), A.shape[0])), np.zeros((len(active_set), len(active_set)))]
            ])
            
            rhs = np.concatenate([-c, b, h_active])
            
            try:
                sol = np.linalg.solve(KKT, rhs)
                x_new = sol[:n]
                
                # Check if solution is feasible
                if np.all(G @ x_new <= h + 1e-10):
                    return x_new
                    
                # Update active set
                violations = G @ x_new - h
                worst_idx = np.argmax(violations)
                active_set.add(worst_idx)
                
            except np.linalg.LinAlgError:
                return None
                
        return None 