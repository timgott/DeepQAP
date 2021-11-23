import math
from qap import GraphAssignmentProblem

def solve_qap_backtracking(qap: GraphAssignmentProblem):
    def recurse(depth):
        if depth == qap.size:
            return qap.compute_value(), []

        variable = depth
        best_value = math.inf
        best_assignment = None
        for target in qap.unassigned_targets:
            qap.assign(variable, target)
            value, assignment = recurse(depth + 1)
            qap.unassign(variable)

            # Keep minimum value
            if value < best_value:
                best_value = value
                best_assignment = [(variable, target)] + assignment

        return best_value, best_assignment
    
    return recurse(0)
