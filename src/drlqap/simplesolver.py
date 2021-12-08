import math
from drlqap.qap import GraphAssignmentProblem

def solve_qap_backtracking(qap: GraphAssignmentProblem):
    variables = list(qap.graph_source.nodes)
    current_assignment = [None] * qap.size

    best_assignment = None
    best_value = math.inf

    def recurse(depth, unassigned_targets: set):
        nonlocal best_value, best_assignment, current_assignment

        # fully assigned?
        if depth == qap.size:
            value = qap.compute_value(current_assignment)
            if value < best_value:
                best_value = value
                best_assignment = current_assignment[:]
            return

        variable = variables[depth]
        for target in unassigned_targets:
            current_assignment[variable] = target
            recurse(depth + 1, unassigned_targets.difference((target,)))
        
        current_assignment[variable] = None

        return
    
    recurse(0, set(qap.graph_target.nodes))
    return best_value, best_assignment


    