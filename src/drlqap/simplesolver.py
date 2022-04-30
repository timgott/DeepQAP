import math
from drlqap.qap import QAP
from drlqap.qapenv import QAPReductionEnv

def solve_qap_backtracking(qap: QAP):
    variables = list(range(qap.size))
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

    recurse(0, set(range(qap.size)))
    return best_value, best_assignment


def find_max_difference_pair(a, b):
    a_sums = a.sum(0) + a.sum(1)
    b_sums = b.sum(0) + b.sum(1)

    min_a, min_a_index = a_sums.min(dim=0)
    max_a, max_a_index = a_sums.max(dim=0)
    min_b, min_b_index = b_sums.min(dim=0)
    max_b, max_b_index = b_sums.max(dim=0)

    if max_a - min_b < max_b - min_a:
        return min_a_index, max_b_index
    else:
        return max_a_index, min_b_index


def find_max_pair(matrix):
    return (matrix == matrix.max()).nonzero()[0]


def solve_qap_maxgreedy(qap: QAP):
    env = QAPReductionEnv(qap)

    while not env.done:
        qap = env.remaining_qap
        env.step(find_max_difference_pair(qap.A, qap.B))

    return env.reward_sum, env.assignment


def solve_qap_lingreedy(qap: QAP):
    env = QAPReductionEnv(qap)

    # Choose first pair based on max difference
    env.step(find_max_difference_pair(qap.A, qap.B))

    while not env.done:
        env.step(find_max_pair(env.remaining_qap.linear_costs))

    return env.reward_sum, env.assignment

