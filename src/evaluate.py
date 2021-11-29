import numpy as np
import random
from qap import GraphAssignmentProblem
from reinforce import ReinforceAgent
from simplesolver import solve_qap_backtracking
import sys
import taskgenerators


def random_assignment(qap: GraphAssignmentProblem):
    unassigned_targets = list(qap.graph_target.nodes)
    random.shuffle(unassigned_targets)
    return unassigned_targets

def compute_random_average_value(qap: GraphAssignmentProblem, samples=100):
    values = []
    for i in range(samples):
       values.append(qap.compute_value(random_assignment(qap)))
    return np.mean(values), np.min(values), np.max(values)

def compute_agent_average_value(agent, qap: GraphAssignmentProblem, samples=100):
    values = []
    for i in range(samples):
       values.append(qap.compute_value(agent.solve(qap)))
    return np.mean(values), np.min(values), np.max(values)

def evaluate(agent, problem_generator):
    eval_qap = problem_generator()
    
    compute_optimum = eval_qap.size < 10
    if compute_optimum:
        print("Finding optimum...")
        optimal_value, optimal_assignment = solve_qap_backtracking(eval_qap)
    
    print("Finding random average...")
    random_value = compute_random_average_value(eval_qap, samples=100)

    print("Finding agent average...")
    agent_value = compute_agent_average_value(agent, eval_qap, samples=100)

    print("Comparison:")
    if compute_optimum:
            print("Optimal:", optimal_value, optimal_assignment)
    print("Random:", random_value)
    print("Agent:", agent_value)
    print("Sample agent assignment:", agent.solve(eval_qap))

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} agent_path problem_generator")
        return 1

    agent = ReinforceAgent()
    agent.load_checkpoint(sys.argv[1])
    problem_generator = taskgenerators.generators[sys.argv[2]]
    evaluate(agent, problem_generator)
    return 0

if __name__ == "__main__":
    sys.exit(main())