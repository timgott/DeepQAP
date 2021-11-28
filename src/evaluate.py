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
    sum = 0
    for i in range(samples):
        sum += qap.compute_value(random_assignment(qap))
    return sum/samples

def evaluate(agent, problem_generator):
    eval_qap = problem_generator()
    compute_optimum = qap.size < 10
    if compute_optimum:
        print("Finding optimum...")
        optimal_value, optimal_assignment = solve_qap_backtracking(eval_qap)
    agent_value, agent_assignment = agent.solve_and_learn(eval_qap)
    print("Finding average value...")
    average_value = compute_random_average_value(eval_qap, samples=500)
    print("Comparison:")
    if compute_optimum:
        print("Optimal:", optimal_value, optimal_assignment)
    print("Average:", average_value)
    print("Agent:", agent_value, agent_assignment)

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} agent_path problem_generator")
        return 1

    agent = ReinforceAgent()
    agent.load_checkpoint(sys.argv[1])
    problem_generator = taskgenerators.generators[sys.argv[2]]
    for i in range(10):
        evaluate(agent, problem_generator)
    return 0

if __name__ == "__main__":
    sys.exit(main())