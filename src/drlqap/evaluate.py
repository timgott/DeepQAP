import numpy as np
import random
from pathlib import Path
import sys
import math
from drlqap.dqn import DQNAgent
from drlqap.qap import GraphAssignmentProblem
from drlqap.simplesolver import solve_qap_backtracking
from drlqap import taskgenerators
from drlqap import agent_configs


def random_assignment(qap: GraphAssignmentProblem):
    unassigned_targets = list(qap.graph_target.nodes)
    random.shuffle(unassigned_targets)
    return unassigned_targets

def compute_random_average_value(qaps, samples=100):
    values = []
    for qap in qaps:
        for _ in range(samples):
           values.append(qap.compute_unscaled_value(random_assignment(qap)))
    return values

def compute_best_of_k_value(qaps, samples=100):
    values = []
    for qap in qaps:
        best = math.inf
        for _ in range(samples):
           best = min(best, qap.compute_unscaled_value(random_assignment(qap)))
        values.append(best)
    return values

def compute_agent_average_value(agent, qaps, samples=100):
    values = []
    if isinstance(agent, DQNAgent):
        samples = 1
    for qap in qaps:
        for _ in range(samples):
            _, assignment = agent.solve(qap)
            value = qap.compute_unscaled_value(assignment)
            values.append(value)
    return values

def evaluate_one(agent, eval_qap):
    samples = 500
    print("Single task")
    print("Samples:", samples)

    print("Finding random average...")
    random_value = compute_random_average_value([eval_qap], samples=samples)

    print("Finding agent average...")
    agent_value = compute_agent_average_value(agent, [eval_qap], samples=samples)

    compute_optimum = eval_qap.size < 10
    if compute_optimum:
        print("Finding optimum...")
        optimal_value, optimal_assignment = solve_qap_backtracking(eval_qap)

    print("Comparison:")
    print("Random:", stat_string(random_value))
    print("Agent:", stat_string(agent_value))
    print("Sample agent assignment:",agent.solve(eval_qap))

    if compute_optimum:
        print("Optimal:", optimal_value, optimal_assignment)

def stat_string(values):
    return f"mean: {np.mean(values)}; min: {np.min(values)}; max: {np.max(values)}"

def compute_gaps(values, target_values):
    assert(len(values) == len(target_values))
    return [(v - t) / t for v, t in zip(values, target_values)]

def evaluate_set(agent, eval_set):
    samples = 100
    print("Tasks:", len(eval_set))
    print("Samples per task:", samples)

    print("Value Comparison:")
    random_values = compute_random_average_value(eval_set, samples=samples)
    print("[Random]", stat_string(random_values))
    best_of_k_values = compute_best_of_k_value(eval_set, samples=samples)
    print(f"[Best-of-{samples}]", stat_string(best_of_k_values))
    agent_values = compute_agent_average_value(agent, eval_set, samples=1)
    print("[Agent]", stat_string(agent_values))

    optimal_values = [qap.known_solution.value if qap.known_solution is not None else None for qap in eval_set]
    unknown_solution = False
    for v in optimal_values:
        if v is None:
            unknown_solution = True
            break

    if unknown_solution:
        print("Optimum unknown")
    else:
        print("[Optimum]", stat_string(optimal_values))
        print()
        print("Tasks:", ", ".join(qap.name for qap in eval_set))
        print("Gaps:")
        print(f"[Best-of-{samples}]", stat_string(compute_gaps(best_of_k_values, optimal_values)))
        print("[Agent]", stat_string(compute_gaps(agent_values, optimal_values)))

        print([int(x) for x in best_of_k_values])
        print([int(x) for x in agent_values])
        print([int(x) for x in optimal_values])


def main():
    if len(sys.argv) not in (2,3):
        print(f"Usage: {sys.argv[0]} checkpoint [problem_generator]")
        return 1

    checkpoint_path = sys.argv[1]

    agent_name = (Path(checkpoint_path).parent / "agenttype").read_text().strip()
    agent = agent_configs.agents[agent_name]()
    agent.load_checkpoint(sys.argv[1])

    if len(sys.argv) >= 3:
        problem_generator_name = sys.argv[2]
    else:
        problem_generator_name = (Path(checkpoint_path).parent / "trainingtask").read_text().strip()

    print(f"Evaluating agent '{agent_name}' on task '{problem_generator_name}'")

    problem_generator = taskgenerators.generators[problem_generator_name]()
    test_set = problem_generator.test_set()

    if len(test_set) == 1:
        evaluate_one(agent, test_set[0])
    else:
        evaluate_set(agent, test_set)

    return 0

if __name__ == "__main__":
    sys.exit(main())
