import numpy as np
import random
from pathlib import Path
import sys
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
        for i in range(samples):
           values.append(qap.compute_unscaled_value(random_assignment(qap)))
    return values

def compute_agent_average_value(agent, qaps, samples=100):
    values = []
    if isinstance(agent, DQNAgent):
        samples = 1
    for qap in qaps:
        for i in range(samples):
            _, assignment = agent.solve(qap)
            value = qap.compute_unscaled_value(assignment).item()
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
    print("Random:", random_value)
    print("Agent:", agent_value)
    print("Sample agent assignment:", agent.solve(eval_qap))

    if compute_optimum:
        print("Optimal:", optimal_value, optimal_assignment)

def stat_string(values):
    if len(values) < 10:
        return ", ".join(str(x) for x in values)
    else:
        return f"mean: {np.mean(values)}; min: {np.min(values)}; max: {np.max(values)}"

def evaluate_set(agent, eval_set):
    samples = 10
    print("Tasks:", len(eval_set))
    print("Samples per task:", samples)

    print("Finding random average...")
    random_values = compute_random_average_value(eval_set, samples=samples)

    print("Finding agent average...")
    agent_values = compute_agent_average_value(agent, eval_set, samples=samples)

    print("Comparison:")
    print("Random:", stat_string(random_values))
    print("Agent:", stat_string(agent_values))

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
