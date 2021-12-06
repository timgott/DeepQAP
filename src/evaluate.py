import numpy as np
import random
from qap import GraphAssignmentProblem
from reinforce import ReinforceAgent
from simplesolver import solve_qap_backtracking
from pathlib import Path
import sys
import taskgenerators
import agent_configs


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
        value, assignment = agent.solve(qap)
        values.append(value)
    return np.mean(values), np.min(values), np.max(values)

def evaluate(agent, problem_generator):
    eval_qap = problem_generator()

    samples = 500
    print("Samples:", samples)
    
    print("Finding random average...")
    random_value = compute_random_average_value(eval_qap, samples=samples)

    print("Finding agent average...")
    agent_value = compute_agent_average_value(agent, eval_qap, samples=samples)

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
    
    problem_generator = taskgenerators.generators[problem_generator_name]
    evaluate(agent, problem_generator)
    return 0

if __name__ == "__main__":
    sys.exit(main())