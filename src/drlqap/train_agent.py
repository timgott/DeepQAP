from drlqap.trainer import StatsCollector, Trainer
from pathlib import Path
from drlqap.taskgenerators import generators as tasks
import logging
import sys
from drlqap.agent_configs import agents, agent_training_steps
import numpy as np
import torch
import random

def create_experiment_folder(path: Path):
    basename = path.name
    counter = 1
    while True:
        try:
            path.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            counter += 1 # First duplicate gets _2
            path = path.with_name(f"{basename}_{counter}")

    return path

def train_agent(agent, problem_generator, experiment_path, training_steps, checkpoint_every):
    logging.basicConfig(filename=experiment_path / "run.log")

    stats_collector = StatsCollector(experiment_path)
    trainer = Trainer(agent, problem_generator, stats_collector)
    trainer.train(training_steps, checkpoint_every)

def print_usage():
    print(f"Usage: {sys.argv[0]} experiment_path agent_type problem_generator [seed]")
    print("Agent types:")
    for agent in agents.keys():
        print(f"- {agent}")
    print("Task generators:")
    for task in tasks.keys():
        print(f"- {task}")

def main():
    if len(sys.argv) not in [4, 5]:
        print_usage()
        return 1

    experiment_path = sys.argv[1]
    agent_name = sys.argv[2]
    task_name = sys.argv[3]
    seed = int(sys.argv[4]) if len(sys.argv) == 5 else 0

    # Fix random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if agent_name not in agents:
        print(f"Unknown agent type '{agent_name}'")
    if task_name not in tasks:
        print(f"Unknown task generator '{task_name}'")

    agent = agents[agent_name]()
    problem_generator = tasks[task_name]()
    training_steps = agent_training_steps[agent_name]

    experiment_path = create_experiment_folder(Path(experiment_path)) 
    print(f"Output folder: {experiment_path}")

    with open(experiment_path / 'agenttype', 'x') as f:
        f.write(agent_name)

    with open(experiment_path / 'trainingtask', 'x') as f:
        f.write(task_name)

    if seed != 0:
        with open(experiment_path / 'seed', 'x') as f:
            f.write(str(seed))

    train_agent(agent,
        problem_generator=problem_generator,
        experiment_path=experiment_path,
        training_steps=training_steps,
        checkpoint_every=1000
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
