from trainer import StatsCollector, Trainer
from pathlib import Path
import taskgenerators
import logging
import sys
import agent_configs
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
            path = path.with_name(f"{basename}_{counter}")
            counter += 1
    
    return path

def train_agent(agent, problem_generator, experiment_path, training_steps, checkpoint_every):
    experiment_path = create_experiment_folder(Path(experiment_path))
    print(f"Output folder: {experiment_path}")

    logging.basicConfig(filename=experiment_path / "run.log")

    stats_collector = StatsCollector(experiment_path)
    trainer = Trainer(agent, problem_generator, stats_collector)
    trainer.train(training_steps, checkpoint_every)

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} experiment_path agent_type problem_generator")
        return 1

    # Fix random seed
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    experiment_path = sys.argv[1]
    agent = agent_configs.agents[sys.argv[2]]()
    problem_generator = taskgenerators.generators[sys.argv[2]]
    train_agent(agent, 
        problem_generator=problem_generator,
        experiment_path=experiment_path,
        training_steps=1000,
        checkpoint_every=100
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())