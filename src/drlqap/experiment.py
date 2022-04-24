from drlqap.trainer import StatsCollector, Trainer
from pathlib import Path
from drlqap.taskgenerators import generators as tasks
import logging
from drlqap.agent_configs import agents, agent_training_steps
import numpy as np
import torch
import random
import json

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

def run_experiment(experiment_path, agent, task_generator, training_steps, metadata, seed=0):
    # Fix random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    experiment_path = create_experiment_folder(Path(experiment_path)) 
    print(f"Output folder: {experiment_path}")

    with open(experiment_path / 'metadata.json', 'x') as f:
        json.dump(metadata, f)

    if seed != 0:
        with open(experiment_path / 'seed', 'x') as f:
            f.write(str(seed))

    train_agent(agent,
        problem_generator=task_generator,
        experiment_path=experiment_path,
        training_steps=training_steps,
        checkpoint_every=1000
    )

def run_experiment_from_config(experiment_path, agent_name, task_name, agent_arguments, seed=0, steps=None):
    agent = agents[agent_name](**agent_arguments)
    problem_generator = tasks[task_name]
    training_steps = steps or agent_training_steps[agent_name]

    metadata = {
        'agent_type': agent_name,
        'agent_arguments': agent_arguments,
        'training_task': task_name,
        'training_steps': training_steps,
        'seed': seed,
    }

    run_experiment(
        experiment_path=experiment_path,
        agent=agent,
        task_generator=problem_generator,
        training_steps=training_steps,
        metadata=metadata,
        seed=seed
    )


def load_metadata(experiment_path):
    """
    Parses the metadata file of the experiment.
    """
    try:
        with open(experiment_path / "metadata.json") as f:
            return json.load(f)
    except FileNotFoundError:
        # Backwards compatibility
        metadata = {}
        with open(experiment_path / "agenttype") as f:
            metadata["agent_type"] = f.read().strip()
        with open(experiment_path / "trainingtask") as f:
            metadata["training_task"] = f.read().strip()
        try:
            with open(experiment_path / "seed") as f:
                metadata["seed"] = int(f.read())
        except FileNotFoundError:
            metadata["seed"] = 0
        return metadata


# Helper functions to determine label automatically
def get_overlapping_keys(dicts):
    return [key for key, _ in set.intersection(*(set(d.items()) for d in dicts))]

def get_key_value_string(dict, ignored_keys):
    return ", ".join(sorted([f'{key}: {dict[key]}' for key in dict if key not in ignored_keys]))
