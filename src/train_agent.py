from qap import GraphAssignmentProblem
from trainer import StatsCollector, Trainer
from pathlib import Path
import testgraphs

def problem_generator():
    a = testgraphs.create_random_graph(8, 1.0)
    b = testgraphs.create_random_graph(8, 1.0)
    return GraphAssignmentProblem(a, b)

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

def train_agent(agent, experiment_path, training_steps, checkpoint_every):
    experiment_path = create_experiment_folder(Path(experiment_path))

    stats_collector = StatsCollector(experiment_path)
    trainer = Trainer(agent, problem_generator, stats_collector)
    trainer.train(training_steps, checkpoint_every)