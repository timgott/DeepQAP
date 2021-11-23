from typing import Callable
from qap import GraphAssignmentProblem
import random

class Trainer:
    def __init__(self, agent, problem_generator: Callable[[], GraphAssignmentProblem], stats_collector):
        self.agent = agent
        self.problem_generator = problem_generator
        self.stats_collector = stats_collector

    def train(self, steps=1, checkpoint_every=None):
        self.stats_collector.start()

        for i in range(steps):
            # Generate Problem
            qap = self.problem_generator()

            # Let agent learn on problem
            self.agent.solve_and_learn(qap)

            # Store statistics
            stats = self.agent.episode_stats()
            for key, value in stats.items():
                self.stats_collector.serial(key, value)

            # Backup agent after some time
            if i % checkpoint_every == 0:
                checkpoint = self.agent.checkpoint()
                self.stats_collector.checkpoint(str(i), checkpoint)

        checkpoint = self.agent.checkpoint()
        self.stats_collector.checkpoint("end", checkpoint)

        self.stats_collector.finish()
