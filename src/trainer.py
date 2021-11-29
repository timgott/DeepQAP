from typing import Callable

import torch
from qap import GraphAssignmentProblem
import random
import os

class StatsCollector:
    def __init__(self, path) -> None:
        self.path = path
        self.data = dict()

    def start(self):
        pass

    def finish(self):
        pass

    def put(self, key, value):
        if not key in self.data:
            self.data[key] = [value]
        else:
            self.data[key].append(value)

    def save(self):
        for key in self.data:
            with open(self.path / f"{key}.txt", mode="a") as f:
                f.write("\n".join(str(x) for x in self.data[key]))
                self.data[key] = []

    def finish(self):
        self.save()
        

class Trainer:
    def __init__(self, agent, problem_generator: Callable[[], GraphAssignmentProblem], stats_collector):
        self.agent = agent
        self.problem_generator = problem_generator
        self.stats_collector = stats_collector

    def save_checkpoint(self, name):
        self.agent.save_checkpoint(self.stats_collector.path / f"checkpoint_{name}.pth")

    def train(self, steps=1, checkpoint_every=None):
        self.stats_collector.start()

        for i in range(steps):
            # Generate Problem
            qap = self.problem_generator()

            # Let agent learn on problem
            self.agent.solve_and_learn(qap)

            # Store statistics
            stats = self.agent.get_episode_stats()
            for key, value in stats.items():
                self.stats_collector.put(key, value)

            # Backup agent after some time
            if checkpoint_every is not None and i % checkpoint_every == 0:
                self.save_checkpoint(str(i))
                self.stats_collector.save()

            if (i+1) % 100 == 0:
                print(f"Step {i+1}/{steps} finished")

        self.save_checkpoint("end")

        self.stats_collector.finish()
