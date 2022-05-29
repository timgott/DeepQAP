import random

import torch
from drlqap import testgraphs
from drlqap.qap import GraphAssignmentProblem, QAP
from drlqap.qaplib import load_qaplib_set, load_qap

generators = dict()

class SingleTask:
    def __init__(self, task, solution=None):
        self.task = task
        self.solution = solution

    def sample(self):
        return self.task

    def test_set(self):
        return [self.task]

class FixedTaskSet:
    def __init__(self, tasks):
        self.tasks = tasks

    def sample(self):
        return random.choice(self.tasks)

    def test_set(self):
        return self.tasks

class RandomTaskGenerator:
    def __init__(self, generator):
        self.generator = generator

    def sample(self):
        return self.generator()

    def test_set(self):
        return [self.generator() for _ in range(500)]

class RandomWeightsTaskGenerator(RandomTaskGenerator):
    def __init__(self, size):
        def random_qap():
            a = testgraphs.create_random_graph(size, 1.0)
            b = testgraphs.create_random_graph(size, 1.0)
            return GraphAssignmentProblem(a, b)
        super().__init__(random_qap)
        
class LinearTaskGenerator(RandomTaskGenerator):
    def __init__(self, size):
        def random_qap():
            a = torch.zeros((size, size))
            b = torch.zeros((size, size))
            l = torch.rand((size, size))
            return QAP(a, b, l, 0)
        super().__init__(random_qap)

class LazyGlobTaskGenerator():
    def __init__(self, glob, normalize=False, filter=None):
        self.glob = glob
        self.normalize = normalize
        self.tasks = None
        self.filter = filter

    def get_tasks(self):
        if self.tasks is None:
            self.tasks = load_qaplib_set(self.glob, normalize=self.normalize)
            if self.filter:
                self.tasks = [qap for qap in self.tasks if self.filter(qap)]
            assert self.tasks, f"No qap file matching {self.glob}"
        return self.tasks

    def sample(self):
        return random.choice(self.get_tasks())

    def test_set(self):
        return self.get_tasks()

def triangle_generator():
    a = testgraphs.create_chain(3, 2)
    b = testgraphs.create_chain(3, 1)
    return GraphAssignmentProblem(a, b)

qaplib_categories = ['bur', 'chr', 'esc', 'had', 'kra', 'lipa', 'nug', 'rou', 'scr', 'sko', 'ste', 'tai', 'tho', 'wil']
generators = {
    'minilinear': LinearTaskGenerator(2),
    'small_random_graphs': RandomWeightsTaskGenerator(8),
    'medium_random_graphs': RandomWeightsTaskGenerator(16),
    'qaplib_bur26a': LazyGlobTaskGenerator("bur26a.dat"),
    'qaplib_bur26a_normalized': LazyGlobTaskGenerator("bur26a.dat", normalize=True),
    'small_fixed': LazyGlobTaskGenerator("testgraph.dat"),
    'triangle': SingleTask(triangle_generator()),
    'qaplib_all_bur': LazyGlobTaskGenerator("bur*.dat", normalize=False),
    'qaplib_sko_42_64_normalized': LazyGlobTaskGenerator("sko[456]?.dat", normalize=True),
    **{
        f"qaplib_all_{category}_normalized": LazyGlobTaskGenerator(f"{category}*.dat", normalize=True)
        for category in qaplib_categories
    },
    'qaplib_tai35a_normalized': LazyGlobTaskGenerator("tai35a.dat", normalize=False),
    'qaplib_all_64_normalized': LazyGlobTaskGenerator("*.dat", normalize=False, filter=lambda qap: qap.size <= 64),
}
