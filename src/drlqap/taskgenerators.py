import random
from drlqap import testgraphs
from drlqap.qap import GraphAssignmentProblem
from pathlib import Path

generators = dict()

def define_problem_generator(f):
    assert f.__name__ not in generators
    generators[f.__name__] = f
    return f

def load_qap(file_name, normalize=False):
    with open(file_name) as f:
        return GraphAssignmentProblem.from_qaplib_string(f.read(), normalize=normalize)

def load_qap_set(file_names, normalize=False):
    return [load_qap(fname, normalize=normalize) for fname in file_names]

class SingleTask:
    def __init__(self, task):
        self.task = task

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

@define_problem_generator
def small_random_graphs():
    return RandomWeightsTaskGenerator(8)

@define_problem_generator
def medium_random_graphs():
    return RandomWeightsTaskGenerator(16)

@define_problem_generator
def qaplib_bur26a():
    qap = load_qap("qapdata/bur26a.dat")
    return SingleTask(qap)

@define_problem_generator
def qaplib_bur26a_normalized():
    qap = load_qap("qapdata/bur26a.dat", normalize=True)
    return SingleTask(qap)

@define_problem_generator
def small_fixed():
    qap = load_qap("qapdata/testgraph.dat")
    return SingleTask(qap)

@define_problem_generator
def qaplib_all_bur():
    qaps = load_qap_set(sorted(Path("qapdata").glob("bur*.dat")))
    return FixedTaskSet(qaps)

@define_problem_generator
def qaplib_all_bur_normalized():
    qaps = load_qap_set(sorted(Path("qapdata").glob("bur*.dat")), normalize=True)
    return FixedTaskSet(qaps)
