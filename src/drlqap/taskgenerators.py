import random
from drlqap import testgraphs
from drlqap.qap import GraphAssignmentProblem

generators = dict()

def define_problem_generator(f):
    assert f.__name__ not in generators
    generators[f.__name__] = f
    return f

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


@define_problem_generator
def small_random_graphs():
    def random_qap():
        a = testgraphs.create_random_graph(8, 1.0)
        b = testgraphs.create_random_graph(8, 1.0)
        return GraphAssignmentProblem(a, b)
    return RandomTaskGenerator(random_qap)

@define_problem_generator
def qaplib_bur26a():
    with open("qapdata/bur26a.dat") as f:
        qap = GraphAssignmentProblem.from_qaplib_string(f.read())
    return SingleTask(qap)

@define_problem_generator
def qaplib_bur26a_normalized():
    with open("qapdata/bur26a.dat") as f:
        qap = GraphAssignmentProblem.from_qaplib_string(f.read(), normalize=True)
    return SingleTask(qap)

@define_problem_generator
def small_fixed():
    with open("qapdata/testgraph.dat") as f:
        qap = GraphAssignmentProblem.from_qaplib_string(f.read())
    return SingleTask(qap)
