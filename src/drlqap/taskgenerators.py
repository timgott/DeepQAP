import random
from drlqap import testgraphs
from drlqap.qap import GraphAssignmentProblem
from drlqap.qaplib import load_qaplib_set, load_qap

generators = dict()

def define_problem_generator(f):
    assert f.__name__ not in generators
    generators[f.__name__] = f
    return f

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

class LazyGlobTaskGenerator():
    def __init__(self, glob, normalize=False):
        self.glob = glob
        self.normalize = normalize
        self.tasks = None

    def get_tasks(self):
        if self.tasks is None:
            self.tasks = load_qaplib_set(self.glob, normalize=True)
        return self.tasks

    def sample(self):
        return random.choice(self.get_tasks())

    def test_set(self):
        return self.get_tasks()

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
    qaps = load_qaplib_set("bur*.dat")
    return FixedTaskSet(qaps)

@define_problem_generator
def qaplib_sko_42_64_normalized():
    return FixedTaskSet(load_qaplib_set("sko[456]?.dat", normalize=True))

@define_problem_generator
def qaplib_all_sko_normalized():
    return 

def define_qaplib_task(category):
    return lambda: LazyGlobTaskGenerator(f"{category}*.dat", normalize=True)

qaplib_categories = ['bur', 'chr', 'esc', 'had', 'kra', 'lipa', 'nug', 'rou', 'scr', 'sko', 'ste', 'tai', 'tho', 'wil']
for category in qaplib_categories:
    generators[f"qaplib_all_{category}_normalized"] = define_qaplib_task(category)
