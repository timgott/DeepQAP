from drlqap import testgraphs
from drlqap.qap import GraphAssignmentProblem

generators = dict()

def define_problem_generator(f):
    assert f.__name__ not in generators
    generators[f.__name__] = f
    return f

@define_problem_generator
def small_random_graphs():
    a = testgraphs.create_random_graph(8, 1.0)
    b = testgraphs.create_random_graph(8, 1.0)
    return GraphAssignmentProblem(a, b)

@define_problem_generator
def qaplib_bur26a():
    with open("qapdata/bur26a.dat") as f:
        return GraphAssignmentProblem.from_qaplib_string(f.read())

@define_problem_generator
def qaplib_bur26a_normalized():
    with open("qapdata/bur26a.dat") as f:
        return GraphAssignmentProblem.from_qaplib_string(f.read(), normalize=True)

@define_problem_generator
def small_fixed():
    with open("qapdata/testgraph.dat") as f:
        return GraphAssignmentProblem.from_qaplib_string(f.read())
