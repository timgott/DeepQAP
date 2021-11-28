import testgraphs
from qap import GraphAssignmentProblem

generators = dict()

def problem_generator(f):
    assert f.__name__ not in generators
    generators[f.__name__] = f

@problem_generator
def small_random_graphs():
    a = testgraphs.create_random_graph(8, 1.0)
    b = testgraphs.create_random_graph(8, 1.0)
    return GraphAssignmentProblem(a, b)

@problem_generator
def qaplib_bur26a():
    with open("qapdata/bur26a.dat") as f:
        return GraphAssignmentProblem.from_qaplib_string(f.read())
