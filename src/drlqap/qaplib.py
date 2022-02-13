from pathlib import Path
from drlqap.qap import GraphAssignmentProblem

qapdata_path = Path("qapdata")
solutions_path = Path("qapsoln")

class QaplibSolution:
    def __init__(self, size, value, assignment):
        self.size = size
        self.value = value
        self.assignment = assignment

    @staticmethod
    def from_string(string):
        data = string.split()
        size = int(data[0])
        assert (len(data) == size + 2)
        value = int(data[1])
        assignment = [int(x)-1 for x in data[2:]]
        return QaplibSolution(size, value, assignment)


def get_solution_path(qap_path):
    return solutions_path / f"{qap_path.stem}.sln"

def load_qap(file_name, normalize=False):
    path = Path(file_name)

    # Search for solution
    try:
        with open(get_solution_path(path)) as f:
            solution = QaplibSolution.from_string(f.read())
    except FileNotFoundError:
        solution = None

    with open(path) as f:
        qap = GraphAssignmentProblem.from_qaplib_string(
            f.read(), 
            normalize=normalize, name=path.stem, known_solution=solution
        )
        assert(solution is None or qap.size == solution.size)
        return qap

def load_qap_set(file_names, normalize=False):
    return [load_qap(fname, normalize=normalize) for fname in file_names]

def load_qaplib_set(glob, normalize=False):
    paths = sorted(qapdata_path.glob(glob))
    return load_qap_set(paths, normalize=True)

