from drlqap.qap import QAP
import gurobipy as gp
from gurobipy import GRB

def create_gurobi_model(qap: QAP):
    m = gp.Model()
    n = qap.size

    # Assignment flag variables
    x = m.addVars(n, n, vtype=GRB.BINARY)

    # Constraints
    # x defines a permutation matrix (every a assigned to exactly one b)
    m.addConstrs((sum(x[i,j] for j in range(n)) == 1 for i in range(n)))
    m.addConstrs((sum(x[i,j] for i in range(n)) == 1 for j in range(n)))

    # Objective
    m.setObjective(
        sum(
            qap.A[i, j] * qap.B[u, v] * x[i, u] * x[j, v]
            for i in range(n)
            for j in range(n)
            for u in range(n)
            for v in range(n)
        )
    )

    return m


def solve_qap_gurobi(qap: QAP):
    m = create_gurobi_model(qap)
    m.optimize()
