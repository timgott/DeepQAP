from drlqap.qap import QAP
import gurobipy as gp
from gurobipy import GRB

def solve_qap_gurobi(qap: QAP, time_limit=None, verbose=False):
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
        gp.quicksum(
            qap.A[i, j] * qap.B[u, v] * x[i, u] * x[j, v]
            for i in range(n)
            for j in range(n)
            for u in range(n)
            for v in range(n)
        )
    )

    if time_limit:
        m.Params.TimeLimit = time_limit
    m.Params.OutputFlag = 1 if verbose else 0
    m.optimize()
    
    assignment = [None] * qap.size
    for i,j in x:
        if x[i,j].X:
            assignment[i] = j
    return m.ObjVal, assignment