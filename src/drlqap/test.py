from drlqap.qap import GraphAssignmentProblem
from drlqap.simplesolver import solve_qap_backtracking as solve
from drlqap import testgraphs
from drlqap.visualisation import draw_qap

a = testgraphs.create_random_graph(8, 1.0)
b = testgraphs.create_random_graph(8, 1.0)
qap = GraphAssignmentProblem(a, b)

value, assignment = solve(qap)
print(value, assignment)

if assignment is None:
    print("No assignment found.")
else:
    draw_qap(qap, assignment)
    
