from qap import GraphAssignmentProblem
from simplesolver import solve_qap_backtracking as solve
import testgraphs
from visualisation import draw_graph_assignment
import networkx as nx

a = testgraphs.create_random_graph(8, 0.8)
b = testgraphs.create_random_graph(8, 0.8)
qap = GraphAssignmentProblem(a, b)

value, assignment = solve(qap)
print(value, assignment)

if assignment is None:
    print("No assignment found.")
else:
    draw_graph_assignment(a, b, assignment)
    
