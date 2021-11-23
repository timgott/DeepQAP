import math
from typing import List
import networkx as nx
import numpy as np
from networkx import Graph
from networkx.classes import graph

class GraphAssignmentProblem:
    graph_source: Graph
    graph_target: Graph
    size: int
    assignment: List[int]
    unassigned_variables: List[int]
    unassigned_targets: List[int]

    # edge_weights: (start, end): weight
    def __init__(self, graph_source: Graph, graph_target: Graph):
        assert graph_source.number_of_nodes() == graph_target.number_of_nodes()
        assert nx.is_weighted(graph_source)
        assert nx.is_weighted(graph_target)

        self.graph_source = graph_source
        self.graph_target = graph_target
        self.size = min(graph_source.number_of_nodes(), graph_target.number_of_nodes())
        self.assignment = [None] * self.size
        self.unassigned_variables = list(range(self.size))
        self.unassigned_targets = list(range(self.size))

    def assign(self, x: int, y: int):
        self.assignment[x] = y
        self.unassigned_variables.remove(x)
        self.unassigned_targets.remove(y)

    def unassign(self, x: int):
        self.unassigned_targets.append(self.assignment[x])
        self.assignment[x] = None
        self.unassigned_variables.append(x)

    def get_unassigned_nodes(self):
        return self.unassigned_variables

    def get_graph(self):
        pass

    def is_fully_assigned(self):
        return not self.unassigned_variables

    def compute_value(self):
        value: int = 0
        for x1, x2, weight in self.graph_source.edges.data("weight"):
            y1, y2 = self.assignment[x1], self.assignment[x2]
            if self.graph_target.has_edge(y1, y2):
                value += weight * self.graph_target.edges[y1, y2]["weight"]
            else:
                value = math.inf

        return value

    # I/O
    def from_qaplib_string(qaplib_str, normalize=False):
        data = qaplib_str.split()
        size = int(data[0])

        assert len(data) == 1 + 2 * size*size

        def parse_matrix(data, offset, size):
            # format and optionally normalize
            array = np.array(data[offset:offset + size*size]).reshape((size, size))
            if normalize:
                return array / np.max(array)
            else:
                return array

        float_data = [float(x) for x in data[1:]]
        graph_a = nx.from_numpy_array(parse_matrix(float_data, 0, size))
        graph_b = nx.from_numpy_array(parse_matrix(float_data, size*size, size))

        assert graph_a.number_of_nodes() == size
        assert graph_b.number_of_nodes() == size

        return GraphAssignmentProblem(graph_a, graph_b)

    def to_qaplib_string(self):
        def matrix_to_string(array):
            return "\n".join(" ".join(str(x) for x in row) for row in array)

        data = [
            str(self.size),
            matrix_to_string(nx.to_numpy_array(self.graph_source)),
            matrix_to_string(nx.to_numpy_array(self.graph_target))
        ]

        return data.join("\n\n")