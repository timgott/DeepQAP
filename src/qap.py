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

    def compute_value(self, assignment):
        value: int = 0
        for x1, x2, weight in self.graph_source.edges.data("weight"):
            y1, y2 = assignment[x1], assignment[x2]
            assert(y1 in self.graph_target), f"{y1} not in graph ({list(self.graph_target)})"
            assert(y2 in self.graph_target), f"{y2} not in graph ({list(self.graph_target)})"
            if self.graph_target.has_edge(y1, y2):
                value += weight * self.graph_target.edges[y1, y2]["weight"]
            # edge not in graph => weight 0

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
                lower = np.min(array)
                upper = np.max(array)
                return (array - lower) / (upper - lower)
            else:
                return array

        float_data = [float(x) for x in data[1:]]
        connectivity_a = parse_matrix(float_data, 0, size)
        connectivity_b = parse_matrix(float_data, size*size, size)
        graph_a = nx.from_numpy_array(connectivity_a, create_using=nx.DiGraph)
        graph_b = nx.from_numpy_array(connectivity_b, create_using=nx.DiGraph)

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

        return "\n\n".join(data)

class AssignmentGraph:
    def __init__(self, a: nx.Graph, b: nx.Graph, assignment) -> None:
        # Rename nodes in b for joining
        offset = len(a)
        self.mapping = dict(zip(b, range(offset, len(b) + offset)))
        b_relabeled = nx.relabel_nodes(b, self.mapping)

        # Create joined graph
        self.graph = nx.union(a, b_relabeled)

        # Add assignment edges
        self.assignment_edges = [(x, self.mapping[y]) for x,y in enumerate(assignment)]
        self.graph.add_edges_from(self.assignment_edges, weight=1.0)

        self.subgraph_a = self.graph.subgraph(a.nodes)
        self.subgraph_b = self.graph.subgraph(b_relabeled.nodes)

        nx.set_node_attributes(self.subgraph_a, 1.0, "side")
        nx.set_node_attributes(self.subgraph_b, -1.0, "side")

    def add_assignment(self, x, y):
        mapped_y = self.map_target_node(y)
        self.graph.add_edge(x, mapped_y, weight=1.0)
        self.assignment_edges.append((x,mapped_y))

    def map_target_node(self, y):
        return self.mapping[y]

    def map_source_node(self, x):
        return x