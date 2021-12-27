from typing import List
import networkx as nx
import numpy as np
from networkx import Graph
import torch

class QAP:
    """
    Represents a quadratic assignment problem with linear term of the form
    ∑ᵢ≠ⱼ (Aᵢ,ⱼ*Bₚ₍ᵢ₎,ₚ₍ⱼ₎) + ∑ᵢ (Lᵢ,ₚ₍ᵢ₎) + C
    """
    A: torch.Tensor
    B: torch.Tensor
    linear_costs: torch.Tensor
    fixed_cost: float
    size: int

    def __init__(self, A: torch.Tensor, B: torch.Tensor, linear_costs: torch.Tensor, fixed_cost: float) -> None:
        # Require 0 diagonals; Any problem can be rewritten
        assert(torch.all(A.diag() == 0))
        assert(torch.all(B.diag() == 0))
        
        self.size = A.size(0)
        assert(A.shape == (self.size, self.size))
        assert(B.shape == A.shape)
        assert(linear_costs.shape == A.shape)

        self.A = A
        self.B = B
        self.linear_costs = linear_costs
        self.fixed_cost = fixed_cost

    def compute_value(self, assignment):
        i = torch.arange(self.size)
        return torch.sum(self.A * self.B[assignment,:][:,assignment]) \
            + torch.sum(self.linear_costs[i,assignment]) \
            + self.fixed_cost

    # I/O
    def from_qaplib_string(qaplib_str, normalize=False):
        data = qaplib_str.split()
        size = int(data[0])

        assert len(data) == 1 + 2 * size*size

        def parse_matrix(data, offset, size):
            # format and optionally normalize
            array = torch.tensor(data[offset:offset + size*size]).reshape((size, size))
            if normalize:
                lower = torch.min(array)
                upper = torch.max(array)
                return (array - lower) / (upper - lower)
            else:
                return array

        float_data = [float(x) for x in data[1:]]
        A = parse_matrix(float_data, 0, size)
        B = parse_matrix(float_data, size*size, size)

        assert A.shape == (size, size)
        assert B.shape == (size, size)

        L = torch.matmul(A.diag().reshape(size, 1), B.diag().reshape(1, size))

        A.fill_diagonal_(0)
        B.fill_diagonal_(0)

        return QAP(A, B, L, 0)

    def create_subproblem_for_assignment(self, a, b):
        """
        Create a the new QAP that represents the problem after assigning a->b
        """

        new_size = self.size - 1

        # New constant term is computed from previous linear_term of the assigned nodes
        constant = self.fixed_cost + self.linear_costs[a, b]

        # all nodes in A except a; all ndoes in B except b
        not_a = list((*range(0, a), *range(a+1,self.size)))
        not_b = list((*range(0, b), *range(b+1,self.size)))

        # New linear cost is computed from quadratic cost from/towards the assigned nodes
        incoming_costs = self.A[a, not_a].reshape(new_size, 1) * self.B[b, not_b].reshape(1, new_size)
        outgoing_costs = self.A[not_a, a].reshape(new_size, 1) * self.B[not_b, b].reshape(1, new_size)
        new_linear_cost = incoming_costs + outgoing_costs + self.linear_costs[not_a,:][:,not_b]

        return QAP(A=self.A[not_a][:,not_a], B=self.B[not_b][:,not_b], linear_costs=new_linear_cost, fixed_cost=constant)


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
