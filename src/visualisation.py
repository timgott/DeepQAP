import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.shortest_paths import weighted

class AssignmentGraph:
    def __init__(self, a: nx.Graph, b: nx.Graph, assignment) -> None:
        # Rename nodes in b for joining
        offset = len(a)
        mapping = dict(zip(b, range(offset, len(b) + offset)))
        b_relabeled = nx.relabel_nodes(b, mapping)

        # Create joined graph
        self.graph = nx.union(a, b_relabeled)

        # Add assignment edges
        self.assignment_edges = [(x, mapping[y]) for x,y in assignment]
        self.graph.add_edges_from(self.assignment_edges)

        self.subgraph_a = self.graph.subgraph(a.nodes)
        self.subgraph_b = self.graph.subgraph(b_relabeled.nodes)

class SubgraphVisualisation:
    def __init__(self, graph) -> None:
        self.graph = graph
        self.edgelist = []
        self.nodelist = []
        self.node_colors = []
        self.edge_styles = []
        self.edge_widths = []
        self.edge_colors = []

    def put_edges(self, edges, style, width=1.0, color=0.5):
        self.edgelist += edges
        self.edge_styles += [style] * len(edges)
        self.edge_widths += [width] * len(edges)
        self.edge_colors += [color] * len(edges)

    def put_weighted_edges(self, edges, style):
        self.edgelist += edges
        self.edge_styles += [style] * len(edges)

        weights = [w for _,_,w in edges.data("weight")]
        # weight to line width
        self.edge_widths += [(w + 0.5) for w in weights]
        # weight to line alpha
        self.edge_colors += [w for w in weights]

    def put_nodes(self, nodes, color):
        self.nodelist += nodes
        self.node_colors += [color] * len(nodes)

    def put_subgraph(self, subgraph, node_color, edge_style):
        self.put_weighted_edges(subgraph.edges, edge_style)
        self.put_nodes(subgraph.nodes, node_color)

    def draw(self):
        nx.draw_networkx(
            self.graph,
            edgelist=self.edgelist,
            nodelist=self.nodelist,
            style=self.edge_styles,
            node_color=self.node_colors,
            width=self.edge_widths,
            edge_color=self.edge_colors,
            edge_cmap = plt.get_cmap("Oranges")
        )


def draw_graph_assignment(a: nx.Graph, b: nx.Graph, assignment):
    assignment_graph = AssignmentGraph(a, b, assignment)
    visualization = SubgraphVisualisation(assignment_graph.graph)
    visualization.put_subgraph(assignment_graph.subgraph_a, node_color=(1,0.5,0), edge_style="solid")
    visualization.put_subgraph(assignment_graph.subgraph_b, node_color=(0,0.5,1), edge_style="solid")
    visualization.put_edges(assignment_graph.assignment_edges, style="dotted")
    visualization.draw()
    plt.show()