import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.shortest_paths import weighted
from qap import AssignmentGraph, GraphAssignmentProblem

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

    def draw(self, **kwargs):
        nx.draw_networkx(
            self.graph,
            edgelist=self.edgelist,
            nodelist=self.nodelist,
            #style=self.edge_styles, #fails in jupyter
            node_color=self.node_colors,
            width=self.edge_widths,
            edge_color=self.edge_colors,
            edge_cmap = plt.get_cmap("Oranges"),
            **kwargs
        )

def draw_qap(qap: GraphAssignmentProblem, assignment):
    assignment_graph = AssignmentGraph(qap.graph_source, qap.graph_target, assignment)
    draw_assignment_graph(assignment_graph)


def draw_assignment_graph(assignment_graph: AssignmentGraph):
    visualization = SubgraphVisualisation(assignment_graph.graph)
    visualization.put_subgraph(assignment_graph.subgraph_a, node_color=(1,0.5,0), edge_style="solid")
    visualization.put_subgraph(assignment_graph.subgraph_b, node_color=(0,0.5,1), edge_style="solid")
    visualization.put_edges(assignment_graph.assignment_edges, style="dotted")
    visualization.draw()
    plt.show()