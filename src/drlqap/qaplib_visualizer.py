import sys

from networkx.drawing.nx_pylab import draw
from drlqap.visualisation import draw_graph_assignment, SubgraphVisualisation
from drlqap.qap import GraphAssignmentProblem as QAP
import matplotlib.pyplot as plt

with open(sys.argv[1]) as f:
    qap = QAP.from_qaplib_string(f.read(), normalize=True)
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)
    vis1 = SubgraphVisualisation(qap.graph_source)
    vis1.put_subgraph(qap.graph_source, node_color="green", edge_style="solid")
    vis1.draw(ax=ax1)

    vis2 = SubgraphVisualisation(qap.graph_target)
    vis2.put_subgraph(qap.graph_target, node_color="cyan", edge_style="solid")
    vis2.draw(ax=ax2)

    plt.show()
