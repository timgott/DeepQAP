from networkx import Graph
import networkx
import random

def create_ring(size):
    """
    Creates a ring with weight 1
    """
    graph = Graph()
    for i in range(size):
        next_i = (i + 1) % size
        graph.add_edge(i,next_i, weight=1.0)

    return graph

def create_single_edge(size):
    graph = Graph()
    graph.add_nodes_from(range(size))
    graph.add_edge(0, 1, weight=1.0)
    return graph

def create_chain(size, chain_length):
    assert chain_length < size
    graph = Graph()
    graph.add_nodes_from(range(size))
    for i in range(chain_length):
        graph.add_edge(i, i+1, weight=1.0)
    return graph

def create_random_graph(size, p):
    graph = networkx.gnp_random_graph(size, p)
    for u,v,data in graph.edges(data=True):
        data["weight"] = random.uniform(0.0, 1.0)
    return graph
