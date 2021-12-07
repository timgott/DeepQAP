import networkx
import torch
from torch.nn import ModuleList, Sequential
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data as GraphData
import torch.nn.functional as F

"""
Concatenate the embedding of each edge with the embedding of its reverse edge.

input: (n,n,d) n: number of nodes

output: (n,n,d*2)
"""
def concat_bidirectional(dense_edge_features: torch.Tensor) -> torch.Tensor:
    transposed = dense_edge_features.transpose(0, 1)
    return torch.cat((dense_edge_features, transposed), dim=2)

def sum_incoming_edges(dense_edge_features: torch.Tensor):
    return torch.sum(dense_edge_features, dim=0)

def aggregate_outgoing_edges(dense_edge_features: torch.Tensor):
    return torch.sum(dense_edge_features, dim=1)

def dense_edge_features_to_sparse(connectivity_matrix, feature_tensor):
    indices = connectivity_matrix.nonzero(as_tuple=True)
    edge_features = feature_tensor[indices]
    indices_tensor = torch.stack(indices)
    return indices_tensor, edge_features

# aka one-hot encoding
def edge_histogram_embeddings(connectivity, bins):
    lower = torch.min(connectivity)
    upper = torch.max(connectivity)
    step = (upper - lower)/bins
    range = torch.arange(lower, upper - step/2, step=step).reshape(1,1,-1)
    connectivity3d = connectivity.unsqueeze(2)
    return torch.where((connectivity3d >= range) & (connectivity3d < range + step), 1.0, 0.0)
    
def bidirectional_edge_histogram_embeddings(connectivity):
    incoming = edge_histogram_embeddings(connectivity)
    outgoing = edge_histogram_embeddings(connectivity.T)
    return torch.cat(incoming, outgoing, dim=1)


# Create matrix where entry ij is cat(a_i,b_j)
def cartesian_product_matrix(a, b):
    assert(a.shape == b.shape)
    
    n = a.shape[0]

    # Repeat a along dimension 1
    a_rows = a.unsqueeze(1).expand(-1, n, -1)

    # Repeat b along dimension 0
    b_columns = b.unsqueeze(0).expand(n, -1, -1)

    return torch.cat((a_rows, b_columns), dim=2)


class NodeTransformer(torch.nn.Module):
    def __init__(self, node_embedding_size, hidden_channels, edge_embedding_size):
        super().__init__()

        self.layers = ModuleList([
            TransformerConv(node_embedding_size, hidden_channels, edge_dim=edge_embedding_size),
            TransformerConv(hidden_channels, hidden_channels, edge_dim=edge_embedding_size),
        ])
        
        self.skip_linear = torch.nn.Linear(len(self.layers) * hidden_channels, node_embedding_size)

    def forward(self, x, edge_index, edge_attr):
        intermediates = []
        for network in self.layers:
            x = network(x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.elu(x)
            intermediates.append(x)
        
        # concatenate layer output for every node
        skip_xs = torch.cat(intermediates, dim=-1)
        x = self.skip_linear(skip_xs)
        x = F.elu(x)
        return x

def FullyConnected(input_channels, hidden_channels, output_channels, depth, activation):
    return FullyConnectedShaped(
        [input_channels] + [hidden_channels] * depth + [output_channels],
        activation
    )

def FullyConnectedShaped(shape, activation):
    layers = Sequential()
    for i, (inputs, outputs) in enumerate(zip(shape, shape[1:])):
        layers.add_module(f"linear[{i}]({inputs}->{outputs})", torch.nn.Linear(inputs, outputs))
        layers.add_module(f"activation[{i}]", activation())
    return layers

class Bidirectional(torch.nn.Module):
    def __init__(self, inner) -> None:
        super().__init__()
        self.inner = inner
    
    def forward(self, matrix):
        # mirror edge weights to create swap-symmetrical tuples
        bidirectional_matrix = concat_bidirectional(matrix.unsqueeze(2))
        # compute edge embedding vectors from weight values
        return self.inner(bidirectional_matrix)

class QAPNet(torch.nn.Module):
    def __init__(self, 
        edge_encoder,
        initial_node_encoder,
        link_probability_net,
        link_encoder,
        message_passing_net=None,
    ) -> None:
        super().__init__()

        # Allow to generate histogram-like embeddings, that are more meaningful
        # than scalars when aggregated
        self.edge_encoder = edge_encoder

        # Initial summed edge embedding -> node embedding network
        self.initial_node_encoder = initial_node_encoder

        # Message passing node embedding net
        self.message_passing_net = message_passing_net
        assert self.message_passing_net == None, "Message passing currently not implemented"

        # Network that computes linked embeddings of assigned nodes
        # input: concatenated node embeddings
        # output: node embedding
        self.link_encoder = link_encoder

        # Network that computes logit probability that two nodes should be linked
        # (asymmetric!)
        self.link_probability_net = link_probability_net

    def transform_initial_graph(self, nx_graph):
        # networkx graph to connectivity matrix
        connectivity_matrix = torch.tensor(networkx.linalg.adjacency_matrix(nx_graph).todense()).float()
        # compute edge encoding
        edge_embeddings = self.edge_encoder(connectivity_matrix)
        # aggregate edges and compute initial node embeddings
        aggregated = sum_incoming_edges(edge_embeddings)
        base_embeddings = self.initial_node_encoder(aggregated)
        # matrix format to adjacency and attribute list
        edge_index, edge_attr = dense_edge_features_to_sparse(connectivity_matrix, edge_embeddings)
        # Create data object required for torch geometric
        data = GraphData(x=base_embeddings, edge_index=edge_index, edge_attr=edge_attr)
        return data

    def initial_step(self, qap):
        # Initial node and edge embeddings
        data_a = self.transform_initial_graph(qap.graph_source)
        data_b = self.transform_initial_graph(qap.graph_target)
        return (data_a, data_b)

    def compute_link_probabilities(self, state, nodes_a, nodes_b):
        data_a, data_b = state
        embeddings_a = data_a.x[nodes_a]
        embeddings_b = data_b.x[nodes_b]
        concat_embedding_matrix = cartesian_product_matrix(embeddings_a, embeddings_b)

        return self.link_probability_net(concat_embedding_matrix)

    def assignment_step(self, state, a, b):
        data_a, data_b = state
        
        embeddings_a = data_a.x
        embeddings_b = data_b.x

        # Compute new embedding for assigned nodes
        frozen_embedding_a = self.link_encoder(torch.cat((embeddings_a[a], embeddings_b[b])))
        frozen_embedding_b = self.link_encoder(torch.cat((embeddings_b[b], embeddings_a[a])))

        # Overwrite original embedding
        new_embeddings_a = embeddings_a.clone() # Clone required for autograd
        new_embeddings_a = embeddings_b.clone()
        new_embeddings_a[a] = frozen_embedding_a
        new_embeddings_a[b] = frozen_embedding_b

        new_data_a = GraphData(x=new_embeddings_a, edge_index=data_a.edge_index, edge_attr=data_a.edge_attr)
        new_data_b = GraphData(x=new_embeddings_a, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr)

        return (new_data_a, new_data_b)

