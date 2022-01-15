import torch
from torch.nn import Sequential
from torch_geometric.nn import TransformerConv, to_hetero, Linear, GATv2Conv
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import copy
from drlqap.qap import QAP
import logging

def concat_bidirectional(dense_edge_features: torch.Tensor) -> torch.Tensor:
    """
    Concatenate the embedding of each edge with the embedding of its reverse edge.

    input: (n,n,d) n: number of nodes

    output: (n,n,d*2)
    """
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

    if step == 0:
        return torch.zeros(connectivity.shape + (bins,))

    range = torch.arange(lower, upper - step/2, step=step).reshape(1,1,-1)
    connectivity3d = connectivity.unsqueeze(2)
    return torch.where((connectivity3d >= range) & (connectivity3d < range + step), 1.0, 0.0)

def bidirectional_edge_histogram_embeddings(connectivity, bins):
    incoming = edge_histogram_embeddings(connectivity, bins)
    outgoing = edge_histogram_embeddings(connectivity.T, bins)
    return torch.cat((incoming, outgoing), dim=1)


# Create matrix where entry ij is cat(a_i,b_j)
def cartesian_product_matrix(a, b):
    assert(a.shape == b.shape)
    
    n = a.shape[0]

    # Repeat a along dimension 1
    a_rows = a.unsqueeze(1).expand(-1, n, -1)

    # Repeat b along dimension 0
    b_columns = b.unsqueeze(0).expand(n, -1, -1)

    return torch.cat((a_rows, b_columns), dim=2)

# Create matrix where entry ij is <a_i . b_j>
def dot_product_matrix(a, b):
    return torch.matmul(a, b.transpose(0, 1))

class GAT(torch.nn.Module):
    def __init__(self, node_channels, hidden_channels, edge_embedding_size):
        super().__init__()
        self.conv1 = GATv2Conv((-1, -1), hidden_channels, edge_dim=-1, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATv2Conv((-1, -1), node_channels, edge_dim=-1, add_self_loops=False)
        self.lin2 = Linear(-1, node_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr) + self.lin2(x)
        return x


def FullyConnected(input_channels, hidden_channels, output_channels, depth, activation):
    return FullyConnectedShaped(
        [input_channels] + [hidden_channels] * depth + [output_channels],
        activation
    )


def FullyConnectedShaped(shape, activation):
    layers = Sequential()
    for i, (inputs, outputs) in enumerate(zip(shape, shape[1:])):
        layers.add_module(f"linear[{i}]({inputs}->{outputs})", Linear(inputs, outputs))
        layers.add_module(f"activation[{i}]", activation())
    return layers


def FullyConnectedLinearOut(in_channels, hidden_channels, out_channels, depth, activation):
    return Sequential(
        FullyConnected(in_channels, hidden_channels, hidden_channels, depth-1, activation),
        Linear(hidden_channels, out_channels)
    )


class BidirectionalDense(torch.nn.Module):
    def __init__(self, inner) -> None:
        super().__init__()
        self.inner = inner
    
    def forward(self, matrix):
        # mirror edge weights to create swap-symmetrical tuples
        bidirectional_matrix = concat_bidirectional(matrix.unsqueeze(2))
        # compute edge embedding vectors from weight values
        return self.inner(bidirectional_matrix)

class ReinforceNet(torch.nn.Module):
    def __init__(self, 
        edge_encoder,
        initial_node_encoder,
        link_probability_net,
        message_passing_net,
    ) -> None:
        super().__init__()

        # Allow to generate histogram-like embeddings, that are more meaningful
        # than scalars when aggregated
        self.edge_encoder = edge_encoder

        # Initial summed edge embedding -> node embedding network
        self.initial_node_encoder = initial_node_encoder

        # Create heterogenous message passing net
        if message_passing_net is not None:
            edge_types = [
                ('a', 'A', 'a'),
                ('b', 'B', 'b'),
                ('a', 'L', 'b'),
                ('b', 'L', 'a'),
            ]
            node_types = ['a', 'b']
            metadata = (node_types, edge_types)
            self.message_passing_net = to_hetero(message_passing_net, metadata)
        else:
            self.message_passing_net = None

        # Network that computes logit probability that two nodes should be linked
        self.link_probability_net = link_probability_net

    def aggregate_edges(self, edge_features):
        # aggregate edges and compute initial node embeddings
        aggregated = sum_incoming_edges(edge_features)
        node_features = self.initial_node_encoder(aggregated)

        return node_features

    def initial_transformation(self, qap):
        def matrix_to_edges(connectivity, edge_features):
            edge_index, edge_attr = dense_edge_features_to_sparse(
                connectivity, edge_features
            )
            return dict(
                    edge_index=edge_index,
                    edge_attr=edge_attr
                    )

        # compute edge encoding and aggregate to nodes
        edge_vectors_a = self.edge_encoder(qap.A)
        edge_vectors_b = self.edge_encoder(qap.B)
        node_vectors_a = self.aggregate_edges(edge_vectors_a)
        node_vectors_b = self.aggregate_edges(edge_vectors_b)

        L = qap.linear_costs
        Lrev = L.T

        # create heterogenous pyg graph
        data = HeteroData({
            ('a'): dict(x=node_vectors_a),
            ('b'): dict(x=node_vectors_b),
            ('a', 'A', 'a'): matrix_to_edges(qap.A, edge_vectors_a),
            ('b', 'B', 'b'): matrix_to_edges(qap.B, edge_vectors_b),
            ('a', 'L', 'b'): matrix_to_edges(L, L),
            ('b', 'L', 'a'): matrix_to_edges(Lrev, Lrev),
        }, aggr='sum')

        return data

    def compute_link_probabilities(self, embeddings_a, embeddings_b):
        if self.link_probability_net:
            concat_embedding_matrix = cartesian_product_matrix(embeddings_a, embeddings_b)
            probs = self.link_probability_net(concat_embedding_matrix)
            n, m = embeddings_a.size(0), embeddings_b.size(0)
            return probs.reshape((n, m))
        else:
            return dot_product_matrix(embeddings_a, embeddings_b)

    def forward(self, qap: QAP):
        # trivial QAPs (can not stop here for DQN, value is not always 1!)
        # if qap.size == 1:
        #     return torch.tensor([[1.]])

        hdata = self.initial_transformation(qap)
        if self.message_passing_net:
            node_dict = self.message_passing_net(hdata.x_dict, hdata.edge_index_dict, hdata.edge_attr_dict)
        else:
            node_dict = hdata.x_dict

        return self.compute_link_probabilities(node_dict['a'], node_dict['b'])
