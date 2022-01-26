import torch
from torch.nn import Sequential, LeakyReLU, ModuleList, LayerNorm
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

def aggregate_incoming_edges(dense_edge_features: torch.Tensor):
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
    assert(len(a.shape) == 2)

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


def FullyConnected(input_channels, hidden_channels, output_channels, depth, activation, layer_norm=False):
    return FullyConnectedShaped(
        [input_channels] + [hidden_channels] * depth + [output_channels],
        activation,
        layer_norm=layer_norm
    )


def FullyConnectedShaped(shape, activation, layer_norm=False):
    layers = Sequential()
    for i, (inputs, outputs) in enumerate(zip(shape, shape[1:])):
        layers.add_module(f"linear[{i}]({inputs}->{outputs})", Linear(inputs, outputs))
        if layer_norm:
            layers.add_module(f"layernorm[{i}]", LayerNorm(outputs))
        layers.add_module(f"activation[{i}]", activation())
    return layers


def FullyConnectedLinearOut(in_channels, hidden_channels, out_channels, depth, activation, layer_norm=False):
    return Sequential(
        FullyConnected(in_channels, hidden_channels, hidden_channels, depth-1, activation, layer_norm=layer_norm),
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

class PygReinforceNet(torch.nn.Module):
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
        aggregated = aggregate_incoming_edges(edge_features)
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


class ConvLayer(torch.nn.Module):
    """
    Transforms incoming edges with `edge_encoder`, 
    aggregates them together with their source node 
    and transforms the aggregated value with `tranformation`.
    """
    def __init__(self, edge_encoder, transformation, layer_norm=False) -> None:
        super().__init__()

        self.edge_encoder = edge_encoder
        self.transformation = transformation
        self.layer_norm = layer_norm

    def forward(self, edges, neighbors):
        assert(len(edges.shape) == 3)
        assert(neighbors is None or len(neighbors.shape) == 2)
        assert(neighbors is None or edges.shape[1] == neighbors.shape[0])

        # Edge encodings
        if neighbors is not None:
            # e_ij = nn(E_ij || N_j)
            n_j = neighbors.unsqueeze(0).expand((edges.shape[0], -1, -1))
            e = self.edge_encoder(torch.cat((edges, n_j), dim=-1))
        else:
            # e_ij = nn(E_ij)
            e = self.edge_encoder(edges)

        # aggregate and transform
        # n_ij = nn(sum_j(e_ij))
        x = aggregate_outgoing_edges(e)

        assert(len(x.shape) == 2)
        if self.layer_norm:
            normalized_shape = x.shape
            x = F.layer_norm(x, normalized_shape)

        return self.transformation(x)


class QapConvLayer(torch.nn.Module):
    """
    Applies ConvLayers for linear and quadratic edges.
    Then transforms the sum (q + l + x) again, where q and l are 
    the outputs of the ConvLayers and x is the old node embedding.
    """
    def __init__(self, edge_width, embedding_width, depth, conv_layer_norm=True) -> None:
        super().__init__()

        # Size of embedding
        w = embedding_width

        self.q_conv = ConvLayer(
            edge_encoder=FullyConnected(edge_width, w, w, depth, activation=LeakyReLU, layer_norm=False),
            transformation=FullyConnected(w, w, w, 0, activation=LeakyReLU, layer_norm=False),
            layer_norm=conv_layer_norm
        )

        self.l_conv = ConvLayer(
            edge_encoder=FullyConnected(edge_width, w, w, depth, activation=LeakyReLU, layer_norm=False),
            transformation=FullyConnected(w, w, w, 0, activation=LeakyReLU, layer_norm=False),
            layer_norm=conv_layer_norm
        )

        self.combined_transformation = FullyConnected(w, w, w, depth, activation=LeakyReLU, layer_norm=False)


    def forward(self, a_to_a, a_to_b, a, b) -> torch.Tensor:
        # Quadratic encoding
        q = self.q_conv(a_to_a, a)

        # Linear encoding
        l = self.l_conv(a_to_b, b)

        # Combined value
        if a is not None:
            c = q + l + a
        else:
            c = q + l

        return self.combined_transformation(c)



class DenseQAPNet(torch.nn.Module):
    """
    Applies multiple QapConvLayers to a QAP.
    """

    def __init__(self, embedding_width, encoder_depth, conv_depth, use_layer_norm=True, conv_layer_norm=True) -> None:
        super().__init__()

        w = embedding_width
        self.a_layers = ModuleList(
            [QapConvLayer(1, w, encoder_depth, conv_layer_norm=conv_layer_norm)] +
            [QapConvLayer(1 + w, w, encoder_depth, conv_layer_norm=conv_layer_norm) for _ in range(conv_depth - 1)],
        )
        self.b_layers = ModuleList(
            [QapConvLayer(1, w, encoder_depth, conv_layer_norm=conv_layer_norm)] +
            [QapConvLayer(1 + w, w, encoder_depth, conv_layer_norm=conv_layer_norm) for _ in range(conv_depth - 1)],
        )
        if use_layer_norm:
            self.pair_norm = LayerNorm(w*2)
        else:
            self.pair_norm = lambda x: x
        self.link_probability_net = FullyConnectedLinearOut(w * 2, w, 1, 3, activation=LeakyReLU, layer_norm=False)

    def compute_link_values(self, embeddings_a, embeddings_b):
        if self.link_probability_net:
            pairs = cartesian_product_matrix(embeddings_a, embeddings_b)
            pairs = self.pair_norm(pairs)
            probs = self.link_probability_net(pairs)
            n, m = embeddings_a.size(0), embeddings_b.size(0)
            return probs.reshape((n, m))
        else:
            return dot_product_matrix(embeddings_a, embeddings_b)

    def forward(self, qap: QAP) -> torch.Tensor:
        # TODO: support directed A and B by using concat_bidirectional
        A = qap.A.unsqueeze(2) # Convert weights into 1-dim vectors
        B = qap.B.unsqueeze(2)
        L = qap.linear_costs.unsqueeze(2)

        # Node embeddings
        a = None
        b = None

        # Apply each conv layer
        for layer_a, layer_b in zip(self.a_layers, self.b_layers):
            new_a = layer_a(A, L, a, b)
            new_b = layer_b(B, L.transpose(0, 1), b, a)

            a = new_a
            b = new_b

        # Store embeddings for debugging
        self.embeddings_a = a
        self.embeddings_b = b

        return self.compute_link_values(a, b)
