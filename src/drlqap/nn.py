import torch
from torch.nn import LeakyReLU, ModuleList, LayerNorm
import torch.nn.functional as F
from drlqap.qap import QAP
import drlqap.nnutils as nnutils
from drlqap.nnutils import FullyConnected, FullyConnectedLinearOut

class SimpleQapNet(torch.nn.Module):
    def __init__(self, edge_encoder, probability_net):
        super().__init__()
        self.edge_encoder = edge_encoder
        self.probability_net = probability_net

    def forward(self, qap: QAP):
        self.embeddings_a = nnutils.aggregate_incoming_edges(self.edge_encoder(qap.A))
        self.embeddings_b = nnutils.aggregate_incoming_edges(self.edge_encoder(qap.B))
        pair_matrix = nnutils.cartesian_product_matrix(self.embeddings_a, self.embeddings_b)
        pair_matrix = torch.cat((pair_matrix, qap.linear_costs.unsqueeze(2)), dim=-1)
        result = self.probability_net(pair_matrix).reshape((qap.size, qap.size))
        return result

class KeepMeanNorm(torch.nn.Module):
    def forward(self, x):
        mean = torch.mean(x, dim=0)
        centered_x = x - mean
        stdev = (torch.square(centered_x).sum(dim=1).mean() + 1e-6).sqrt()
        return centered_x / stdev + mean

class TransformedMeanNorm(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mean_linear = torch.nn.Linear(channels, channels)

    def forward(self, x):
        mean = torch.mean(x, dim=0)
        centered_x = x - mean
        stdev = (torch.square(centered_x).sum(dim=1).mean() + 1e-6).sqrt()

        transformed_mean = F.relu(self.mean_linear(mean))
        return centered_x / stdev + transformed_mean

class MeanSeparationLayer(torch.nn.Module):
    def __init__(self, channels, residual_scale=1):
        super().__init__()
        self.mean_linear = torch.nn.Linear(channels, channels)
        self.residual_linear = torch.nn.Linear(channels, channels)
        self.residual_linear.weight.data.mul_(residual_scale)

    def forward(self, x):
        mean = torch.mean(x, dim=0)
        residual = x - mean
        transformed_mean = F.leaky_relu(self.mean_linear(mean))
        transformed_residual = F.leaky_relu(self.residual_linear(residual))
        return transformed_residual + transformed_mean

class FullLayerNorm(torch.nn.Module):
    def forward(self, x):
        normalized_shape = x.shape
        return F.layer_norm(x, normalized_shape)

class ConvLayer(torch.nn.Module):
    """
    Transforms incoming edges with `edge_encoder`, 
    aggregates them together with their source node 
    and transforms the aggregated value with `tranformation`.
    """
    def __init__(self, edge_encoder, transformation, norm=None, aggregation='sum') -> None:
        super().__init__()

        self.edge_encoder = edge_encoder
        self.transformation = transformation
        self.aggregation = aggregation
        self.norm = norm

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
        if self.aggregation == 'sum':
            x = torch.sum(e, dim=1)
        elif self.aggregation == 'mean':
            x = torch.mean(e, dim=1)
        elif self.aggregation == 'min':
            x = torch.min(e, dim=1)[0]
        elif self.aggregation == 'max':
            x = torch.max(e, dim=1)[0]
        else:
            raise ValueError(f"Invalid aggregation {self.aggregation}")

        assert(len(x.shape) == 2)

        if self.norm:
            x = self.norm(x)

        return self.transformation(x)


class QapConvLayer(torch.nn.Module):
    """
    Applies ConvLayers for linear and quadratic edges.
    Then transforms the sum (q + l + x) again, where q and l are 
    the outputs of the ConvLayers and x is the old node embedding.
    """
    def __init__(self, edge_width, embedding_width, depth, conv_norm=None, q_aggr='sum', l_aggr='sum', combined_transform=True) -> None:
        super().__init__()

        # Size of embedding
        w = embedding_width

        def create_norm(norm):
            if norm == 'batch_norm':
                return torch.nn.BatchNorm1d(num_features=w, track_running_stats=False, affine=False)
            elif norm == 'layer_norm':
                return FullLayerNorm()
            elif norm == 'keep_mean':
                return KeepMeanNorm()
            elif norm == 'transformed_mean':
                return TransformedMeanNorm(w)
            elif norm == 'mean_separation':
                return MeanSeparationLayer(w)
            elif norm == 'mean_separation_100x':
                return MeanSeparationLayer(w, residual_scale=100)
            elif norm:
                raise ValueError(f"Invalid norm {self.norm_type}")
            else:
                return None

        self.q_conv = ConvLayer(
            edge_encoder=FullyConnected(edge_width, w, w, depth, activation=LeakyReLU, layer_norm=False),
            transformation=FullyConnected(w, w, w, 0, activation=LeakyReLU, layer_norm=False),
            norm=create_norm(conv_norm),
            aggregation=q_aggr
        )

        self.l_conv = ConvLayer(
            edge_encoder=FullyConnected(edge_width, w, w, depth, activation=LeakyReLU, layer_norm=False),
            transformation=FullyConnected(w, w, w, 0, activation=LeakyReLU, layer_norm=False),
            norm=create_norm(conv_norm),
            aggregation=l_aggr
        )

        if combined_transform:
            self.combined_transformation = FullyConnected(w, w, w, depth, activation=LeakyReLU, layer_norm=False)
        else:
            self.combined_transformation = None


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

        if self.combined_transformation:
            c = self.combined_transformation(c)

        return c


class EdgeEncoderLayer(torch.nn.Module):
    def __init__(self, bidirectional, channels, depth):
        super().__init__()
        input_channels = 2 if bidirectional else 1
        self.fc = nnutils.FullyConnected(
            input_channels=input_channels,
            hidden_channels=channels,
            output_channels=channels,
            depth=depth,
            activation=LeakyReLU,
        )
        self.bidirectional = bidirectional

    def forward(self, edge_weights):
        edge_vectors = edge_weights.unsqueeze(2)
        if self.bidirectional:
            # mirror edge weights to create swap-symmetrical tuples
            edge_vectors = nnutils.concat_bidirectional(edge_vectors)
        return self.fc(edge_vectors)


class DenseQAPNet(torch.nn.Module):
    """
    Applies multiple QapConvLayers to a QAP.
    """

    def __init__(self, embedding_width, encoder_depth, conv_depth, use_layer_norm, conv_norm, q_aggr, l_aggr, combined_transform, random_start, use_edge_encoder, bidirectional) -> None:
        super().__init__()

        w = embedding_width
        edge_width = w if use_edge_encoder else 1
        initial_w = w if random_start else 0

        conv_kwargs = dict(conv_norm=conv_norm, q_aggr=q_aggr, l_aggr=l_aggr, combined_transform=combined_transform)
        self.a_layers = ModuleList(
            [QapConvLayer(edge_width + initial_w, w, encoder_depth, **conv_kwargs)] +
            [QapConvLayer(edge_width + w, w, encoder_depth, **conv_kwargs) for _ in range(conv_depth - 1)],
        )
        self.b_layers = ModuleList(
            [QapConvLayer(edge_width + initial_w, w, encoder_depth, **conv_kwargs)] +
            [QapConvLayer(edge_width + w, w, encoder_depth, **conv_kwargs) for _ in range(conv_depth - 1)],
        )
        if use_layer_norm:
            self.pair_norm = LayerNorm(w*2)
        else:
            self.pair_norm = lambda x: x
        self.link_probability_net = FullyConnectedLinearOut(w * 2, w, 1, 3, activation=LeakyReLU, layer_norm=False)
        self.random_start = random_start
        self.embedding_width = embedding_width

        assert(not (bidirectional and not use_edge_encoder))
        self.use_edge_encoder = use_edge_encoder
        if use_edge_encoder:
            self.a_edge_encoder = EdgeEncoderLayer(bidirectional, w, 2)
            self.b_edge_encoder = EdgeEncoderLayer(bidirectional, w, 2)
            self.l_edge_encoder = EdgeEncoderLayer(bidirectional, w, 2)

    def compute_link_values(self, embeddings_a, embeddings_b):
        if self.link_probability_net:
            pairs = nnutils.cartesian_product_matrix(embeddings_a, embeddings_b)
            pairs = self.pair_norm(pairs)
            probs = self.link_probability_net(pairs)
            n, m = embeddings_a.size(0), embeddings_b.size(0)
            return probs.reshape((n, m))
        else:
            return nnutils.dot_product_matrix(embeddings_a, embeddings_b)

    def forward(self, qap: QAP) -> torch.Tensor:
        if self.use_edge_encoder:
            A = self.a_edge_encoder(qap.A)
            B = self.b_edge_encoder(qap.B)
            L = self.l_edge_encoder(qap.linear_costs)
        else:
            A = qap.A.unsqueeze(2) # Convert weights into 1-dim vectors
            B = qap.B.unsqueeze(2)
            L = qap.linear_costs.unsqueeze(2)

        # Node embeddings
        if self.random_start:
            a = torch.rand(qap.size, self.embedding_width)
            b = torch.rand(qap.size, self.embedding_width)
        else:
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
