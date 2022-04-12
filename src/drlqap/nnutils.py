import torch
from torch.nn import Sequential, LayerNorm
from torch_geometric.nn import Linear

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

    range = torch.arange(lower, upper - step/2, step=step)
    range[0] -= 1
    range.reshape(1,1,-1)
    connectivity3d = connectivity.unsqueeze(2)
    bin = (connectivity3d > range).sum(dim=-1) - 1
    return F.one_hot(bin, num_classes=bins)

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

