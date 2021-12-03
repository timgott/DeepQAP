import torch
from torch.nn.modules.container import ModuleList, Sequential
import torch_geometric
from torch_geometric.nn import TransformerConv, JumpingKnowledge
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