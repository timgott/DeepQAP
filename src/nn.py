import torch
from torch.nn.modules.container import ModuleList
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

def aggregate_incoming_edges(dense_edge_features: torch.Tensor):
    return torch.mean(dense_edge_features, dim=0)

def aggregate_outgoing_edges(dense_edge_features: torch.Tensor):
    return torch.mean(dense_edge_features, dim=1)

def dense_edge_features_to_sparse(connectivity_matrix, feature_tensor):
    indices = connectivity_matrix.nonzero(as_tuple=True)
    edge_features = feature_tensor[indices]
    indices_tensor = torch.stack(indices)
    return indices_tensor, edge_features

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
            x = F.relu(x)
            intermediates.append(x)
        
        # concatenate layer output for every node
        skip_xs = torch.cat(intermediates, dim=-1)
        x = self.skip_linear(skip_xs)
        x = F.relu(x)

        return x