import torch
from torch.nn import ELU, Sequential, Linear
from drlqap import nn

identity_encoder = lambda x: x

def simple_link_prediction_undirected(embedding_size, hidden_channels, depth):
    histogram_encoder = lambda C: nn.edge_histogram_embeddings(C, bins=embedding_size)
    link_probability_net = nn.FullyConnectedLinearOut(
        embedding_size * 2 + 1, hidden_channels, 1,
        depth=depth, activation=ELU
    )

    return nn.SimpleQapNet(
        edge_encoder=histogram_encoder,
        probability_net=link_probability_net,
    )


def dense(channels, encoder_depth, conv_depth=2, layer_norm=False, conv_norm='layer_norm', q_aggr='sum', l_aggr='sum', combined_transform=True, random_start=False, use_edge_encoder=False, bidirectional=False):
    return nn.DenseQAPNet(
        embedding_width=channels, 
        encoder_depth=encoder_depth,
        conv_depth = conv_depth,
        use_layer_norm = layer_norm,
        conv_norm = conv_norm,
        q_aggr = q_aggr,
        l_aggr = l_aggr,
        combined_transform = combined_transform,
        random_start = random_start,
        use_edge_encoder = use_edge_encoder,
        bidirectional = bidirectional
    )

