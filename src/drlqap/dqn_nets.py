import torch
from torch.nn import ELU, Sequential, Linear
from drlqap import nn

identity_encoder = lambda x: x

def mp_gat(hidden_channels, edge_embedding_size, node_embedding_size, depth):
    # Edge encoder. Concatenates weights in both direction
    # before passing to NN.
    edge_embedding_net = nn.BidirectionalDense(
        nn.FullyConnected(
            2, hidden_channels, edge_embedding_size,
            depth=depth, activation=ELU
        )
    )

    # summed edge embedding -> node embedding after initialization
    initial_node_embedding_net = nn.FullyConnected(
        edge_embedding_size, hidden_channels, node_embedding_size,
        depth=depth, activation=ELU
    )

    # Message passing node embedding net
    message_passing_net = nn.GAT(node_embedding_size, hidden_channels, edge_embedding_size)

    link_probability_net = nn.FullyConnectedLinearOut(
        node_embedding_size * 2, hidden_channels, 1,
        depth=depth, activation=ELU
    )

    return nn.PygReinforceNet(
        edge_encoder=edge_embedding_net,
        initial_node_encoder=initial_node_embedding_net,
        message_passing_net=message_passing_net,
        link_probability_net=link_probability_net
    )


def mp_gat_no_edge_nn(hidden_channels, edge_embedding_size, node_embedding_size, depth):
    # Edge encoder. Concatenates weights in both direction
    # before passing to NN.
    edge_embedding_net = nn.BidirectionalDense(identity_encoder)

    # summed edge embedding -> node embedding after initialization
    initial_node_embedding_net = lambda e: torch.zeros(e.size(0), node_embedding_size)

    # Message passing node embedding net
    message_passing_net = nn.GAT(node_embedding_size, hidden_channels, edge_embedding_size)

    link_probability_net = nn.FullyConnectedLinearOut(
        node_embedding_size * 2, hidden_channels, 1,
        depth=depth, activation=ELU
    )

    return nn.PygReinforceNet(
        edge_encoder=edge_embedding_net,
        initial_node_encoder=initial_node_embedding_net,
        message_passing_net=message_passing_net,
        link_probability_net=link_probability_net
    )


def mp_gat_no_edge_nn_undirected(hidden_channels, edge_embedding_size, node_embedding_size, depth):
    # Edge encoder. Concatenates weights in both direction
    # before passing to NN.
    edge_embedding_net = identity_encoder

    # summed edge embedding -> node embedding after initialization
    initial_node_embedding_net = lambda e: torch.zeros(e.size(0), node_embedding_size)

    # Message passing node embedding net
    message_passing_net = nn.GAT(node_embedding_size, hidden_channels, edge_embedding_size)

    link_probability_net = nn.FullyConnectedLinearOut(
        node_embedding_size * 2, hidden_channels, 1,
        depth=depth, activation=ELU
    )

    return nn.PygReinforceNet(
        edge_encoder=edge_embedding_net,
        initial_node_encoder=initial_node_embedding_net,
        message_passing_net=message_passing_net,
        link_probability_net=link_probability_net
    )


def mp_histogram_gat(hidden_channels, embedding_size, depth):
    # Edge encoder.
    histogram_encoder = lambda C: nn.edge_histogram_embeddings(C, bins=embedding_size)

    # Message passing node embedding net
    message_passing_net = nn.GAT(embedding_size, hidden_channels, embedding_size)

    link_probability_net = nn.FullyConnectedLinearOut(
        embedding_size * 2, hidden_channels, 1,
        depth=depth, activation=ELU
    )

    return nn.PygReinforceNet(
        edge_encoder=histogram_encoder,
        initial_node_encoder=identity_encoder,
        message_passing_net=message_passing_net,
        link_probability_net=link_probability_net
    )


def simple_link_prediction_undirected(embedding_size, hidden_channels, depth):
    histogram_encoder = lambda C: nn.edge_histogram_embeddings(C, bins=embedding_size)
    link_probability_net = nn.FullyConnectedLinearOut(
        embedding_size * 2, hidden_channels, 1,
        depth=depth, activation=ELU
    )

    return nn.PygReinforceNet(
        edge_encoder=histogram_encoder,
        initial_node_encoder=identity_encoder,
        link_probability_net=link_probability_net,
        message_passing_net=None
    )


def mp_gat_no_lp(hidden_channels, edge_embedding_size, node_embedding_size, depth):
    # Edge encoder. Concatenates weights in both direction
    # before passing to NN.
    edge_embedding_net = nn.BidirectionalDense(
        nn.FullyConnected(
            2, hidden_channels, edge_embedding_size,
            depth=depth, activation=ELU
        )
    )

    # summed edge embedding -> node embedding after initialization
    initial_node_embedding_net = nn.FullyConnected(
        edge_embedding_size, hidden_channels, node_embedding_size,
        depth=depth, activation=ELU
    )

    # Message passing node embedding net
    message_passing_net = nn.GAT(node_embedding_size, hidden_channels, edge_embedding_size)

    return nn.PygReinforceNet(
        edge_encoder=edge_embedding_net,
        initial_node_encoder=initial_node_embedding_net,
        message_passing_net=message_passing_net,
        link_probability_net=None
    )


def dense(channels, encoder_depth, conv_depth=2, layer_norm=False, conv_layer_norm=True, q_aggr='sum', l_aggr='sum', combined_transform=True):
    return nn.DenseQAPNet(
        channels, encoder_depth, conv_depth, layer_norm, conv_layer_norm, q_aggr, l_aggr, combined_transform
    )

