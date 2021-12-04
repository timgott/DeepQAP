from torch.nn import ELU
from nn import Bidirectional, FullyConnected, NodeTransformer, QAPNet, edge_histogram_embeddings

def mp_transformer(hidden_channels, edge_embedding_size, node_embedding_size, depth):
    # Edge encoder. Concatenates weights in both direction
    # before passing to NN.
    edge_embedding_net = Bidirectional(
        FullyConnected(
            2, hidden_channels, edge_embedding_size,
            depth=depth, activation=ELU
        )
    )

    # summed edge embedding -> node embedding after initialization
    initial_node_embedding_net = FullyConnected(
        edge_embedding_size, hidden_channels, node_embedding_size,
        depth=depth, activation=ELU
    )

    # Message passing node embedding net
    message_passing_net = NodeTransformer(node_embedding_size, hidden_channels, edge_embedding_size)

    # concatenated node embeddings -> node embedding
    link_embedding_net = FullyConnected(
        node_embedding_size * 2, hidden_channels, node_embedding_size,
        depth=depth, activation=ELU
    )

    link_probability_net = FullyConnected(
        node_embedding_size * 2, hidden_channels, 1,
        depth=depth, activation=ELU
    )

    return QAPNet(
        edge_encoder=edge_embedding_net,
        initial_node_encoder=initial_node_embedding_net,
        message_passing_net=message_passing_net,
        link_encoder=link_embedding_net,
        link_probability_net=link_probability_net
    )

def link_prediction_only(embedding_size, hidden_channels, depth):
    histogram_encoder = lambda C: edge_histogram_embeddings(C, bins=embedding_size)
    identity_encoder = lambda x: x
    link_probability_net = FullyConnected(
        embedding_size * 2, hidden_channels, 1,
        depth=depth, activation=ELU
    )

    return QAPNet(
        edge_encoder=histogram_encoder,
        initial_node_encoder=identity_encoder,
        link_probability_net=link_probability_net,
        link_encoder=identity_encoder
    )