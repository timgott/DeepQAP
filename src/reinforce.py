import networkx
from networkx.drawing.nx_pylab import draw
from networkx.readwrite.graph6 import to_graph6_bytes
import numpy as np
import torch
from torch.nn import ELU, ModuleList
from nn import FullyConnected, NodeTransformer, cartesian_product_matrix, concat_bidirectional, sum_incoming_edges, dense_edge_features_to_sparse, edge_histogram_embeddings
from torch_geometric.data import Data as GraphData
from qap import GraphAssignmentProblem

class Categorical2D:
    def __init__(self, logits, shape=None) -> None:
        flat_logits = logits.reshape((-1,))
        self.shape = shape or logits.shape
        self.distribution = torch.distributions.Categorical(logits=flat_logits)

    def unravel(self, vector):
        return np.unravel_index(vector, self.shape)

    def ravel(self, pair):
        return torch.tensor(np.ravel_multi_index(pair, dims=self.shape))

    def sample(self):
        flat_result = self.distribution.sample()
        return self.unravel(flat_result)

    def log_prob(self, pair):
        return self.distribution.log_prob(self.ravel(pair))

    def entropy(self):
        return self.distribution.entropy()

class ReinforceAgent:
    def __init__(self, learning_rate=1e-3):
        hidden_channels = 64
        edge_embedding_size = 32
        node_embedding_size = 32
        
        self.hidden_channels = hidden_channels
        self.edge_embedding_size = edge_embedding_size
        self.node_embedding_size = node_embedding_size
        
        # Allow to generate histogram-like embeddings, that are more meaningful
        # than scalars when aggregated
        self.edge_embedding_net = FullyConnected(
            2, hidden_channels, edge_embedding_size,
            depth=4, activation=ELU
        )

        # Initial summed edge embedding -> node embedding network
        self.initial_node_embedding_net = FullyConnected(
            edge_embedding_size, hidden_channels, node_embedding_size,
            depth=3, activation=ELU
        )

        # Message passing node embedding net
        self.messaging_net = NodeTransformer(node_embedding_size, hidden_channels, edge_embedding_size)

        # Network that computes linked embeddings of assigned nodes
        # input: concatenated node embeddings
        self.link_freeze_net = FullyConnected(
            node_embedding_size * 2, hidden_channels, node_embedding_size,
            depth=3, activation=ELU
        )
        # output: node embedding

        # Network that computes logit probability that two nodes should be linked
        # asymmetric!
        self.link_probability_net = FullyConnected(
            node_embedding_size * 2, hidden_channels, 1,
            depth=3, activation=ELU
        )

        # List of all networks the agent uses for storing
        self.networks = ModuleList([
            #self.edge_embedding_net,
            self.initial_node_embedding_net,
            #self.messaging_net,
            self.link_probability_net,
            self.link_freeze_net,
        ])

        self.optimizer = torch.optim.Adam(self.networks.parameters(), lr=learning_rate)

    def compute_loss(self, value, policies, actions):
        return value * sum(p.log_prob(actions[i]) for i, p in enumerate(policies))

    def compute_neural_edge_embeddings(self, connectivity):
        # mirror edge weights to create swap-symmetrical tuples
        bidirectional_edge_weights = concat_bidirectional(connectivity.unsqueeze(2))
        # compute edge embedding vectors from weight values
        return self.edge_embedding_net(bidirectional_edge_weights)

    def transform_initial_graph(self, nx_graph):
        # networkx graph to connectivity matrix
        connectivity_matrix = torch.tensor(networkx.linalg.adjacency_matrix(nx_graph).todense()).float()
        # compute histogram
        edge_embeddings = edge_histogram_embeddings(connectivity_matrix, bins=self.edge_embedding_size)
        # aggregate edges and compute initial node embeddings
        aggregated = sum_incoming_edges(edge_embeddings)
        base_embeddings = self.initial_node_embedding_net(aggregated)
        # matrix format to adjacency and attribute list
        edge_index, edge_attr = dense_edge_features_to_sparse(connectivity_matrix, edge_embeddings)
        # Create data object required for torch geometric
        data = GraphData(x=base_embeddings, edge_index=edge_index, edge_attr=edge_attr)
        return data

    def compute_link_probabilities(self, embeddings_a, embeddings_b):
        n = embeddings_a.shape[0]

        concat_embedding_matrix = cartesian_product_matrix(embeddings_a, embeddings_b)

        # Ensure sizes
        assert concat_embedding_matrix.shape == (n,n,self.node_embedding_size*2), f"{concat_embedding_matrix.shape} vs {n},{n},{self.node_embedding_size*2}"

        return self.link_probability_net(concat_embedding_matrix)

    def solve_and_learn(self, qap: GraphAssignmentProblem):
        unassigned_a = list(qap.graph_source.nodes)
        unassigned_b = list(qap.graph_target.nodes)
        assignment = np.empty(qap.size)

        # Initial node and edge embeddings
        data_a = self.transform_initial_graph(qap.graph_source)
        data_b = self.transform_initial_graph(qap.graph_target)

        # Statistics
        entropies = []

        # Sum of log(policy(action)), part of loss
        log_probs = 0

        for _ in range(qap.size):
            # Message passing step
            #embeddings_a = self.messaging_net(data_a.x, data_a.edge_index, data_a.edge_attr)
            #embeddings_b = self.messaging_net(data_b.x, data_b.edge_index, data_b.edge_attr)
            embeddings_a = data_a.x
            embeddings_b = data_b.x

            probabilities = self.compute_link_probabilities(embeddings_a[unassigned_a], embeddings_b[unassigned_b])
            policy = Categorical2D(logits=probabilities)
            pair = policy.sample()

            # pair contains indices of unassigned_a, unassigned_b
            a = unassigned_a.pop(pair[0])
            b = unassigned_b.pop(pair[1])
            assignment[a] = b

            # Compute new embedding for assigned nodes
            frozen_embedding_a = self.link_freeze_net(torch.cat((embeddings_a[a], embeddings_b[b])))
            frozen_embedding_b = self.link_freeze_net(torch.cat((embeddings_b[b], embeddings_a[a])))

            # Add log prob
            log_probs += policy.log_prob(pair)

            # Overwrite original embedding
            data_a.x = data_a.x.clone() # Clone for autograd
            data_b.x = data_b.x.clone() # Clone for autograd
            data_a.x[a] = frozen_embedding_a
            data_b.x[b] = frozen_embedding_b

            # Entropy stats
            with torch.no_grad():
                entropies.append(policy.entropy())

        # Compute assignment value
        value = qap.compute_value(assignment)

        # Optimize
        loss = value * log_probs
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        # Training statistics
        gradient_magnitude = 0
        for param in self.networks.parameters():
            gradient_magnitude += torch.norm(param.grad)
        
        self.episode_stats = {
            "value": value,
            "entropy_average": np.mean(entropies),
            "episode_entropies": entropies,
            "gradient_magnitude": gradient_magnitude
        }

        return value, assignment

    def get_episode_stats(self):
        return self.episode_stats

    def save_checkpoint(self, path):
        torch.save({
            'policy_state_dict': self.networks.state_dict(),
            'policy_optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.networks.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])




