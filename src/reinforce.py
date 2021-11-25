from networkx.drawing.nx_pylab import draw
import numpy as np
import torch
from torch.nn import Linear, ReLU
from torch_geometric.nn import Sequential, GCNConv
from torch_geometric.utils import from_networkx
from qap import GraphAssignmentProblem, AssignmentGraph
from visualisation import draw_assignment_graph
import matplotlib.pyplot as plt

class Categorical2D:
    def __init__(self, logits) -> None:
        flat_logits = logits.reshape((-1,))
        self.shape = logits.shape
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
        hidden_channels = 16
        out_channels = 8
        self.network = Sequential('x, edge_index, edge_attr', [
            (GCNConv(1, hidden_channels), 'x, edge_index, edge_attr -> x'),
            ReLU(),
            (GCNConv(hidden_channels, out_channels), 'x, edge_index, edge_attr -> x'),
            ReLU()
        ])
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

    def compute_policy(self, data, unassigned_source_nodes, unassigned_target_nodes):
        embeddings = self.network(data.x, data.edge_index, data.edge_attr)
        source_embeddings = embeddings[unassigned_source_nodes]
        target_embeddings = embeddings[unassigned_target_nodes]
        probabilities = torch.matmul(source_embeddings, target_embeddings.T)
        policy = Categorical2D(logits=probabilities)
        return policy

    def compute_loss(self, value, policies, actions):
        return value * sum(p.log_prob(actions[i]) for i, p in enumerate(policies))

    def solve_and_learn(self, qap: GraphAssignmentProblem):
        nx_graph = AssignmentGraph(qap.graph_source, qap.graph_target, [])

        unassigned_a = list(qap.graph_source.nodes)
        unassigned_b = list(qap.graph_target.nodes)
        assignment = [None] * qap.size

        policies = []
        actions = []

        for i in range(qap.size):
            data = from_networkx(nx_graph.graph, ["side"], ["weight"])

            policy = self.compute_policy(data, unassigned_a, unassigned_b)
            pair = policy.sample()

            # Store for learning (recomputing policy might be expensive)
            policies.append(policy)
            actions.append(pair)

            x = unassigned_a.pop(pair[0])
            y = unassigned_b.pop(pair[1])
            assignment[x] = y

            nx_graph.add_assignment(x, y)

        value = qap.compute_value(assignment)

        self.optimizer.zero_grad()
        loss = self.compute_loss(value, policies, actions)

        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            average_entropy = np.mean([p.entropy() for p in policies])
        
            self.episode_stats = {
                "value": value,
                "entropy_average": average_entropy
            }

        return value, assignment
    
    def get_episode_stats(self):
        return self.episode_stats

    def save_checkpoint(self, path):
        torch.save({
            'policy_state_dict': self.network.state_dict(),
            'policy_optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])




