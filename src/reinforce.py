import numpy as np
import torch
from nn import QAPNet
from qap import GraphAssignmentProblem
import utils

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
    def __init__(self, policy_net: QAPNet, learning_rate=1e-3):
        self.policy_net = policy_net
        self.baseline = utils.IncrementalStats()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def compute_loss(self, value, policies, actions):
        return value * sum(p.log_prob(actions[i]) for i, p in enumerate(policies))

    def solve_and_learn(self, qap: GraphAssignmentProblem, learn=True):
        unassigned_a = list(qap.graph_source.nodes)
        unassigned_b = list(qap.graph_target.nodes)
        assignment = np.empty(qap.size)

        state = self.policy_net.initial_step(qap)

        # Statistics
        entropies = []

        # Sum of log(policy(action)), part of loss
        log_probs = 0

        for i in range(qap.size):
            # Compute probabilities for every pair p_ij = P(a_i -> b_j)
            probabilities = self.policy_net.compute_link_probabilities(state, unassigned_a, unassigned_b)

            # Assert shape
            n = qap.size - i
            assert probabilities.shape == (n, n, 1), f"{probabilities.shape} != {n},{n},1"

            # Sample assignment
            policy = Categorical2D(logits=probabilities)
            pair = policy.sample()

            # pair contains indices of unassigned_a, unassigned_b
            a = unassigned_a.pop(pair[0])
            b = unassigned_b.pop(pair[1])
            assignment[a] = b

            # Add log prob
            log_probs += policy.log_prob(pair)
            
            # Entropy stats
            with torch.no_grad():
                entropies.append(policy.entropy())

        # Compute assignment value
        value = qap.compute_value(assignment)
        
        if learn:
            self.baseline.add(value)
            baselined_value = value - self.baseline.mean()

            # Optimize
            loss = baselined_value * log_probs
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        # Training statistics
        gradient_magnitude = 0
        for param in self.policy_net.parameters():
            if param.grad is not None:
                gradient_magnitude += torch.norm(param.grad).item()
        
        self.episode_stats = {
            "value": value,
            "entropy_average": np.mean(entropies),
            "episode_entropies": np.array(entropies),
            "gradient_magnitude": gradient_magnitude
        }

        return value, assignment

    def solve(self, qap):
        return self.solve_and_learn(qap, learn=False)

    def get_episode_stats(self):
        return self.episode_stats

    def save_checkpoint(self, path):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'policy_optimizer_state_dict': self.optimizer.state_dict(),
            'baseline_state_dict': self.baseline.state_dict()
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.baseline.load_state_dict(checkpoint['baseline_state_dict'])




