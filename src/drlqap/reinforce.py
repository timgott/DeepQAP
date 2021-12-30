import numpy as np
import torch
from drlqap import utils
from drlqap.qap import QAP
from drlqap.qapenv import QAPEnv
from drlqap.utils import IncrementalStats, Categorical2D

class ReinforceAgent:
    def __init__(self, policy_net, learning_rate=1e-3):
        self.policy_net = policy_net
        self.baseline = IncrementalStats()

        self.optimizer = torch.optim.Adam(
                self.policy_net.parameters(),
                lr=learning_rate
                )


    def get_policy(self, qap: QAP):
        # Compute probabilities for every pair p_ij = P(a_i -> b_j)
        probabilities = self.policy_net(qap)

        # Assert shape
        n = qap.size
        assert probabilities.shape == (n, n), \
                f"{probabilities.shape} != {n},{n}"

        # Create distribution over pairs
        policy = Categorical2D(logits=probabilities, shape=(n,n))

        return policy


    def run_episode(self, env: QAPEnv, learn=True):
        # Statistics
        entropies = []

        # log(policy(action)), part of loss
        log_probs = []

        # Rewards
        rewards = []

        while not env.done:
            qap = env.get_state()
            policy = self.get_policy(qap)
            pair = policy.sample()

            # Add log prob
            log_probs.append(policy.log_prob(pair))

            # Entropy stats
            with torch.no_grad():
                entropies.append(policy.entropy())

            # Env step
            reward = env.step(pair)
            rewards.append(reward)

        if learn:
            for x in rewards:
                self.baseline.add(x)

            baselined_rewards = np.array(rewards) - self.baseline.mean()

            # Compute accumulated rewards
            returns = utils.reverse_cumsum(baselined_rewards)

            # Optimize
            loss = -torch.sum(torch.tensor(returns) * torch.stack(log_probs))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        # Training statistics
        gradient_magnitude = 0
        for param in self.policy_net.parameters():
            if param.grad is not None:
                gradient_magnitude += torch.norm(param.grad).item()

        self.episode_stats = {
                "value": env.reward_sum,
                "entropy_average": np.mean(entropies),
                "episode_entropies": np.array(entropies),
                "gradient_magnitude": gradient_magnitude
                }


    def solve_and_learn(self, qap: QAP):
        env = QAPEnv(qap)
        self.run_episode(env, learn=True)
        assert env.done
        return env.reward_sum, env.assignment

    def solve(self, qap: QAP):
        env = QAPEnv(qap)
        self.run_episode(env, learn=False)
        assert env.done
        return env.reward_sum, env.assignment

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
        self.optimizer.load_state_dict(
                checkpoint['policy_optimizer_state_dict'])
        self.baseline.load_state_dict(checkpoint['baseline_state_dict'])
