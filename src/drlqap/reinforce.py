import numpy as np
import torch
from drlqap import utils
from drlqap.qap import QAP
from drlqap.qapenv import QAPReductionEnv
from drlqap.utils import IncrementalStats

class ReinforceAgent:
    def __init__(self, env_class, policy_net, policy_sampler, learning_rate=1e-3, use_baseline=True, weight_decay=0):
        self.env_class = env_class
        self.policy_net = policy_net
        self.policy_sampler = policy_sampler
        self.baseline = IncrementalStats()
        self.use_baseline = use_baseline

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )


    def get_policy(self, qap: QAP):
        # Compute probabilities for every pair p_ij = P(a_i -> b_j)
        logits = self.policy_net(qap)
        # Create policy to sample action from
        return self.policy_sampler(logits=logits, qap=qap)


    def run_episode(self, env: QAPReductionEnv, learn=True):
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

            if self.use_baseline:
                baselined_rewards = np.array(rewards) - self.baseline.mean()
            else:
                baselined_rewards = np.array(rewards)

            # Compute accumulated rewards
            returns = utils.reverse_cumsum(baselined_rewards)

            # Optimize
            loss = -torch.sum(torch.tensor(returns) * torch.stack(log_probs))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        self.episode_stats = {
            "value": env.reward_sum,
            "entropy_average": np.mean(entropies),
            "episode_entropies": np.array(entropies),
        }

        if learn:
            self.episode_stats["loss"] = loss.item()

        # Training statistics
        for name, param in self.policy_net.named_parameters():
            if param.grad is not None:
                gradient_magnitude = torch.norm(param.grad).item()
            else:
                gradient_magnitude = 0
            self.episode_stats["gradient_" + name] = gradient_magnitude

    def solve_and_learn(self, qap: QAP):
        env = self.env_class(qap)
        self.run_episode(env, learn=True)
        assert env.done
        return env.reward_sum, env.assignment

    def solve(self, qap: QAP):
        env = self.env_class(qap)
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
