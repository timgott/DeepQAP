import numpy as np
import torch
import math
import random
from drlqap import utils
from drlqap.qap import QAP
from drlqap.qapenv import QAPReductionEnv

class MonteCarloQAgent:
    def __init__(self,
            env_class,
            policy_net,
            learning_rate=1e-3,
            eps_start = 0.9,
            eps_end = 0.05,
            eps_decay = 0.0005,
            weight_decay=0
        ):
        self.env_class = env_class
        self.policy_net = policy_net
        self.episode_count = 0
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay


    def compute_epsilon(self, i):
        t = math.exp(-1. * i * self.eps_decay)
        return self.eps_end + (self.eps_start - self.eps_end) * t


    def run_episode(self, env: QAPReductionEnv, learn=True):
        # epsilon-greedy when learning, greedy otherwise
        epsilon = self.compute_epsilon(self.episode_count) if learn else 0

        # Rewards
        rewards = []
        predicted_returns = []
        while not env.done:
            # Compute q-values (even if taking random step, required for training)
            state = env.get_state()
            q_values = self.policy_net(state)

            # epsilon-greedy policy
            if random.random() < epsilon:
                action = env.random_action()
            else:
                action = utils.argmax2d(q_values)

            # Env step
            reward = env.step(action)

            # Save data for training
            if learn:
                rewards.append(reward)
                predicted_returns.append(q_values[action])

        self.episode_stats = {
            "qvalue": predicted_returns[0].item(),
            "value": env.reward_sum,
            "epsilon": epsilon,
        }

        if learn:
            self.episode_count += 1

            # Compute accumulated rewards
            returns = utils.reverse_cumsum(rewards)

            # Optimize
            criterion = torch.nn.MSELoss()
            loss = criterion(torch.tensor(returns), torch.stack(predicted_returns))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.episode_stats["loss"] = loss.item()

            # Training statistics
            gradient_magnitudes = []
            for _, param in self.policy_net.named_parameters():
                if param.grad is not None:
                    gradient_magnitudes.append(torch.norm(param.grad).item())
            self.episode_stats["gradient_average"] = np.mean(gradient_magnitudes)

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
            }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(
                checkpoint['policy_optimizer_state_dict'])
