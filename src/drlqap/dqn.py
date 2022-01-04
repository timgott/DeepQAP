from drlqap.qapenv import QAPEnv
from drlqap.qap import QAP
from drlqap import utils
import random
import torch
import math
from collections import deque, namedtuple
import copy

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self,
            policy_net,
            batch_size = 128,
            gamma = 0.999,
            eps_start = 0.9,
            eps_end = 0.05,
            eps_decay = 2000,
            policy_update_every = 10,
            target_update_every = 100,
            learning_rate = 1e-2
        ) -> None:
        self.policy_net = policy_net
        self.target_net = copy.deepcopy(policy_net)
        self.optimizer = torch.optim.RMSprop(
            policy_net.parameters(), 
            lr=learning_rate
        )
        self.memory = ReplayMemory(10000)

        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.episode_count = 0
        self.policy_update_every = policy_update_every
        self.target_update_every = target_update_every

    def compute_epsilon(self, i):
        return self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * i / self.eps_decay)

    def optimize_policy(self):
        if len(self.memory) < self.batch_size:
            return

        # Take sample from observations
        transitions = self.memory.sample(self.batch_size)

        # batching is not possible with the ReinforceNet class at the moment :/
        loss_sum = torch.tensor(0.)
        criterion = torch.nn.SmoothL1Loss()
        for t in transitions:
            q = self.policy_net(t.state)[t.action]
            if t.next_state:
                with torch.no_grad():
                    max_next_q = self.target_net(t.next_state).flatten().max()
            else:
                max_next_q = torch.tensor(0.)
            target_q = t.reward + max_next_q
            loss_sum += criterion(q, target_q)

        # Compute average loss
        loss = loss_sum / len(transitions)

        # Optimize
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()


    def step(self, env: QAPEnv, epsilon):
        state = env.get_state()

        if random.random() < epsilon:
            action = env.random_action()
        else:
            q_values = self.policy_net(state)
            action = utils.argmax2d(q_values)

        # Env step
        reward = env.step(action)

        # Save to memory
        next_state = env.get_state() if not env.done else None
        self.memory.push(state, action, next_state, reward)


    def run_episode(self, env: QAPEnv, learn=True):
        # epsilon-greedy when learning, greedy otherwise
        epsilon = self.compute_epsilon(self.episode_count) if learn else 0

        while not env.done:
            self.step(env, epsilon)

        self.episode_stats = {
            "value": env.reward_sum,
            "epsilon": epsilon,
        }

        if learn:
            self.episode_count += 1
            if self.episode_count % self.policy_update_every == 0:
                self.optimize_policy()
            if self.episode_count % self.target_update_every == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Training statistics
            gradient_magnitude = 0
            for param in self.policy_net.parameters():
                if param.grad is not None:
                    gradient_magnitude += torch.norm(param.grad).item()
            self.episode_stats["gradient_magnitude"] = gradient_magnitude

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
            'target_state_dict': self.target_net.state_dict()
            }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(
                checkpoint['policy_optimizer_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
