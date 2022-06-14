from drlqap.qapenv import QAPReductionEnv
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
            env_class,
            policy_net,
            batch_size = 16,
            gamma = 0.999,
            eps_start = 0.9,
            eps_end = 0.05,
            eps_decay = 0.0005,
            policy_update_every = 1,
            target_update_every = 100,
            learning_rate = 1e-3,
            use_mse_loss=False
        ) -> None:
        self.env_class = env_class
        self.policy_net = policy_net
        self.target_net = copy.deepcopy(policy_net)
        self.optimizer = torch.optim.Adam(
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
        self.use_mse_loss = use_mse_loss

    def compute_epsilon(self, i):
        return self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * i * self.eps_decay)

    def optimize_policy(self):
        if len(self.memory) < self.batch_size:
            return

        # Take sample from observations
        transitions = self.memory.sample(self.batch_size)

        loss_sum = torch.tensor(0.)
        if self.use_mse_loss:
            criterion = torch.nn.MSELoss()
        else:
            criterion = torch.nn.SmoothL1Loss() # Probably worse but default because most runs used it
        
        # batching is not possible with the network class at the moment :/        
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

        return loss.item()


    def step(self, env: QAPReductionEnv, epsilon):
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


    def run_episode(self, env: QAPReductionEnv, learn=True):
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
                loss = self.optimize_policy()
                self.episode_stats["loss"] = loss
            if self.episode_count % self.target_update_every == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

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
            'target_state_dict': self.target_net.state_dict()
            }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(
                checkpoint['policy_optimizer_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
