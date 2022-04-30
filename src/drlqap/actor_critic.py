import numpy as np
import torch
from drlqap.qap import QAP
from drlqap.qapenv import QAPReductionEnv

class A2CAgent:
    def __init__(self, env_class, policy_net, critic_net, policy_sampler, p_learning_rate, c_learning_rate, p_weight_decay=0, c_weight_decay=0):
        self.env_class = env_class
        self.policy_net = policy_net
        self.critic_net = critic_net
        self.policy_sampler = policy_sampler

        self.p_optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=p_learning_rate,
            weight_decay=p_weight_decay,
        )
        self.c_optimizer = torch.optim.Adam(
            self.critic_net.parameters(),
            lr=c_learning_rate,
            weight_decay=c_weight_decay,
        )


    def get_policy(self, qap: QAP):
        # Compute probabilities for every pair p_ij = P(a_i -> b_j)
        logits = self.policy_net(qap)
        # Create policy to sample action from
        return self.policy_sampler(logits=logits, qap=qap)


    def run_episode(self, env: QAPReductionEnv, learn=True):
        # Statistics
        entropies = []
        advantages = []

        while not env.done:
            qap = env.get_state()
            policy = self.get_policy(qap)
            pair = policy.sample()

            # Entropy stats
            with torch.no_grad():
                entropies.append(policy.entropy())

            # Env step
            reward = env.step(pair)

            if learn:
                next_state = env.get_state()
                current_state_value = self.critic_net(qap)

                # Compute advantage / TD-Error
                with torch.no_grad(): # no gradient of advantage required
                    next_state_value = self.critic_net(next_state) if not env.done else 0
                    q = reward + next_state_value
                    advantage = q - current_state_value
                    advantages.append(advantage.item())

                # Critic updates state value weight to minimize squared error to q value
                # Gradient: -advantage * grad(current_state_value)
                critic_loss = -(advantage * current_state_value)

                # Optimize critic
                self.c_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                self.c_optimizer.step()

                # Actor updates policy weights to increase action advantage
                policy_loss = -(advantage * policy.log_prob(pair))

                # Optimize actor
                self.p_optimizer.zero_grad(set_to_none=True)
                policy_loss.backward()
                self.p_optimizer.step()

        self.episode_stats = {
            "value": env.reward_sum,
            "entropy_average": np.mean(entropies),
            "entropy_steps": np.array(entropies),
        }

        if learn:
            self.episode_stats["advantage_average"] = np.mean(advantages)
            self.episode_stats["advantage_steps"] = np.array(advantages)


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
            'policy_optimizer_state_dict': self.p_optimizer.state_dict(),
            'critic_state_dict': self.critic_net.state_dict(),
            'critic_optimizer_state_dict': self.c_optimizer.state_dict(),
            }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.p_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.critic_net.load_state_dict(checkpoint['critic_state_dict'])
        self.c_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
