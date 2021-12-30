import numpy as np
import torch
from typing import Tuple

class IncrementalStats():
    def __init__(self) -> None:
        self.sum = 0
        self.sqr_sum = 0
        self.count = 0

    def add(self, value):
        self.sum += value
        self.sqr_sum += value * value
        self.count += 1

    def mean(self):
        return self.sum / self.count

    def variance(self):
        m = self.mean()
        return self.sqr_sum / self.count - m * m

    def std(self):
        return np.sqrt(self.variance())

    def normalize(self, value, min_std=1.0):
        return (value - self.mean()) / max(self.std(), min_std)

    def state_dict(self):
        return {
            "sum": self.sum,
            "sqr_sum": self.sqr_sum,
            "count": self.count
        }

    def load_state_dict(self, state):
        self.sum = state["sum"]
        self.sqr_sum = state["sqr_sum"]
        self.count = state["count"]


class Categorical2D:
    def __init__(self, logits, shape) -> None:
        assert(logits.shape == shape or logits.shape == shape + (1,))
        assert(len(shape) == 2)
        flat_logits = logits.reshape((-1,))
        self.shape = shape
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


def reverse_cumsum(data):
    cumulative = np.array(data)
    flipped = np.flip(cumulative)
    np.cumsum(flipped, out=flipped)
    assert(np.allclose(cumulative[0], sum(data)))
    assert(np.allclose(cumulative[-1], data[-1]))
    return cumulative

def argmax2d(x):
    max_index = torch.argmax(x)
    return np.unravel_index(max_index, shape=x.shape)
