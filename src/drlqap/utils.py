import numpy as np
import torch
import math

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

def decay_halflife(n):
    return decay_through(n, 0.5)

def decay_through(n, v):
    return -math.log(v) / n
