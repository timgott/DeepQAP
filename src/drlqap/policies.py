import torch
import numpy as np

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


def sample_pair(logits, qap):
    # Assert shape
    n = qap.size
    assert logits.shape == (n, n), \
            f"{logits.shape} != {n},{n}"

    # Create distribution over pairs
    return Categorical2D(logits=logits, shape=(n,n))

class AssignmentSampler:
    def __init__(self, logits, qap):
        super().__init__()
        self.logits = logits
        self.qap = qap
        self.last_assignment = None

    def sample(self):
        n = self.qap.size
        nodes_a = list(range(n))
        nodes_b = list(range(n))
        assignment = [None] * n

        log_prob = 0
        while n > 0:
            remaining_logits = self.logits[nodes_a,:][:,nodes_b]
            distribution = Categorical2D(remaining_logits, shape=(n,n))
            pair = distribution.sample()
            log_prob = log_prob + distribution.log_prob(pair)
            i, j = pair
            a = nodes_a.pop(i)
            b = nodes_b.pop(j)
            assignment[a] = b
            n = n-1

        self.last_assignment = assignment
        self.last_log_prob = log_prob
        return assignment

    def log_prob(self, assignment):
        assert assignment == self.last_assignment
        return self.last_log_prob

    def entropy(self):
        return np.inf
