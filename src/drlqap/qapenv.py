from drlqap.qap import QAP
from typing import Union, List
import numpy as np
import random


class QAPEnv:
    assignment: List[Union[int, None]]
    qap: QAP
    remaining_qap: QAP
    unassigned_a: List[int]
    unassigned_b: List[int]

    def __init__(self, qap: QAP):
        self.assignment = [None] * qap.size
        self.qap = qap
        self.remaining_qap = qap
        self.unassigned_a = list(range(qap.size))
        self.unassigned_b = list(range(qap.size))
        self.done = False

        self.reward_sum = 0  # Used for assertion

    def get_state(self) -> QAP:
        return self.remaining_qap

    def get_unassigned_nodes(self):
        return self.unassigned_a, self.unassigned_b

    def random_action(self):
        n = self.remaining_qap.size
        return (random.randrange(n), random.randrange(n))

    def step(self, pair):
        i, j = pair
        a = self.unassigned_a.pop(i)
        b = self.unassigned_b.pop(j)
        assert self.assignment[a] is None
        self.assignment[a] = b

        subproblem = self.remaining_qap.create_subproblem_for_assignment(i, j)
        reward = self.remaining_qap.fixed_cost - subproblem.fixed_cost
        assert(reward <= 0)

        self.remaining_qap = subproblem
        self.done = (subproblem.size == 0)

        self.reward_sum += -reward
        if self.done:
            assert np.isclose(self.qap.compute_value(self.assignment).item(), self.reward_sum)

        return reward
