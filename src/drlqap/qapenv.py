from drlqap.qap import QAP
from typing import Union, List
import numpy as np
import random
import torch


class QAPReductionEnv:
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
        if subproblem.size == 1:
            subproblem = subproblem.create_subproblem_for_assignment(0, 0)
            self.assignment[self.unassigned_a.pop()] = self.unassigned_b.pop()

        reward = self.remaining_qap.fixed_cost - subproblem.fixed_cost
        assert(reward <= 0)

        self.remaining_qap = subproblem
        self.done = (subproblem.size == 0)

        self.reward_sum += -reward
        if self.done:
            assert not self.unassigned_a
            assert not self.unassigned_b
            assert np.isclose(self.qap.compute_value(self.assignment).item(), self.reward_sum)

        return reward


class QAPPairLinkEnv(QAPReductionEnv):
    """
    Modification of the QAPEnv that only adds a new edge between two assigned nodes
    """

    def get_state(self):
        """
        Tuple of A matrix, B matrix and L matrix.
        """
        assert torch.all(self.qap.linear_costs == 0)
        n = self.qap.size
        assigned_a = [a for a in range(n) if self.assignment[a]]
        assigned_b = [self.assignment[a] for a in assigned_a]
        links = torch.zeros((n, n))
        links[assigned_a, assigned_b] = 1
        return self.qap, links, list(self.unassigned_a), list(self.unassigned_b)


class QAPOneStepEnv:
    """
    Solve the entire QAP in one step.
    """
    assignment: List[Union[int, None]]
    qap: QAP

    def __init__(self, qap: QAP):
        self.assignment = [None] * qap.size
        self.qap = qap
        self.done = False

        self.reward_sum = 0

    def get_state(self) -> QAP:
        return self.qap

    def step(self, assignment):
        self.done = True
        reward = self.qap.compute_value(assignment).item()

        # Used by the agents to get the final reward and assignment
        self.reward_sum = reward
        self.assignment = assignment

        return reward
