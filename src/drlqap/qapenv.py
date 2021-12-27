from drlqap import QAP

class QAPEnv:
    def __init__(self, qap: QAP):
        self.assignment = [None] * qap.size
        self.qap = qap
        self.remaining_qap = qap
        self.unassigned_a = list(range(qap.size))
        self.unassigned_b = list(range(qap.size))

        self.reward_sum = 0 # Used for assertion

    def get_state():
        return remaining_qap

    def get_unassigned_nodes():
        return self.unassigned_a, self.unassigned_b

    def step(self, i, j):
        a = self.unassigned_a.pop(a)
        b = self.unassigned_b.pop(b)
        assert self.assignment[a] is None
        self.assignment[a] = b

        subproblem = self.remaining_qap.create_subproblem_for_assignment(i, j)
        reward = subproblem.fixed_cost - self.remaining_qap.fixed_cost
        done = (subproblem.size == 0)

        if done:
            self.reward_sum += reward
            assert qap.compute_value(self.assignment) == self.reward_sum

        return reward, done

