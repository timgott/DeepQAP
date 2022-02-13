from typing import Optional
import torch
from torch._C import Graph
from drlqap.qap import GraphAssignmentProblem
from drlqap.qapenv import QAPEnv
from drlqap.reinforce import ReinforceAgent
from matrix import MatrixDataSource

class AgentState:
    def __init__(self, qap: GraphAssignmentProblem, agent: ReinforceAgent) -> None:
        self.agent = agent
        self.env = QAPEnv(qap)
        self.compute_net_state()

    def has_policy_distribution(self):
        return hasattr(self.agent, "get_policy")

    def get_policy(self):
        with torch.no_grad():
            return self.agent.get_policy(self.env.get_state())

    def compute_net_state(self):
        with torch.no_grad():
            qap = self.env.get_state()
            net = self.agent.policy_net
            probs = net(qap)
            self.probs = probs
            self.embeddings = (net.embeddings_a, net.embeddings_b)

    def assignment_step(self, a, b):
        try:
            i = self.env.unassigned_a.index(a)
            j = self.env.unassigned_b.index(b)
        except ValueError:
            print("invalid assignment")
        else:
            self.env.step((i,j))
            self.compute_net_state()
            print(self.env.reward_sum)

class AgentStateViewModel:
    def __init__(self):
        super().__init__()

        self.state: Optional[AgentState] = None
        self.agent = None
        self.qap: Optional[GraphAssignmentProblem] = None

        self.probability_matrix_source = MatrixDataSource('a', 'b', ['p', 'l'])
        self.node_embedding_sources = (
            MatrixDataSource('i', 'j', ['base']),
            MatrixDataSource('i', 'j', ['base']),
        )

    def update_state(self):
        state = self.state
        assert(state)

        nodes_a = state.env.unassigned_a
        nodes_b = state.env.unassigned_b
        with torch.no_grad():
            if state.has_policy_distribution():
                policy = state.get_policy()
                n = len(nodes_a)
                m = len(nodes_b)
                probs = policy.distribution.probs.reshape(n, m)
                log_probs = state.probs
            else:
                probs = state.probs
                log_probs = probs

        self.probability_matrix_source.set_data(a=nodes_a, b=nodes_b, p=probs, l=log_probs)

        for i in (0,1):
            data_source = self.node_embedding_sources[i]
            data_source.set_data(base=state.embeddings[i])

    def set_agent(self, agent):
        self.agent = agent
        self.reset_state()

    def set_qap(self, qap):
        self.qap = qap
        self.reset_state()

    def reset_state(self):
        if self.qap is not None and self.agent is not None:
            self.state = AgentState(self.qap, self.agent)
            self.update_state()

    def assignment_step(self, a, b):
        assert(self.state)
        self.state.assignment_step(a, b)
        self.update_state()

