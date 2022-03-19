from typing import Optional
import torch
from torch._C import Graph
from drlqap.qap import GraphAssignmentProblem, QAP
from drlqap.qapenv import QAPEnv
from drlqap.reinforce import ReinforceAgent
from matrix import MatrixDataSource
import numpy as np
import random
import threading
import concurrent.futures
from functools import partial

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
            self.qap = qap
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
    def __init__(self, doc):
        super().__init__()

        self.state: Optional[AgentState] = None
        self.agent = None
        self.qap: Optional[GraphAssignmentProblem] = None

        self.probability_matrix_source = MatrixDataSource('a', 'b', ['p', 'l'])
        self.node_embedding_sources = (
            MatrixDataSource('i', 'j', ['base']),
            MatrixDataSource('i', 'j', ['base']),
        )

        self.qap_viewmodel = QapStateViewModel(doc)
        self.doc = doc

    def update_state(self):
        state = self.state
        assert(state)

        # Node labels
        nodes_a = state.env.unassigned_a
        nodes_b = state.env.unassigned_b

        # Compute probabilities
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

        # Update data sources (asynchronously)
        def update_ui_data():
            self.probability_matrix_source.set_data(a=nodes_a, b=nodes_b, p=probs, l=log_probs)

            for i in (0,1):
                data_source = self.node_embedding_sources[i]
                indices = nodes_a if i == 0 else nodes_b
                data_source.set_data(i=indices, base=state.embeddings[i])
        self.doc.add_next_tick_callback(update_ui_data)

    def update_qap_state(self):
        # Compute QAP state
        state = self.state
        assert(state)
        nodes_a = state.env.unassigned_a
        nodes_b = state.env.unassigned_b
        self.qap_viewmodel.update_qap(qap=state.qap, nodes_a=nodes_a, nodes_b=nodes_b)

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
            self.update_qap_state()

    def assignment_step(self, a, b):
        assert(self.state)
        self.state.assignment_step(a, b)
        self.update_state()
        self.update_qap_state()

def random_assignment(qap: QAP):
    unassigned_targets = list(range(qap.size))
    random.shuffle(unassigned_targets)
    return unassigned_targets

class QapStateViewModel:
    def __init__(self, doc):
        super().__init__()
        self.matrix_a_source = MatrixDataSource('i', 'j', ['A'])
        self.matrix_b_source = MatrixDataSource('i', 'j', ['B'])
        self.linear_costs_source = MatrixDataSource('i', 'j', ['L'])
        self.estimated_value_source = MatrixDataSource('a', 'b', ['average', 'minimum'])
        self.doc = doc
        self.qap = None
        self.qap_updated = threading.Event()

        self.value_thread = threading.Thread(target=self.compute_value)
        self.value_thread.start()

    def update_qap(self, qap, nodes_a, nodes_b):
        if qap == self.qap:
            return

        self.qap = qap
        self.nodes_a = nodes_a # Labels of nodes in A
        self.nodes_b = nodes_b # Labels of nodes in B

        def update_ui_data():
            self.matrix_a_source.set_data(i=nodes_a, j=nodes_a, A=qap.A)
            self.matrix_b_source.set_data(i=nodes_b, j=nodes_b, B=qap.B)
            self.linear_costs_source.set_data(i=nodes_a, j=nodes_b, L=qap.linear_costs)
        self.doc.add_next_tick_callback(update_ui_data)

        self.qap_updated.set()


    def compute_value(self):
        qap = None
        max_samples = 20000

        samples = 0
        while True:
            if samples >= max_samples or not qap or self.qap_updated.is_set():
                self.qap_updated.wait()
                self.qap_updated.clear()
                samples = 0

                qap = self.qap
                nodes_a = np.array(self.nodes_a)
                nodes_b = np.array(self.nodes_b)
                n = self.qap.size

                counts = np.zeros((n,n))
                sums = np.zeros((n,n))
                averages = np.zeros((n,n))
                minimum = np.ones((n,n)) * np.inf

            for _ in range(1000):
                samples += 1
                assignment = random_assignment(qap)
                value = qap.compute_value(assignment)
                for i,j in enumerate(assignment):
                    counts[i,j] += 1
                    sums[i,j] += -value
                    minimum[i,j] = min(value,minimum[i,j])
            averages = sums / counts

            presented_data = dict(
                a=nodes_a,
                b=nodes_b,
                average=averages,
                minimum=-minimum
            )

            def update_ui_data():
                self.estimated_value_source.set_data(**presented_data)

            self.doc.add_next_tick_callback(update_ui_data)
