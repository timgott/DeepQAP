from drlqap import reinforce_nets
from drlqap.dqn import DQNAgent
from drlqap.reinforce import ReinforceAgent
from drlqap import dqn_nets
from drlqap import utils

agents = dict()

def define_agent_config(f):
    assert f.__name__ not in agents
    agents[f.__name__] = f
    return f

@define_agent_config
def reinforce_simple_histogram():
    return ReinforceAgent(reinforce_nets.link_prediction_only_undirected(32,64,3))

@define_agent_config
def reinforce_simple_node_embeddings():
    return ReinforceAgent(reinforce_nets.simple_node_embeddings_undirected(32,32,64,3))

@define_agent_config
def reinforce_simple_node_embeddings_normalized():
    return ReinforceAgent(reinforce_nets.simple_node_embeddings_undirected(32,32,64,3,normalize_embeddings=True))

@define_agent_config
def reinforce_gat():
    return ReinforceAgent(reinforce_nets.mp_gat(64,32,32,3))

@define_agent_config
def reinforce_gat_low_lr():
    return ReinforceAgent(reinforce_nets.mp_gat(64,32,32,3), learning_rate=1e-5)

@define_agent_config
def reinforce_gat_histogram():
    return ReinforceAgent(reinforce_nets.mp_histogram_gat(64,32,3))

@define_agent_config
def reinforce_gat_histogram_shallow():
    return ReinforceAgent(reinforce_nets.mp_histogram_gat(64,32,1))

@define_agent_config
def dqn_gat():
    return DQNAgent(dqn_nets.mp_gat(64,32,32,3), eps_decay=3465.7)

@define_agent_config
def dqn_gat_histogram():
    return DQNAgent(dqn_nets.mp_histogram_gat(64,32,3), learning_rate=1e-2)

@define_agent_config
def dqn_gat_histogram_lower_lr():
    return DQNAgent(dqn_nets.mp_histogram_gat(64,32,3), learning_rate=1e-3)

@define_agent_config
def dqn_gat_histogram_low_lr():
    return DQNAgent(dqn_nets.mp_histogram_gat(64,32,3), learning_rate=1e-4)

@define_agent_config
def dqn_gat_histogram_low_epsilon():
    return DQNAgent(
        dqn_nets.mp_histogram_gat(64,32,3),
        learning_rate=1e-3,
        eps_decay=utils.exp_halflife(1000)
    )

@define_agent_config
def dqn_simple_lp():
    return DQNAgent(
        dqn_nets.simple_link_prediction_undirected(64,32,3),
        learning_rate=1e-3,
        eps_decay=utils.exp_halflife(2000)
    )

@define_agent_config
def dqn_gat_no_lp():
    return DQNAgent(
        dqn_nets.mp_gat_no_lp(64,32,32,3),
        learning_rate=1e-3,
        eps_decay=utils.exp_halflife(2000),
        eps_end=0
    )
