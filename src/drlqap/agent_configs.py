from drlqap import reinforce_nets
from drlqap.dqn import DQNAgent
from drlqap.reinforce import ReinforceAgent

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
def dqn_gat():
    return DQNAgent(reinforce_nets.mp_gat(64,32,32,3))
