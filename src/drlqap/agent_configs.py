from drlqap import reinforce_nets
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
def reinforce_mp_transformer_no_norm():
    return ReinforceAgent(reinforce_nets.mp_transformer(64,32,32,3, layer_norm=False))

@define_agent_config
def reinforce_mp_transformer():
    return ReinforceAgent(reinforce_nets.mp_transformer(64,32,32,3))

@define_agent_config
def reinforce_mp_gat():
    return ReinforceAgent(reinforce_nets.mp_gat(64,32,32,3))

@define_agent_config
def reinforce_mp_transformer_low_lr():
    return ReinforceAgent(reinforce_nets.mp_transformer(64,32,32,3), learning_rate=1e-5)

@define_agent_config
def reinforce_mp_transformer_histogram():
    return ReinforceAgent(reinforce_nets.mp_histogram_transformer(64,32,3))
