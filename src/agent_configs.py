import reinforce_nets
from reinforce import ReinforceAgent

agents = dict()

def define_agent_config(f):
    assert f.__name__ not in agents
    agents[f.__name__] = f
    return f

@define_agent_config
def reinforce_simple_histogram():
    return ReinforceAgent(reinforce_nets.link_prediction_only(32,64,3))