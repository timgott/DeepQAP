from torch.nn.functional import layer_norm
from drlqap import reinforce_nets
from drlqap.dqn import DQNAgent
from drlqap.reinforce import ReinforceAgent
from drlqap import dqn_nets
from drlqap import utils
import drlqap.nn

agents = dict()
agent_training_steps = dict()

def define_agent_config(training_steps=10000):
    def decorator(f):
        name = f.__name__
        assert name not in agents
        agents[name] = f
        agent_training_steps[name] = training_steps
        return f
    return decorator

@define_agent_config()
def reinforce_simple_histogram():
    return ReinforceAgent(reinforce_nets.link_prediction_only_undirected(32,64,3))

@define_agent_config()
def reinforce_simple_node_embeddings():
    return ReinforceAgent(reinforce_nets.simple_node_embeddings_undirected(32,32,64,3))

@define_agent_config()
def reinforce_simple_node_embeddings_normalized():
    return ReinforceAgent(reinforce_nets.simple_node_embeddings_undirected(32,32,64,3,normalize_embeddings=True))

@define_agent_config()
def reinforce_gat():
    return ReinforceAgent(reinforce_nets.mp_gat(64,32,32,3))

@define_agent_config()
def reinforce_gat_low_lr():
    return ReinforceAgent(reinforce_nets.mp_gat(64,32,32,3), learning_rate=1e-5)

@define_agent_config()
def reinforce_gat_histogram():
    return ReinforceAgent(reinforce_nets.mp_histogram_gat(64,32,3))

@define_agent_config()
def reinforce_gat_histogram_shallow():
    return ReinforceAgent(reinforce_nets.mp_histogram_gat(64,32,1))

@define_agent_config()
def dqn_gat():
    return DQNAgent(dqn_nets.mp_gat(64,32,32,3), eps_decay=3465.7)

@define_agent_config()
def dqn_gat_histogram():
    return DQNAgent(dqn_nets.mp_histogram_gat(64,32,3), learning_rate=1e-2)

@define_agent_config()
def dqn_gat_histogram_lower_lr():
    return DQNAgent(dqn_nets.mp_histogram_gat(64,32,3), learning_rate=1e-3)

@define_agent_config()
def dqn_gat_histogram_low_lr():
    return DQNAgent(dqn_nets.mp_histogram_gat(64,32,3), learning_rate=1e-4)

@define_agent_config()
def dqn_gat_histogram_low_epsilon():
    return DQNAgent(
        dqn_nets.mp_histogram_gat(64,32,3),
        learning_rate=1e-3,
        eps_decay=utils.decay_halflife(1000)
    )

@define_agent_config()
def dqn_simple_lp():
    return DQNAgent(
        dqn_nets.simple_link_prediction_undirected(64,32,3),
        learning_rate=1e-3,
        eps_decay=utils.decay_halflife(2000)
    )

@define_agent_config()
def dqn_simpler_lp():
    return DQNAgent(
        dqn_nets.simple_link_prediction_undirected(8,32,3),
        learning_rate=1e-4,
        eps_decay=utils.decay_halflife(2000)
    )

@define_agent_config()
def dqn_gat_no_lp():
    return DQNAgent(
        dqn_nets.mp_gat_no_lp(64,32,32,3),
        learning_rate=1e-3,
        eps_decay=utils.decay_halflife(2000),
        eps_end=0
    )

@define_agent_config()
def dqn_dense():
    return DQNAgent(
        dqn_nets.dense(32, 1, 2, layer_norm=True),
        learning_rate=5e-4,
        eps_decay=utils.decay_halflife(2000),
        eps_end=0
    )

@define_agent_config()
def dqn_dense_learnmore():
    return DQNAgent(
        dqn_nets.dense(32, 1, 2, layer_norm=True),
        learning_rate=5e-4,
        eps_decay=utils.decay_halflife(2000),
        eps_end=0,
        batch_size=128
    )

@define_agent_config()
def dqn_dense_3_0():
    return DQNAgent(
        dqn_nets.dense(32, 0, 3, layer_norm=True),
        learning_rate=5e-4,
        eps_decay=utils.decay_halflife(2000),
        eps_end=0
    )

@define_agent_config()
def dqn_dense_max():
    return DQNAgent(
        dqn_nets.dense(32, 1, 2, layer_norm=True, l_aggr='min', q_aggr='max'),
        learning_rate=5e-4,
        eps_decay=utils.decay_halflife(2000),
        eps_end=0
    )

@define_agent_config()
def dqn_dense_no_c():
    return DQNAgent(
        dqn_nets.dense(32, 1, 2, layer_norm=True, combined_transform=False),
        learning_rate=5e-4,
        eps_decay=utils.decay_halflife(2000),
        eps_end=0
    )

@define_agent_config(training_steps=50000)
def dqn_dense_no_c_long():
    return DQNAgent(
        dqn_nets.dense(32, 1, 2, layer_norm=True, combined_transform=False),
        learning_rate=5e-4,
        eps_decay=utils.decay_through(50000, 0.005),
        eps_end=0
    )

@define_agent_config()
def dqn_dense_bn():
    return DQNAgent(
        dqn_nets.dense(32, 1, 2, layer_norm=True, conv_norm='batch_norm'),
        learning_rate=5e-4,
        eps_decay=utils.decay_halflife(2000),
        eps_end=0
    )

@define_agent_config()
def dqn_dense_bn_lr_up():
    return DQNAgent(
        dqn_nets.dense(32, 1, 2, layer_norm=True, conv_norm='batch_norm'),
        learning_rate=5e-3,
        eps_decay=utils.decay_halflife(2000),
        eps_end=0
    )

@define_agent_config()
def dqn_dense_kmn():
    return DQNAgent(
        dqn_nets.dense(32, 1, 2, layer_norm=True, conv_norm='keep_mean'),
        learning_rate=5e-4,
        eps_decay=utils.decay_halflife(2000),
        eps_end=0
    )

@define_agent_config()
def dqn_dense_tmn():
    return DQNAgent(
        dqn_nets.dense(32, 1, 2, layer_norm=True, conv_norm='transformed_mean'),
        learning_rate=5e-4,
        eps_decay=utils.decay_halflife(2000),
        eps_end=0
    )

@define_agent_config()
def dqn_dense_tmn_slower():
    return DQNAgent(
        dqn_nets.dense(32, 1, 2, layer_norm=True, conv_norm='transformed_mean'),
        learning_rate=1e-5,
        eps_decay=utils.decay_halflife(2000),
        eps_end=0
    )

@define_agent_config(training_steps=30000)
def dqn_dense_tmn_slow_long():
    return DQNAgent(
        dqn_nets.dense(32, 1, 2, layer_norm=True, conv_norm='transformed_mean'),
        learning_rate=5e-5,
        eps_decay=utils.decay_through(30000, 0.01),
        eps_end=0
    )

@define_agent_config()
def dqn_dense_tmn_rni():
    return DQNAgent(
        dqn_nets.dense(32, 1, 2, layer_norm=True, conv_norm='transformed_mean', random_start=True),
        learning_rate=5e-4,
        eps_decay=utils.decay_halflife(2000),
        eps_end=0
    )

@define_agent_config(training_steps=30000)
def dqn_dense_tmn_rni_long():
    return DQNAgent(
        dqn_nets.dense(32, 1, 2, layer_norm=True, conv_norm='transformed_mean', random_start=True),
        learning_rate=5e-5,
        eps_decay=utils.decay_through(30000, 0.01),
        eps_end=0
    )

@define_agent_config()
def dqn_dense_tmn_tup():
    return DQNAgent(
        dqn_nets.dense(32, 1, 2, layer_norm=True, conv_norm='transformed_mean'),
        learning_rate=5e-4,
        eps_decay=utils.decay_halflife(2000),
        eps_end=0,
        target_update_every=10
    )

@define_agent_config()
def dqn_dense_tmn_eps0():
    return DQNAgent(
        dqn_nets.dense(32, 1, 2, layer_norm=True, conv_norm='transformed_mean'),
        learning_rate=5e-4,
        eps_start=0,
        eps_end=0,
    )

@define_agent_config()
def reinforce_dense():
    return ReinforceAgent(
        dqn_nets.dense(32, 1, 2, layer_norm=True),
        learning_rate=5e-4,
    )

@define_agent_config()
def reinforce_dense_slow():
    return ReinforceAgent(
        dqn_nets.dense(32, 1, 2, layer_norm=True),
        learning_rate=1e-4,
    )

@define_agent_config(training_steps=30000)
def reinforce_dense_slow_long():
    return ReinforceAgent(
        dqn_nets.dense(32, 1, 2, layer_norm=True),
        learning_rate=1e-4,
    )

@define_agent_config()
def reinforce_dense_3_0():
    return ReinforceAgent(
        dqn_nets.dense(32, 0, 3, layer_norm=True),
        learning_rate=5e-4,
    )

@define_agent_config()
def reinforce_dense_3_0_slower():
    return ReinforceAgent(
        dqn_nets.dense(32, 0, 3, layer_norm=True),
        learning_rate=1e-5,
    )

@define_agent_config()
def reinforce_dense_3_0_faster():
    return ReinforceAgent(
        dqn_nets.dense(32, 0, 3, layer_norm=True),
        learning_rate=1e-3,
    )

@define_agent_config()
def reinforce_dense_tmn():
    return ReinforceAgent(
        dqn_nets.dense(32, 1, 2, layer_norm=True, conv_norm='transformed_mean'),
        learning_rate=1e-3,
    )

@define_agent_config()
def reinforce_dense_bn():
    return ReinforceAgent(
        dqn_nets.dense(32, 1, 2, layer_norm=True, conv_norm='batch_norm'),
        learning_rate=1e-4,
    )

@define_agent_config()
def reinforce_dense_bn_lr2():
    return ReinforceAgent(
        dqn_nets.dense(32, 1, 2, layer_norm=True, conv_norm='batch_norm'),
        learning_rate=1e-3,
    )

@define_agent_config(training_steps=30000)
def reinforce_tmn_slow_long():
    return ReinforceAgent(
        dqn_nets.dense(32, 1, 2, layer_norm=True, conv_norm='transformed_mean'),
        learning_rate=1e-5,
    )

@define_agent_config(training_steps=30000)
def reinforce_tmn_fast_long():
    return ReinforceAgent(
        dqn_nets.dense(32, 1, 2, layer_norm=True, conv_norm='transformed_mean'),
        learning_rate=1e-3,
    )

@define_agent_config(training_steps=30000)
def reinforce_tmn_medium_long():
    return ReinforceAgent(
        dqn_nets.dense(32, 1, 2, layer_norm=True, conv_norm='transformed_mean'),
        learning_rate=1e-4,
    )

@define_agent_config(training_steps=30000)
def reinforce_tmn_medium_long_nobl():
    return ReinforceAgent(
        dqn_nets.dense(32, 1, 2, layer_norm=True, conv_norm='transformed_mean'),
        learning_rate=1e-4,
        use_baseline=False
    )

@define_agent_config(training_steps=30000)
def reinforce_tmn_slower_long():
    return ReinforceAgent(
        dqn_nets.dense(32, 1, 2, layer_norm=True, conv_norm='transformed_mean'),
        learning_rate=1e-6,
    )

@define_agent_config(training_steps=30000)
def reinforce_tmn_rni():
    return ReinforceAgent(
        dqn_nets.dense(32, 1, 2, layer_norm=True, conv_norm='transformed_mean', random_start=True),
        learning_rate=1e-4,
    )

