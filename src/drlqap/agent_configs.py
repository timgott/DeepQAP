from drlqap.dqn import DQNAgent
from drlqap.reinforce import ReinforceAgent
from drlqap.actor_critic import A2CAgent
from drlqap.mcq import MonteCarloQAgent
from drlqap import nn_configs
from drlqap import utils
from drlqap import policies
from drlqap.qapenv import QAPReductionEnv, QAPPairLinkEnv, QAPOneStepEnv
import torch

agents = dict()
agent_training_steps = dict()

def define_agent_config(training_steps=10000):
    def decorator(f, name=None):
        name = name or f.__name__
        assert name not in agents
        agents[name] = f
        agent_training_steps[name] = training_steps
        return f
    return decorator


@define_agent_config()
def dqn_simple_lp():
    return DQNAgent(
        QAPReductionEnv,
        nn_configs.simple_link_prediction_undirected(64,32,3),
        learning_rate=1e-3,
        eps_decay=utils.decay_halflife(2000)
    )

@define_agent_config()
def dqn_simpler_lp():
    return DQNAgent(
        QAPReductionEnv,
        nn_configs.simple_link_prediction_undirected(8,32,3),
        learning_rate=1e-4,
        eps_decay=utils.decay_halflife(2000)
    )

@define_agent_config()
def dqn_dense():
    return DQNAgent(
        QAPReductionEnv,
        nn_configs.dense(32, 1, 2, layer_norm=True),
        learning_rate=5e-4,
        eps_decay=utils.decay_halflife(2000),
        eps_end=0
    )

@define_agent_config()
def dqn_dense_learnmore():
    return DQNAgent(
        QAPReductionEnv,
        nn_configs.dense(32, 1, 2, layer_norm=True),
        learning_rate=5e-4,
        eps_decay=utils.decay_halflife(2000),
        eps_end=0,
        batch_size=128
    )

@define_agent_config()
def dqn_dense_3_0():
    return DQNAgent(
        QAPReductionEnv,
        nn_configs.dense(32, 0, 3, layer_norm=True),
        learning_rate=5e-4,
        eps_decay=utils.decay_halflife(2000),
        eps_end=0
    )

@define_agent_config()
def dqn_dense_max():
    return DQNAgent(
        QAPReductionEnv,
        nn_configs.dense(32, 1, 2, layer_norm=True, l_aggr='min', q_aggr='max'),
        learning_rate=5e-4,
        eps_decay=utils.decay_halflife(2000),
        eps_end=0
    )

@define_agent_config()
def dqn_dense_no_c():
    return DQNAgent(
        QAPReductionEnv,
        nn_configs.dense(32, 1, 2, layer_norm=True, combined_transform=False),
        learning_rate=5e-4,
        eps_decay=utils.decay_halflife(2000),
        eps_end=0
    )

@define_agent_config(training_steps=50000)
def dqn_dense_no_c_long():
    return DQNAgent(
        QAPReductionEnv,
        nn_configs.dense(32, 1, 2, layer_norm=True, combined_transform=False),
        learning_rate=5e-4,
        eps_decay=utils.decay_through(50000, 0.005),
        eps_end=0
    )

@define_agent_config()
def dqn_dense_bn():
    return DQNAgent(
        QAPReductionEnv,
        nn_configs.dense(32, 1, 2, layer_norm=True, conv_norm='batch_norm'),
        learning_rate=5e-4,
        eps_decay=utils.decay_halflife(2000),
        eps_end=0
    )

@define_agent_config()
def dqn_dense_bn_lr_up():
    return DQNAgent(
        QAPReductionEnv,
        nn_configs.dense(32, 1, 2, layer_norm=True, conv_norm='batch_norm'),
        learning_rate=5e-3,
        eps_decay=utils.decay_halflife(2000),
        eps_end=0
    )

@define_agent_config()
def dqn_dense_kmn():
    return DQNAgent(
        QAPReductionEnv,
        nn_configs.dense(32, 1, 2, layer_norm=True, conv_norm='keep_mean'),
        learning_rate=5e-4,
        eps_decay=utils.decay_halflife(2000),
        eps_end=0
    )

@define_agent_config()
def dqn_dense_tmn():
    return DQNAgent(
        QAPReductionEnv,
        nn_configs.dense(32, 1, 2, layer_norm=True, conv_norm='transformed_mean'),
        learning_rate=5e-4,
        eps_decay=utils.decay_halflife(2000),
        eps_end=0
    )

@define_agent_config()
def dqn_dense_tmn_slower():
    return DQNAgent(
        QAPReductionEnv,
        nn_configs.dense(32, 1, 2, layer_norm=True, conv_norm='transformed_mean'),
        learning_rate=1e-5,
        eps_decay=utils.decay_halflife(2000),
        eps_end=0
    )

@define_agent_config(training_steps=30000)
def dqn_dense_tmn_slow_long():
    return DQNAgent(
        QAPReductionEnv,
        nn_configs.dense(32, 1, 2, layer_norm=True, conv_norm='transformed_mean'),
        learning_rate=5e-5,
        eps_decay=utils.decay_through(30000, 0.01),
        eps_end=0
    )

@define_agent_config()
def dqn_dense_tmn_rni():
    return DQNAgent(
        QAPReductionEnv,
        nn_configs.dense(32, 1, 2, layer_norm=True, conv_norm='transformed_mean', random_start=True),
        learning_rate=5e-4,
        eps_decay=utils.decay_halflife(2000),
        eps_end=0
    )

@define_agent_config(training_steps=30000)
def dqn_dense_tmn_rni_long():
    return DQNAgent(
        QAPReductionEnv,
        nn_configs.dense(32, 1, 2, layer_norm=True, conv_norm='transformed_mean', random_start=True),
        learning_rate=5e-5,
        eps_decay=utils.decay_through(30000, 0.01),
        eps_end=0
    )

@define_agent_config()
def dqn_dense_tmn_tup():
    return DQNAgent(
        QAPReductionEnv,
        nn_configs.dense(32, 1, 2, layer_norm=True, conv_norm='transformed_mean'),
        learning_rate=5e-4,
        eps_decay=utils.decay_halflife(2000),
        eps_end=0,
        target_update_every=10
    )

@define_agent_config()
def dqn_dense_tmn_eps0():
    return DQNAgent(
        QAPReductionEnv,
        nn_configs.dense(32, 1, 2, layer_norm=True, conv_norm='transformed_mean'),
        learning_rate=5e-4,
        eps_start=0,
        eps_end=0,
    )

@define_agent_config()
def dqn_dense_tmn_eps0_noln():
    return DQNAgent(
        QAPReductionEnv,
        nn_configs.dense(32, 1, 2, layer_norm=False, conv_norm='transformed_mean'),
        learning_rate=5e-4,
        eps_start=0,
        eps_end=0,
    )

@define_agent_config()
def dqn_dense_nonorm_eps0():
    return DQNAgent(
        QAPReductionEnv,
        nn_configs.dense(32, 1, 2, layer_norm=False, conv_norm=None),
        learning_rate=5e-4,
        eps_start=0,
        eps_end=0,
    )

@define_agent_config()
def dqn_dense_ms_eps0():
    return DQNAgent(
        QAPReductionEnv,
        nn_configs.dense(32, 1, 2, layer_norm=False, conv_norm='mean_separation', use_edge_encoder=True, combined_transform=False),
        learning_rate=5e-4,
        eps_start=0,
        eps_end=0,
    )

# worse than plain dqn_dense_ms_eps0
@define_agent_config()
def dqn_dense_ms100x_eps0():
    return DQNAgent(
        QAPReductionEnv,
        nn_configs.dense(32, 1, 2, layer_norm=False, conv_norm='mean_separation_100x', use_edge_encoder=True, combined_transform=False),
        learning_rate=5e-4,
        eps_start=0,
        eps_end=0,
    )

@define_agent_config()
def dqn_dense_tmn_ec_eps0():
    return DQNAgent(
        QAPReductionEnv,
        nn_configs.dense(32, 1, 2, layer_norm=False, conv_norm='transformed_mean', use_edge_encoder=True, combined_transform=False),
        learning_rate=5e-4,
        eps_start=0,
        eps_end=0,
    )

@define_agent_config()
def dqn_dense_ms_ec_eps0(learning_rate=5e-4, hidden_size=32, mlp_depth=1, gnn_depth=2, random_start=False):
    return dqn_dense_ec_eps0(learning_rate, hidden_size, mlp_depth, gnn_depth, random_start, gnn_norm='mean_separation')

@define_agent_config()
def dqn_dense_ec_eps0(learning_rate=5e-4, hidden_size=32, mlp_depth=1, gnn_depth=2, random_start=False, gnn_norm=None):
    return DQNAgent(
        QAPReductionEnv,
        nn_configs.dense(hidden_size, mlp_depth, gnn_depth, layer_norm=False, conv_norm=gnn_norm, use_edge_encoder=True, combined_transform=False, random_start=random_start),
        learning_rate=learning_rate,
        eps_start=0,
        eps_end=0,
    )

@define_agent_config()
def dqn_dense_ms_ec(learning_rate=5e-4, hidden_size=32, mlp_depth=1, gnn_depth=2, random_start=False):
    return DQNAgent(
        QAPReductionEnv,
        nn_configs.dense(hidden_size, mlp_depth, gnn_depth, layer_norm=False, conv_norm='mean_separation', use_edge_encoder=True, combined_transform=False, random_start=random_start),
        learning_rate=learning_rate,
        eps_start=0.5,
        eps_end=0.005,
    )

@define_agent_config()
def dqn_bi(learning_rate=5e-4, hidden_size=32, mlp_depth=1, gnn_depth=2, random_start=False, gnn_norm='mean_separation'):
    return DQNAgent(
        QAPReductionEnv,
        nn_configs.dense(hidden_size, mlp_depth, gnn_depth, layer_norm=False, conv_norm=gnn_norm, use_edge_encoder=True, combined_transform=False, random_start=random_start, bidirectional=True),
        learning_rate=learning_rate,
        eps_start=0,
        eps_end=0,
    )

@define_agent_config()
def dqn_linkedqap(learning_rate=5e-4, hidden_size=32, mlp_depth=1, gnn_depth=2, random_start=False):
    return DQNAgent(
        QAPPairLinkEnv,
        nn_configs.mpgnn_pairs_linkedqap(hidden_size, mlp_depth, gnn_depth, conv_norm='mean_separation', use_edge_encoder=True, combined_transform=False, random_start=random_start),
        learning_rate=learning_rate,
        eps_start=0,
        eps_end=0,
    )

@define_agent_config()
def dqn_linkedqap_eps(learning_rate=5e-4, hidden_size=32, mlp_depth=1, gnn_depth=2, random_start=False):
    return DQNAgent(
        QAPPairLinkEnv,
        nn_configs.mpgnn_pairs_linkedqap(hidden_size, mlp_depth, gnn_depth, conv_norm='mean_separation', use_edge_encoder=True, combined_transform=False, random_start=random_start),
        learning_rate=learning_rate,
        eps_start=0.9,
        eps_end=0.005,
    )

@define_agent_config()
def dqn_mse_eps0(learning_rate=5e-4, hidden_size=32, mlp_depth=1, gnn_depth=2, random_start=False):
    return DQNAgent(
        QAPReductionEnv,
        nn_configs.dense(hidden_size, mlp_depth, gnn_depth, layer_norm=False, conv_norm='mean_separation', use_edge_encoder=True, combined_transform=False, random_start=random_start),
        learning_rate=learning_rate,
        eps_start=0.0,
        eps_end=0.0,
        use_mse_loss=True
    )

@define_agent_config()
def reinforce_dense():
    return ReinforceAgent(
        QAPReductionEnv,
        nn_configs.dense(32, 1, 2, layer_norm=True),
        policies.sample_pair,
        learning_rate=5e-4,
    )

@define_agent_config(training_steps=30000)
def reinforce_ms():
    return ReinforceAgent(
        QAPReductionEnv,
        nn_configs.dense(32, 1, 2, layer_norm=False, conv_norm='mean_separation', use_edge_encoder=True, combined_transform=False),
        policies.sample_pair,
        learning_rate=1e-4,
    )

@define_agent_config(training_steps=30000)
def reinforce(learning_rate=1e-4, gnn_depth=2, mlp_depth=1, hidden_size=32, weight_decay=0, gnn_norm='mean_separation_100x'):
    return ReinforceAgent(
        QAPReductionEnv,
        nn_configs.dense(hidden_size, mlp_depth, gnn_depth, layer_norm=False, conv_norm=gnn_norm, use_edge_encoder=True, combined_transform=False),
        policies.sample_pair,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

# For compatibility
define_agent_config(training_steps=30000)(reinforce, name="reinforce_ms100x")

@define_agent_config(training_steps=30000)
def reinforce_ms100x_onestep(learning_rate=1e-4, gnn_depth=2, mlp_depth=1, hidden_size=32, weight_decay=0):
    return ReinforceAgent(
        QAPOneStepEnv,
        nn_configs.dense(hidden_size, mlp_depth, gnn_depth, layer_norm=False, conv_norm='mean_separation_100x', use_edge_encoder=True, combined_transform=False),
        policies.AssignmentSampler,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

@define_agent_config()
def a2c(learning_rate=5e-4, gnn_depth=2, mlp_depth=1, hidden_size=32, weight_decay=0):
    return A2CAgent(
        QAPReductionEnv,
        nn_configs.mpgnn_pairs(hidden_size, mlp_depth, gnn_depth, use_edge_encoder=True),
        nn_configs.mpgnn_global(hidden_size, mlp_depth, gnn_depth, use_edge_encoder=True),
        policies.sample_pair,
        p_learning_rate=learning_rate,
        c_learning_rate=learning_rate,
        p_weight_decay=weight_decay,
        c_weight_decay=weight_decay
    )

@define_agent_config()
def a2c_ms100x(learning_rate=5e-4, gnn_depth=2, mlp_depth=1, hidden_size=32, weight_decay=0):
    return A2CAgent(
        QAPReductionEnv,
        nn_configs.mpgnn_pairs(hidden_size, mlp_depth, gnn_depth, use_edge_encoder=True, conv_norm='mean_separation_100x'),
        nn_configs.mpgnn_global(hidden_size, mlp_depth, gnn_depth, use_edge_encoder=True),
        policies.sample_pair,
        p_learning_rate=learning_rate,
        c_learning_rate=learning_rate,
        p_weight_decay=weight_decay,
        c_weight_decay=weight_decay
    )

@define_agent_config()
def a2c_ms100x_cyclelr(gnn_depth=2, mlp_depth=1, hidden_size=32, weight_decay=0):
    agent = A2CAgent(
        QAPReductionEnv,
        nn_configs.mpgnn_pairs(hidden_size, mlp_depth, gnn_depth, use_edge_encoder=True, conv_norm='mean_separation_100x'),
        nn_configs.mpgnn_global(hidden_size, mlp_depth, gnn_depth, use_edge_encoder=True),
        policies.sample_pair,
        p_learning_rate=5e-4,
        c_learning_rate=5e-4,
        p_weight_decay=weight_decay,
        c_weight_decay=weight_decay
    )
    p_scheduler = torch.optim.lr_scheduler.CyclicLR(agent.p_optimizer, base_lr=5e-6, max_lr=4e-4, mode='triangular2', cycle_momentum=False, step_size_up=2500)
    c_scheduler = torch.optim.lr_scheduler.CyclicLR(agent.c_optimizer, base_lr=5e-6, max_lr=4e-4, mode='triangular2', cycle_momentum=False, step_size_up=2500)
    agent.schedulers.extend([p_scheduler, c_scheduler])
    return agent

@define_agent_config()
def a2c_ms100x_steplr(learning_rate=5e-4, gnn_depth=2, mlp_depth=1, hidden_size=32, weight_decay=0):
    agent = A2CAgent(
        QAPReductionEnv,
        nn_configs.mpgnn_pairs(hidden_size, mlp_depth, gnn_depth, use_edge_encoder=True, conv_norm='mean_separation_100x'),
        nn_configs.mpgnn_global(hidden_size, mlp_depth, gnn_depth, use_edge_encoder=True),
        policies.sample_pair,
        p_learning_rate=learning_rate,
        c_learning_rate=learning_rate,
        p_weight_decay=weight_decay,
        c_weight_decay=weight_decay
    )
    p_scheduler = torch.optim.lr_scheduler.StepLR(agent.p_optimizer, step_size=10000, gamma=0.5)
    c_scheduler = torch.optim.lr_scheduler.StepLR(agent.c_optimizer, step_size=10000, gamma=0.5)
    agent.schedulers.extend([p_scheduler, c_scheduler])
    return agent

@define_agent_config()
def mcq_eps0(learning_rate=5e-4, hidden_size=32, mlp_depth=1, gnn_depth=2, gnn_norm='mean_separation_100x'):
    return MonteCarloQAgent(
        QAPReductionEnv,
        nn_configs.mpgnn_pairs(hidden_size, mlp_depth, gnn_depth, use_edge_encoder=True, conv_norm=gnn_norm),
        learning_rate=learning_rate,
        eps_start=0,
        eps_end=0,
    )

