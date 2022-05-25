#!/usr/bin/env python
from drlqap.experiment import run_experiment_from_config

agent = "dqn_dense_ec_eps0"
for seed in range(4):
    for depth in [2,3,4]:
        for norm in [None, 'batch_norm', 'mean_separation', 'mean_separation_100x']:
            args = {
                'learning_rate': 5e-4,
                'gnn_norm': norm,
                'gnn_depth': depth
            }
            path = f'runs/{agent}_norm_study/s{seed}_{norm}'
            run_experiment_from_config(
                experiment_path=path,
                agent_name=agent,
                task_name="small_random_graphs",
                agent_arguments=args,
                seed=seed
            )
