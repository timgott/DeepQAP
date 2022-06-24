#!/usr/bin/env python
from drlqap.experiment import run_experiment_from_config

agent = "a2c_ms100x"
for seed in range(4):
    for norm in [None, 'pair_norm', 'keep_mean', 'mean_separation', 'mean_separation_100x']:
        args = {
            'learning_rate': 4e-5,
            'gnn_norm': norm,
        }
        path = f'runs/{agent}_norm_study/s{seed}_{norm}'
        run_experiment_from_config(
            experiment_path=path,
            agent_name=agent,
            task_name="small_random_graphs",
            agent_arguments=args,
            seed=seed,
            steps=20000
        )
