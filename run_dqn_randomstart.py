#!/usr/bin/env python
from drlqap.experiment import run_experiment_from_config

agent = "dqn_dense_ms_ec_eps0"
for seed in range(4):
    for rs in [False, True]:
        args = {
            'learning_rate': 5e-4,
            'random_start': rs,
            'gnn_depth': 3,
        }
        path = f'runs/{agent}_rni_study/s{seed}_{"rs" if rs else "nors"}'
        run_experiment_from_config(
            experiment_path=path,
            agent_name=agent,
            task_name="small_random_graphs",
            agent_arguments=args,
            seed=seed
        )
