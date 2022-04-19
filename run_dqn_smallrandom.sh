#!/bin/bash
agent=dqn_dense_ms_ec_eps0
for i in $(seq 4); do
    for lr in 1e-5 5e-4 3e-4 1e-4; do
        python scripts/train_agent.py runs/"$agent"_smallrandoms/s$i $agent small_random_graphs --seed $i -lr $lr;
    done
done
