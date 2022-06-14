#!/bin/bash
agent=dqn_mse_eps0
for i in $(seq 4); do
    for lr in 5e-4 3e-4; do
        python scripts/train_agent.py runs/"$agent"_smallrandoms/lr"$lr"_s"$i" $agent small_random_graphs --seed $i -lr $lr --steps 20000;
    done
done
