#!/bin/bash
agent=dqn_linkedqap_eps
for i in $(seq 4); do
    for lr in 1e-5 5e-5 1e-4 5e-4; do
        python scripts/train_agent.py runs/"$agent"_smallrandoms/lr"$lr"_s"$i" $agent small_random_graphs --seed $i -lr $lr;
    done
done
