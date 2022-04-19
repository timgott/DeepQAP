#!/bin/bash
agent=reinforce_ms100x
for i in $(seq 4); do
    for lr in 5e-4 3e-4 1e-4; do
        python scripts/train_agent.py runs/"$agent"_smallrandoms/lr"$lr"_s"$i" $agent small_random_graphs --seed $i -lr $lr;
    done
done
