#!/bin/bash
agent=reinforce_ms100x
for i in $(seq 4); do
    for lr in 8e-5 4e-5 2e-5; do
        python scripts/train_agent.py runs/"$agent"_smallrandoms/lr"$lr"_s"$i" $agent small_random_graphs --seed $i -lr $lr;
    done
done
