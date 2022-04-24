#!/bin/bash
for agent in a2c a2c_ms100x; do
    for i in $(seq 4); do
        for lr in 8e-4 5e-4 2e-4; do
            python scripts/train_agent.py runs/"$agent"_mediumrandoms/lr"$lr"_s"$i" $agent medium_random_graphs --seed $i -lr $lr;
        done
    done
done
