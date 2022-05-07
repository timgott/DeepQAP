#!/bin/bash
for agent in a2c_ms100x; do
    for i in $(seq 4); do
        for lr in 2e-4 8e-5 4e-5; do
            python scripts/train_agent.py runs/a2c_ms100x_mediumrandoms/lr"$lr"_s"$i" $agent medium_random_graphs --seed $i -lr $lr --steps 30000;
        done
    done
done
