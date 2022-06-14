#!/bin/bash
for agent in a2c_ms100x; do
    for i in $(seq 4); do
        for lr in 4e-5 2e-5 8e-5; do
            python scripts/train_agent.py runs/a2c_ms100x_smallrandoms/lr"$lr"_s"$i" $agent small_random_graphs --seed $i -lr $lr --steps 40000;
        done
    done
done
