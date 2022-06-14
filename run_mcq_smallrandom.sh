#!/bin/bash
for agent in mcq_eps0; do
    for i in $(seq 4); do
        for lr in 4e-5 8e-5 3e-4 ; do
            python scripts/train_agent.py runs/mcq_eps0_smallrandoms/lr"$lr"_s"$i" $agent small_random_graphs --seed $i -lr $lr --steps 30000;
        done
    done
done
