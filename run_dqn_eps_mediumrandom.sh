#!/bin/bash
agent=dqn_dense_ms_ec
for i in $(seq 4); do
    for lr in 5e-4 3e-4; do
        for depth in 2 3; do
            for hs in 32; do
                python scripts/train_agent.py runs/"$agent"_mediumrandoms/lr"$lr"_s"$i" $agent medium_random_graphs --seed $i -lr $lr -gd $depth -hs $hs --steps 15000;
            done
        done
    done
done
