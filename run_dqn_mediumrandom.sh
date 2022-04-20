#!/bin/bash
agent=dqn_dense_ms_ec_eps0
for i in $(seq 4); do
    for lr in 1e-5 5e-4 3e-4; do
        for depth in 2 3 4; do
            for hs in 32 64; do
                python scripts/train_agent.py runs/"$agent"_mediumrandoms/lr"$lr"_s"$i" $agent medium_random_graphs --seed $i -lr $lr -gd $depth -hs $hs;
            done
        done
    done
done
