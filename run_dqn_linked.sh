#!/bin/bash
agent=dqn_linkedqap
for i in $(seq 4); do
    for lr in 5e-4 5e-5 1e-4; do
        for depth in 2 3 4; do
            for hs in 32 64; do
                python scripts/train_agent.py runs/"$agent"_smallrandoms/lr"$lr"_s"$i" $agent small_random_graphs --seed $i -lr $lr -gd $depth -hs $hs;
            done
        done
    done
done
