#!/bin/bash
agent=dqn_dense_ms_ec_eps0
for i in $(seq 4); do
    for lr in 1e-4 5e-4 3e-4; do
        for depth in 2 3 4; do
		python scripts/train_agent.py runs/"$agent"_mini/lr"$lr"_s"$i" $agent minilinear --seed $i -lr $lr -gd $depth --steps 5000;
        done
    done
done
