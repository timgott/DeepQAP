#!/bin/bash
agent=dqn_bi
for i in $(seq 4); do
    for lr in 8e-4 5e-4 3e-4 1e-4; do
        python scripts/train_agent.py runs/"$agent"_increasing/lr"$lr"_s"$i" $agent increasing_1000 --seed $i -lr $lr --steps 15000;
    done
done
