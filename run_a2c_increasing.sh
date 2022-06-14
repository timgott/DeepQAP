#!/bin/bash
agent=a2c_ms100x
for i in $(seq 4); do
    for lr in 8e-5 5e-5 3e-5 1e-4; do
        python scripts/train_agent.py runs/"$agent"_increasing/lr"$lr"_s"$i" $agent increasing_simple_1000 --seed $i -lr $lr --steps 15000;
    done
done
