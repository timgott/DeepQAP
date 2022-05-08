#!/bin/bash
for agent in a2c_ms100x_steplr; do
    for i in $(seq 4); do
        python scripts/train_agent.py \
            runs/a2c_ms100x_mediumrandoms_stepped/s"$i" \
            $agent \
            medium_random_graphs \
            --learning_rate 8e-5 --seed $i --steps 30000;
    done
done
