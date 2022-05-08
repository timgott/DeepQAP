#!/bin/bash
for agent in a2c_ms100x_cyclelr; do
    for i in $(seq 4); do
        python scripts/train_agent.py \
            runs/a2c_ms100x_mediumrandoms_cyclic/s"$i" \
            $agent \
            medium_random_graphs \
            --seed $i --steps 40000;
    done
done
