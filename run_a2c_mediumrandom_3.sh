#!/bin/bash
for agent in a2c_ms100x; do
    for i in $(seq 4); do
        for lr in 4e-5 2e-5; do
            for md in 2 4 6; do
                python scripts/train_agent.py \
                    runs/a2c_ms100x_mediumrandoms/lr"$lr"_s"$i"_gd4_md"$md" \
                    $agent \
                    medium_random_graphs \
                    --seed $i  --steps 30000 \
                    -lr $lr --gnn_depth 4 --mlp_depth $md;
            done
        done
    done
done
