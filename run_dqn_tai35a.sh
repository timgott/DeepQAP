#!/bin/bash
agent=dqn_dense_ms_ec
for i in $(seq 4); do
    for lr in 5e-4 3e-4 1e-4; do
	python scripts/train_agent.py runs/"$agent"_tai35a/lr"$lr"_s"$i" $agent qaplib_tai35a_normalized --seed $i -lr $lr --steps 20000;
    done
done
