#!/bin/bash
agent=dqn_dense_ms_ec_eps0
for i in $(seq 4); do
	python scripts/train_agent.py runs/"$agent"_mediumrandoms/lr"$lr"_s"$i" $agent medium_random_graphs --seed $i -lr $lr -gd $depth -hs $hs;
done
