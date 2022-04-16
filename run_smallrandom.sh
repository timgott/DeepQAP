#!/bin/bash
for i in $(seq 8); do
    python src/drlqap/train_agent.py runs/$1_smallrandoms/s$i $1 small_random_graphs $i &;
done
