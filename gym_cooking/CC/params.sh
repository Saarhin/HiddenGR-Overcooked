#!/bin/bash

# Define an array
dqn_input=("Summary")

seeds=(1 2 3 4 5 6 7 8 9 10)

single=("--single-agent" "")

eps=(500000)

intervals=(10000)

map=("SarahDesignSingleAgent5x5" "SarahDesignOne")


output_file="parameters.txt"
> "$output_file"

for item in "${dqn_input[@]}"; do
    for seed in "${seeds[@]}"; do
        for s in "${single[@]}"; do
            for ep in "${eps[@]}"; do
                for i in "${intervals[@]}"; do
                    for m in "${map[@]}"; do

                        if [[ $s == "--single-agent" ]]; then
                            echo "--num-agents 1 --level $m --belief-experiments --seed $seed --dqn-input $item $s  --num-eps $ep --save-interval $i --folder-name singlefetcher" >> "$output_file"
                        else
                            echo "--num-agents 2 --level $m --belief-experiments --seed $seed --dqn-input $item $s  --num-eps $ep --save-interval $i --folder-name fetcherDQN_simpleChef" >> "$output_file"
                        fi
                    done
                done
            done
        done
    done
done
