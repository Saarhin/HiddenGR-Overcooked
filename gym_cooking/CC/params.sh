#!/bin/bash

# Define an array
dqn_input=("Summary")

seeds=(1 2 3 4 5 6 7 8 9 10)

single=("--single-agent")

eps=(500000)

intervals=(10000)


output_file="parameters.txt"
> "$output_file"

for item in "${dqn_input[@]}"; do
    for seed in "${seeds[@]}"; do
        for s in "${single[@]}"; do
            for ep in "${eps[@]}"; do
                for i in "${intervals[@]}"; do

                    if [[ $s == "--single-agent" ]]; then
                        echo "--num-agents 1 --level SarahDesignSingleAgent5x5 --belief-experiments --seed $seed --dqn-input $item $s  --num-eps $ep --save-interval $i --folder-name singlefetcher5x5_1" >> "$output_file"
                    else
                        echo "--num-agents 1 --level SarahDesignSingleAgent5x5 --belief-experiments --seed $seed --dqn-input $item $s  --num-eps $ep --save-interval $i --folder-name fetcherDQN_simpleChef5x5" >> "$output_file"
                    fi
                done
            done
        done
    done
done
