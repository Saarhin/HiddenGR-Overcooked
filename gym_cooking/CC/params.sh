#!/bin/bash

# Define an array
dqn_input=("Full" "Summary")

seeds=(1 2 3 4 5 6 7 8 9 10)


output_file="parameters_leaps.txt"
> "$output_file"

for item in "${dqn_input[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "--num-agents 2 --level SarahDesignOne --record --belief-experiments --seed $seed --dqn-input $item" >> "$output_file"
    done
done
