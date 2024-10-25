#!/bin/bash

# Define the range of values for mcs, ms, mu, and alpha
mcs_values=(50)
ms_values=(50)
mu_values=(1.0 0.95 0.9 0.8 0.7 0.6 0.5)
alpha_values=(0.0 0.05 0.1 0.2 0.3 0.4 0.5)
seeds=(0)

# Loop through all combinations of values

for mcs in "${mcs_values[@]}"; do
    for ms in "${ms_values[@]}"; do
        for mu in "${mu_values[@]}"; do
            for alpha in "${alpha_values[@]}"; do
                for seed in "${seeds[@]}"; do
                    # Call run_fct.py with the current settings
                    python run_fct.py $mcs $ms $alpha $mu $seed &
                done
            wait
            done
        done
    done
done