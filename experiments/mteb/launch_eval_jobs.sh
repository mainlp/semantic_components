#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

# Define the range of values for mcs, ms, mu, and alpha
orders=(1)
is=(0)

# Loop through all combinations of values

for order in "${orders[@]}"; do
    for i in "${is[@]}"; do
        # Call run_fct.py with the current settings
        python mteb_eval.py $i cuda:0 $order &
    done
done