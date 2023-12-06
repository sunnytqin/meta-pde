#!/bin/bash

cd ..
for i in {1..8}
do
  python -m src.nn_pde --pde td_burgers --ground_truth_resolution 512 --xmin 0.0 --xmax 1.0 --num_tsteps 201 --tmax 1.0 --novary_source --max_reynolds 100. --siren_omega 30.0 --siren_omega0 30.0 --viz_every 10 --log_every 5 --optimizer adam --outer_lr 1.0e-6 --num_layers 10 --layer_size 128 --bc_weight 1. --outer_steps 200 --outer_points 512 --validation_points 1024 --load_model_from_expt td_burgers_leap_results/default_final_cont/ --seed $i --expt_name cpu_pretrain_1_seed_$i

done