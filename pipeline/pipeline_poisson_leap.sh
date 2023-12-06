#!/bin/bash

cd ..
for i in {1..8}
do
  python -m src.nn_pde --pde poisson --ground_truth_resolution 32 --siren_omega 30.0 --siren_omega0 30.0 --viz_every 10 --log_every 5 --optimizer adam --outer_lr 2.5e-5 --num_layers 5 --layer_size 64 --bc_weight 1.0 --outer_steps 200 --outer_points 512 --validation_points 1024 --load_model_from_expt poisson_leap_results/default_7_cont/ --seed $i --expt_name cpu_pretrain_1_seed_$i

done
