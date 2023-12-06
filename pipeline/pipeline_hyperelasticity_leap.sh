#!/bin/bash

cd ..
for i in {1..8}
do
  python -m src.nn_pde --pde hyper_elasticity --ground_truth_resolution 64 --xmin 0.0 --xmax 1.0 --ymin 0.0 --ymax 1.0 --max_holes 5 --max_hole_size 1.0 --novary_source --novary_bc --max_reynolds 100. --siren_omega 30.0 --siren_omega0 30.0 --viz_every 10 --log_every 5 --optimizer adam --outer_lr 5.0e-6 --measure_grad_norm_every 10 --grad_clip 1000. --num_layers 10 --layer_size 128 --bc_weight 1. --outer_steps 200 --inner_points 1024 --outer_points 1024 --validation_points 1024 --load_model_from_expt hyper_elasticity_leap_results/full_3 --seed $i --expt_name cpu_pretrain_1_seed_$i

done