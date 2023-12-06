#!/bin/bash

cd ..
for i in {1..8}
do
  python -m src.nn_pde_maml --pde td_burgers --ground_truth_resolution 512 --xmin 0.0 --xmax 1.0 --num_tsteps 201 --tmax 1.0 --novary_source --max_reynolds 100. --siren_omega 30.0 --siren_omega0 30.0 --viz_every 10 --log_every 5 --optimizer adam --outer_lr 5.0e-6 --measure_grad_norm_every 5 --grad_clip 100. --inner_grad_clip 100. --num_layers 8 --layer_size 64 --bc_weight 1. --outer_steps 200 --inner_points 1024 --outer_points 1024 --validation_points 1024 --load_model_from_expt td_burgers_maml_results/default_5/ --seed $i --expt_name cpu_maml_pretrain_1_seed_$i

done