#!/bin/bash

cd ..
for i in {1..8}
do
   python -m src.nn_pde_maml --pde poisson --ground_truth_resolution 32 --siren_omega 30.0 --siren_omega0 30.0 --viz_every 10 --log_every 5 --optimizer adam --outer_lr 1.0e-5 --measure_grad_norm_every 5 --grad_clip 100. --num_layers 3 --layer_size 64 --bc_weight 1.0 --outer_steps 200 --outer_points 1024 --validation_points 1024 --load_model_from_expt poisson_maml_results/default_2_cont/ --seed $i --expt_name cpu_maml_pretrain_3_seed_$i

done