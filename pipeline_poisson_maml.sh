#!/bin/bash

python -m src.nn_pde_maml --pde poisson --ground_truth_resolution 32 --siren_omega 30.0 --siren_omega0 30.0 --viz_every 10 --log_every 5 --optimizer adam --outer_lr 1.0e-5 --measure_grad_norm_every 5 --grad_clip 100. --num_layers 3 --layer_size 64 --bc_weight 1.0 --outer_steps 200 --outer_points 1024 --validation_points 1024 --load_model_from_expt poisson_maml_results/default_2_cont/ --seed 1 --expt_name cpu_maml_pretrain_3_seed_1

python -m src.nn_pde_maml --pde poisson --ground_truth_resolution 32 --siren_omega 30.0 --siren_omega0 30.0 --viz_every 10 --log_every 5 --optimizer adam --outer_lr 1.0e-5 --measure_grad_norm_every 5 --grad_clip 100. --num_layers 3 --layer_size 64 --bc_weight 1.0 --outer_steps 200 --outer_points 1024 --validation_points 1024 --load_model_from_expt poisson_maml_results/default_2_cont/ --seed 2 --expt_name cpu_maml_pretrain_3_seed_2

python -m src.nn_pde_maml --pde poisson --ground_truth_resolution 32 --siren_omega 30.0 --siren_omega0 30.0 --viz_every 10 --log_every 5 --optimizer adam --outer_lr 1.0e-5 --measure_grad_norm_every 5 --grad_clip 100. --num_layers 3 --layer_size 64 --bc_weight 1.0 --outer_steps 200 --outer_points 1024 --validation_points 1024 --load_model_from_expt poisson_maml_results/default_2_cont/ --seed 3 --expt_name cpu_maml_pretrain_3_seed_3

python -m src.nn_pde_maml --pde poisson --ground_truth_resolution 32 --siren_omega 30.0 --siren_omega0 30.0 --viz_every 10 --log_every 5 --optimizer adam --outer_lr 1.0e-5 --measure_grad_norm_every 5 --grad_clip 100. --num_layers 3 --layer_size 64 --bc_weight 1.0 --outer_steps 200 --outer_points 1024 --validation_points 1024 --load_model_from_expt poisson_maml_results/default_2_cont/ --seed 4 --expt_name cpu_maml_pretrain_3_seed_4

python -m src.nn_pde_maml --pde poisson --ground_truth_resolution 32 --siren_omega 30.0 --siren_omega0 30.0 --viz_every 10 --log_every 5 --optimizer adam --outer_lr 1.0e-5 --measure_grad_norm_every 5 --grad_clip 100. --num_layers 3 --layer_size 64 --bc_weight 1.0 --outer_steps 200 --outer_points 1024 --validation_points 1024 --load_model_from_expt poisson_maml_results/default_2_cont/ --seed 5 --expt_name cpu_maml_pretrain_3_seed_5

python -m src.nn_pde_maml --pde poisson --ground_truth_resolution 32 --siren_omega 30.0 --siren_omega0 30.0 --viz_every 10 --log_every 5 --optimizer adam --outer_lr 1.0e-5 --measure_grad_norm_every 5 --grad_clip 100. --num_layers 3 --layer_size 64 --bc_weight 1.0 --outer_steps 200 --outer_points 1024 --validation_points 1024 --load_model_from_expt poisson_maml_results/default_2_cont/ --seed 6 --expt_name cpu_maml_pretrain_3_seed_6

python -m src.nn_pde_maml --pde poisson --ground_truth_resolution 32 --siren_omega 30.0 --siren_omega0 30.0 --viz_every 10 --log_every 5 --optimizer adam --outer_lr 1.0e-5 --measure_grad_norm_every 5 --grad_clip 100. --num_layers 3 --layer_size 64 --bc_weight 1.0 --outer_steps 200 --outer_points 1024 --validation_points 1024 --load_model_from_expt poisson_maml_results/default_2_cont/ --seed 7 --expt_name cpu_maml_pretrain_3_seed_7

python -m src.nn_pde_maml --pde poisson --ground_truth_resolution 32 --siren_omega 30.0 --siren_omega0 30.0 --viz_every 10 --log_every 5 --optimizer adam --outer_lr 1.0e-5 --measure_grad_norm_every 5 --grad_clip 100. --num_layers 3 --layer_size 64 --bc_weight 1.0 --outer_steps 200 --outer_points 1024 --validation_points 1024 --load_model_from_expt poisson_maml_results/default_2_cont/ --seed 8 --expt_name cpu_maml_pretrain_3_seed_8

