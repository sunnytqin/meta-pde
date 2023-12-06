#!/bin/bash

cd ..

nohup python -m src.leap_pde --pde poisson --xmin -1.0 --ymin -1.0 --ground_truth_resolution 32 --siren_omega 30.0 --siren_omega0 30.0 --viz_every 1_000 --log_every 500 --optimizer adam --inner_lr 2.5e-5 --outer_lr 5.0e-5 --measure_grad_norm_every 1000 --inner_steps 60 --num_layers 5 --layer_size 64 --bc_weight 1.0 --outer_steps 500_000 --inner_points 4096 --outer_points 4096 --validation_points 4096 --n_eval 8 --bsize 8 --expt_name default_final &

nohup python -m src.leap_pde --pde td_burgers --xmin 0.0 --max_reynolds 100. --ground_truth_resolution 512 --num_tsteps 201 --siren_omega 30.0 --siren_omega0 30.0 --viz_every 1_000 --log_every 500 --optimizer adam --outer_lr 5.0e-5 --inner_lr 1.0e-6 --inner_steps 80 --num_layers 10 --layer_size 128 --bc_weight 1.0 --outer_steps 500_000 --inner_points 2048 --outer_points 2048 --validation_points 2048 --n_eval 8 --bsize 8 --novary_bc --novary_source --vary_geometry --expt_name default_final &

nohup python -m src.leap_pde --pde hyper_elasticity --xmin 0.0 --ymin 0.0 --max_holes 5 --max_hole_size 0.5 --max_reynolds 100. --ground_truth_resolution 32 --relaxation_parameter 0.2 --siren_omega 30.0 --siren_omega0 30.0 --viz_every 2_000 --log_every 500 --optimizer adam --outer_lr 5.0e-6 --inner_lr 5.0e-6 --grad_clip 1000. --inner_grad_clip 1000. --measure_grad_norm_every 500 --inner_steps 20 --num_layers 10 --layer_size 128 --bc_weight 1.0 --outer_steps 500_000 --inner_points 2048 --outer_points 2048 --validation_points 1024 --n_eval 8 --bsize 8 --novary_bc --novary_source --vary_geometry --expt_name full_default &