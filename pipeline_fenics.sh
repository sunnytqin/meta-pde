#!/bin/bash

taskset -c 0 python -m src.fenics_baseline --pde poisson --ground_truth_resolution 64 --relaxation_parameter 0.01 --spatial_resolutions 2,4,8,16,32 --time_resolutions 1 --boundary_resolutions 4,16,64,256,512 --n_eval 16 --expt_name cpu_pore_res

#taskset -c 0 python -m src.fenics_baseline --pde td_burgers --xmin 0.0 --xmax 1.0 --ground_truth_resolution 512 --spatial_resolutions 16,32,64,128,256 --time_resolutions 1,2,3,4 --novary_source --max_reynolds 100. --num_tsteps 9 --n_eval 8 --expt_name cpu_default_2 > td_burgers.out &

#taskset -c 1 python -m src.fenics_baseline --pde hyper_elasticity --xmin 0.0 --ymin 0.0 --max_holes 5 --max_hole_size 1.0 --ground_truth_resolution 64 --relaxation_parameter 0.01 --novary_source --novary_bc --spatial_resolutions 4,8,16,32 --boundary_resolutions 2,4,6,8,10,12 --n_eval 8 --expt_name cpu_pore_res > hyper_elasticity.out &
