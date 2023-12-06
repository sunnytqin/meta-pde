#!/bin/bash


#scp -r ubuntu@3.18.105.148:/home/ubuntu/meta-pde/hyper_elasticity_leap_results/full_default* ../../hyper_elasticity_leap_results/
#scp -r ubuntu@3.23.87.8:/home/ubuntu/meta-pde/td_burgers_leap_results/default_final* ../../td_burgers_leap_results/
#scp -r ubuntu@18.223.116.143:/home/ubuntu/meta-pde/poisson_nn_results/cpu_maml_pretrain_3* ../../poisson_nn_results/
#scp -r ubuntu@3.22.242.197:/home/ubuntu/meta-pde/hyper_elasticity_nn_results/cpu* ../../hyper_elasticity_nn_results/
#scp -r ubuntu@18.191.107.188:/home/ubuntu/meta-pde/hyper_elasticity_nn_results/cpu_pretain_1* ../../td_burgers_nn_results/
#scp -r ubuntu@3.139.72.219:/home/ubuntu/meta-pde/td_burgers_fenics_results/jax ../../td_burgers_fenics_results/
#cp -r ubuntu@18.223.116.143:/home/ubuntu/meta-pde/td_burgers_nn_results/cpu_pretrain_1* ../../td_burgers_nn_results/
#scp -r ubuntu@3.15.201.36:/home/ubuntu/meta-pde/poisson_nn_results/gpu_maml_pretrain_3* ../../poisson_nn_results/
#scp -r ubuntu@18.222.232.9:/home/ubuntu/meta-pde/hyper_elasticity_fenics_results/pore_res ../../hyper_elasticity_fenics_results/
#scp -r ubuntu@18.218.104.130:/home/ubuntu/meta-pde/td_burgers_nn_results/cpu_pretrain_1* ../../td_burgers_nn_results/
#scp -r ubuntu@18.219.119.188:/home/ubuntu/meta-pde/hyper_elasticity_nn_results/gpu* ../../hyper_elasticity_nn_results/
#scp -r ubuntu@18.223.116.143:/home/ubuntu/meta-pde/src/nn_pde_maml.py ../../src/

scp -r ../../pipeline  ubuntu@3.143.219.93:/home/ubuntu/meta-pde/
scp -r ../../src  ubuntu@3.143.219.93:/home/ubuntu/meta-pde/
#scp -r ../../*.sh  ubuntu@18.118.150.64:/home/ubuntu/meta-pde/
#scp -r ../../poisson_maml_results/default_2_cont/ ubuntu@18.118.150.64:/home/ubuntu/meta-pde/poisson_maml_results/
#scp -r ../../poisson_leap_results/default_7_cont/ ubuntu@18.118.150.64:/home/ubuntu/meta-pde/poisson_leap_results/
#scp -r ../../hyper_elasticity_maml_results/default_2_cont/ ubuntu@18.219.119.188:/home/ubuntu/meta-pde/hyper_elasticity_maml_results/
#scp -r ../../hyper_elasticity_leap_results/full_3  ubuntu@18.118.150.64:/home/ubuntu/meta-pde/hyper_elasticity_leap_results/
#scp -r ../../td_burgers_maml_results/default_5  ubuntu@18.118.150.64:/home/ubuntu/meta-pde/td_burgers_maml_results/
#scp -r ../../td_burgers_leap_results/default_final_cont  ubuntu@18.218.104.130:/home/ubuntu/meta-pde/td_burgers_leap_results/
#scp -r ubuntu@18.220.177.72:/home/ubuntu/meta-pde/pip*.sh ../../
#scp -r ubuntu@3.139.59.225:/home/ubuntu/meta-pde/src/burgers/td_burgers_common.py ../../src/burgers/

echo Done

# useful command 

#Notes:
