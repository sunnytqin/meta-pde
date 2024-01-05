# meta-PDE
meta-learning initializations for mesh-free amortization of PDE solving  
[paper](https://arxiv.org/abs/2211.01604)

## Setup:
1. Follow `setup.sh` (I would suggest against running it as a bash file because the entire process takes a while)
2. The package versions are a bit outdated but please use the versions listed in the script until I update the code
     -  two packages used baselines (Fenics and mshr) are very finicky.
4. Check `requirement.txt` for any version discrepancies if necessary

> [!NOTE]
> You will likely encounter this error:  
   `ImportError: /usr/lib/aarch64-linux-gnu/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found`  
    Fix: in command line, type: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<your_path_to_conda_environment>/envs/meta-pde/lib`  
   `your_path_to_conda_environment` is hinted at the end of your error message  


## Reproducing Experiments
Poisson + MAML is the quickest to run (5-6hrs until full convergence)
Sample
```
python -m src.nn_pde_maml --pde poisson --ground_truth_resolution 32 --siren_omega 30.0 --siren_omega0 30.0 --viz_every 10 --log_every 5 --optimizer adam --outer_lr 1.0e-5 --measure_grad_norm_every 5 --grad_clip 100. --num_layers 3 --layer_size 64 --bc_weight 1.0 --outer_steps 200 --outer_points 1024 --validation_points 1024 --seed 1 --expt_name cpu_maml_pretrain_3_seed_1
```
`pipline` folder contains all the scripts I used for the paper 
