#!/bin/bash
#SBATCH -A lips -t 72:00:00 -c 16 --gres=gpu:1 --mem=50G --nodes=1 --tasks-per-node=1 --reservation=lips-interactive
export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
nvidia-smi  # check we have access to gpu

#module load rh/devtoolset/8 cudatoolkit/10.1 cudnn7/7.3.0

source /n/fs/pde-opt/activate_conda.sh
# conda activate mo2
conda activate meta-pde

"$@"
