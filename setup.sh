# Activate Conda and then source this script: "yes | source setup.sh"
# It takes 10-15 minutes, so then go get a coffee.
conda deactivate
conda remove --name meta-pde --all
conda create -n meta-pde python==3.9
conda activate meta-pde
conda install -c conda-forge mshr=2019.1.0=py39h9e2e2ce_9
conda install -c conda-forge scipy=1.11.4
conda install -c conda-forge matplotlib
conda install -c conda-forge tensorflow==2.15.0  
conda install cudatoolkit==12
conda install cudnn==8
pip install --upgrade pip
pip install --upgrade jax==0.2.20
# pip install --upgrade jaxlib==0.1.69+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install --upgrade jaxlib==0.1.75 -f https://storage.googleapis.com/jax-releases/jax_releases.htmlpip 
pip install flax==0.3.3

## Trouble shoot
# if you encounter error: ImportError: /usr/lib/aarch64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.30' not found
# do the following in command line
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<your path to conda environment>/envs/meta-pde/lib