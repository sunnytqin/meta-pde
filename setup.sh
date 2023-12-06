# Activate Conda and then source this script: "yes | source setup.sh"
# It takes 10-15 minutes, so then go get a coffee.
conda deactivate
conda remove --name meta-pde --all
conda create -n meta-pde python==3.9
conda activate meta-pde
# conda install -c conda-forge dolfin-adjoint==2019.1.0
conda install -c conda-forge mshr=2019.1.0=*_4
conda install -c conda-forge scipy
conda install -c conda-forge matplotlib
#conda install -c conda-forge jupyter
conda install -c conda-forge tensorflow
conda install cudatoolkit==11.1
conda install cudnn==8
pip install --upgrade pip
pip install --upgrade jax==0.2.20
pip install --upgrade jaxlib==0.1.69+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install flax==0.3.3
pip install git+https://github.com/nestordemeure/flaxOptimizers.git

