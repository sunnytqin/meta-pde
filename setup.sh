# Activate Conda and then source this script: "yes | source setup.sh"
# It takes 5-10 minutes, so then go get a coffee.
conda deactivate
conda remove --name meta-pde --all
conda create -n meta-pde python==3.8
conda activate meta-pde
# conda install -c conda-forge dolfin-adjoint==2019.1.0
conda install -c conda-forge mshr=2019.1.0=*_4
conda install -c conda-forge numpy scipy matplotlib==3.0.3 jupyter
conda install -c conda-forge tensorflow
conda install cudatoolkit==11.1
conda install cudnn==8
pip install --upgrade pip
pip install --upgrade jax jaxlib==0.1.65+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install flax
pip install git+https://github.com/nestordemeure/flaxOptimizers.git
pip install git+https://github.com/nestordemeure/AdaHessianJax.git

