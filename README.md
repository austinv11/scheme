# SCHEME
(Single-Cell Heterogeneous Expression Matrix Emulation)

To install, create and then activate the environment with conda (or mamba):

```bash 
conda env create -f environment.yml
conda activate scheme_env
```

If already in an environment you can also run
    
```bash
conda env update -f environment.yml
```

*Windows Note* In order to install on windows, you must install jax via pip by manually making an environment, then using one of the builds produced by:  
https://github.com/cloudhan/jax-windows-builder
```bash 
# CPU only
pip install "jax[cpu]" -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver
# GPU
pip install jax[cuda111] -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver

# Then, install numpyro via pip
pip install numpyro

# Then, install the rest of the packages from the environment.yml file. Example:
# Install conda packages
conda install -y -c conda-forge tqdm networkx scanpy
# Install pip packages
pip install treeo louvain leidenalg
```

If you have a GPU available, uninstall jax and reinstall it with gpu support like so:
```bash 
conda install -y --force-reinstall -c conda-forge -c nvidia jax cuda-nvcc
```

To use a ready-made docker image, simply build the docker image as so:
```bash
docker build -t scheme .
```

Then, run the docker image as so:
```bash
docker run -it scheme
```

To enable GPU support just add the `--gpus all` flag to the `docker run` command.

Additionally, this image has several ports exposed that you may want to map.

- 22: SSH Connections (allows for remote interpreters in IDEs)
- 8888: Jupyter Notebook Server
- 64456 and 41502: Ports that allow for remote debugging in PyCharm

The SSH connection by default has a user `root` with password `password`.
The Jupyter Notebook server has no password by default. 
As such, this image is not recommended for production use!

As a final note, when running python you should use the convenience alias of `python-conda`, 
which will automatically ensure that the correct conda environment (`scheme_env`) is activated.