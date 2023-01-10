FROM ubuntu:latest

# Based on https://github.com/mamba-org/micromamba-docker/blob/main/Dockerfile
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

RUN apt-get update

# Fix tzdata install
RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y tzdata && \
    dpkg-reconfigure --frontend noninteractive tzdata
RUN apt-get install -y wget bzip2 ca-certificates curl git openssh-server rsync && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_22.11.1-1-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -f -b -p /opt/conda && \
    rm miniconda.sh

# Update conda and use the libmamba solver to resolve packages faster
RUN /opt/conda/bin/conda update --yes -n base -c defaults conda

RUN /opt/conda/bin/conda install --yes -n base conda-libmamba-solver
RUN /opt/conda/bin/conda config --system --set solver libmamba

# Install dependencies
ADD environment.yml /tmp/environment.yml
RUN /opt/conda/bin/conda env create -f /tmp/environment.yml

RUN /opt/conda/bin/conda init bash
RUN /opt/conda/bin/conda config --env --set always_yes true
RUN echo "CONDA_CHANGEPS1=false conda activate scheme_env" >> /etc/profile

# Install SSH
RUN /opt/conda/bin/conda install -y -n scheme_env -c conda-forge openssh
# SSH stuff
RUN mkdir /var/run/sshd

RUN echo 'root:password' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PermitRootLogin without-password/PermitRootLogin yes/' /etc/ssh/sshd_config

RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
RUN echo "ALL:ALL       :allow" >> /etc/hosts.allow
RUN touch /etc/hosts.deny
RUN sed -i '/^ALL/d' /etc/hosts.deny

RUN /etc/init.d/ssh start

EXPOSE 22

RUN /opt/conda/bin/conda install -y -n scheme_env -c conda-forge openssl=1 llvmlite=0.37.0 clang llvm gcc>=12.1 mkl openssh jupyterlab
EXPOSE 8888
RUN /opt/conda/bin/conda run -n scheme_env jupyter notebook --allow-root --generate-config

RUN echo "c = get_config()" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_root=True" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.password = ''" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_origin = '*'" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_remote_access = True" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.port = 8888" >> ~/.jupyter/jupyter_notebook_config.py

# Speedup for numba operations
RUN /opt/conda/bin/conda install -n scheme_env -c numba -y icc_rt
RUN /opt/conda/bin/conda install -n scheme_env -c conda-forge -y graphviz

# Note there is a mismatch error with conda-forge's h5py so we install it with pip
RUN /opt/conda/bin/conda install -n scheme_env -c anaconda --yes hdf5=1.10.6
RUN /opt/conda/bin/conda remove -n scheme_env --force --yes h5py
RUN /opt/conda/bin/conda run -n scheme_env pip uninstall --yes h5py
RUN /opt/conda/bin/conda install -n scheme_env -c conda-forge --yes h5py scanpy

# GPU Support
RUN /opt/conda/bin/conda install -n scheme_env -y --force-reinstall -c conda-forge -c nvidia jax cuda-nvcc

# Clean temporary files
RUN /opt/conda/bin/conda clean --all --yes
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/

# Hack to force conda to activate scheme_env
# Conda wrapper for python
RUN echo "#!/usr/bin/env bash" > /usr/local/bin/python-conda
RUN echo "/opt/conda/bin/conda run -n scheme_env python3 \$@" >> /usr/local/bin/python-conda
RUN chmod +x /usr/local/bin/python-conda

# Allow for pycharm debugging
EXPOSE 64456
EXPOSE 41502

ENTRYPOINT mkdir -p /tmp/pycharm_project && ( /usr/sbin/sshd -D & disown ) && micromamba run -n scheme_env jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser --notebook-dir=/ --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.allow_remote_access=True
