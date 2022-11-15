FROM mambaorg/micromamba
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yaml /tmp/environment.yaml
RUN micromamba install -y -n base -f /tmp/environment.yaml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Required to fix dbus and ssh issues
RUN apt-get update && \
    (apt-get install -f -y dbus || true) && \
    dbus-uuidgen > /var/lib/dbus/machine-id && \
    dpkg --configure -a && \
    apt-get install -y -f && \
    apt-get install -y ca-certificates && \
    update-ca-certificates && \
    apt-get install -y -f software-properties-common && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y build-essential systemd rsync && \
    apt-get install -y wget libcurl4-gnutls-dev && \
    apt-get install -y git && \
    apt-get install -y openssh-server

RUN apt-get install -y -f libcairo2-dev libxt-dev && \
    apt-get install -y -f libxml2-dev libglpk-dev libhdf5-dev && \
    apt-get install -y -f libudunits2-dev libgdal-dev gdal-bin && \
    apt-get install -y -f libproj-dev proj-data proj-bin && \
    apt-get install -y -f libgeos-dev libnlopt-dev cmake tk-dev && \
    apt-get install -y -f libbz2-dev gnutls-dev

RUN wget https://github.com/curl/curl/releases/download/curl-7_83_1/curl-7.83.1.tar.gz && \
    tar -xvf curl-7.83.1.tar.gz && \
    cd curl-7.83.1 && \
    ./configure --with-gnutls && \
    make && \
    make install && \
    cd ..

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

RUN micromamba install -y -c conda-forge openssl=1 llvmlite=0.37.0 clang llvm gcc>=12.1 mkl openssh jupyterlab
EXPOSE 8888
RUN jupyter notebook --allow-root --generate-config

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
RUN conda install -c numba --yes icc_rt
RUN conda install -c conda-forge -y graphviz

# Note there is a mismatch error with conda-forge's h5py so we install it with pip
RUN conda install -c anaconda --yes hdf5=1.10.6
RUN conda remove --force --yes h5py
RUN pip uninstall --yes h5py
RUN conda install -c conda-forge --yes h5py

# Clean temporary files
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/

# Allow for pycharm debugging
EXPOSE 64456
EXPOSE 41502
RUN mkdir /tmp/pycharm_project

ENTRYPOINT ( /usr/sbin/sshd -D & disown ) && jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser --notebook-dir=/ --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.allow_remote_access=True
