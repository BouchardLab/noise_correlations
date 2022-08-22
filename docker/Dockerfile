FROM debian:latest

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /home/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y build-essential gfortran wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

SHELL [ "/bin/bash", "--login", "-c" ]

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /home/conda && \
    rm ~/miniconda.sh && \
    /home/conda/bin/conda clean -tipsy && \
    ln -s /home/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /home/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Initializes conda
ENV PATH=/home/conda/bin:$PATH
# Install
RUN conda update -n base -c defaults conda
RUN conda install python=3.7
RUN conda install numpy scipy scikit-learn h5py pytorch cpuonly -c pytorch -y
RUN python -m pip install pynwb



WORKDIR /tmp

ARG mpich=3.3
ARG mpich_prefix=mpich-$mpich

RUN \
    wget https://www.mpich.org/static/downloads/$mpich/$mpich_prefix.tar.gz && \
    tar xvzf $mpich_prefix.tar.gz                                           && \
    cd $mpich_prefix                                                        && \
    ./configure                                                             && \
    make -j 4                                                               && \
    make install                                                            && \
    make clean                                                              && \
    cd ..                                                                   && \
    rm -rf $mpich_prefix

RUN /sbin/ldconfig

RUN python -m pip install mpi4py

WORKDIR /home

# Clone additional repos
RUN echo
ADD noise_correlations /home/noise_correlations
RUN git clone https://github.com/BouchardLab/mpi_utils.git /home/mpi_utils && git clone https://github.com/BouchardLab/neuropacks.git /home/neuropacks

# Install additional repos (assuming they are properly configured as packages with setup.py files)
RUN python -m pip install -e /home/neuropacks && python -m pip install -e /home/noise_correlations && python -m pip install -e /home/mpi_utils

# Setup entrypoint
ADD entrypoint.sh /home/entrypoint.sh
RUN chmod 777 /home/entrypoint.sh
ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT [ "/usr/bin/tini", "--", "/home/entrypoint.sh" ]
CMD [ "/bin/bash" ]
