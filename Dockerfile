FROM nvidia/cuda:10.1-base

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# install linux libraries
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion vim && \
    apt-get clean

# install anaconda (python 3.7)
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

# set the default directory in container
WORKDIR /home/root/

# copy contents of this repo in (to the container WORKDIR)
COPY envs/*.yml ./envs/

# Create a conda environment from the environment specification file
RUN conda env create -f ./envs/cat_species_classifier_cpu_env.yml

# get jupyter themes for nicer notebook colours
RUN pip install jupyterthemes

# change themes and adjust cell width
RUN jt -t chesterish -cellw 95%

# add an alias "jn" to launch a jupyter-notebook process on port 8888
RUN echo 'alias jn="jupyter-notebook --ip=0.0.0.0 --no-browser --allow-root --port=8888"' >> ~/.bashrc

# Jupyter notebook
EXPOSE 8888
# Dask scheduler
EXPOSE 8787

CMD [ "/bin/bash" ]
