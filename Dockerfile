# * Build the GPU docker image of PyTorch
#   - CUDA 11.04
#   - Ubuntu 18.04
#   - Python 3.9.12
# * Multi-stage builds to reduce the size

## STAGE 1 ##

FROM continuumio/miniconda3:latest AS compile-image

# Install apt libraries
RUN apt-get update && \
    apt-get install -y curl git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

# Create a conda environment named 'mrp'
RUN conda create -n mrp python=3.9.12 pip
ENV PATH /opt/conda/envs/mrp/bin:$PATH
ENV PIP_ROOT_USER_ACTION=ignore

# Activate the bash shell
RUN chsh -s /bin/bash 
SHELL ["/bin/bash", "-c"]

# Activate the conda environment
# Install Pytorch through the direct torch wheel since CUDA isn't available yet
RUN source activate mrp && \
    python3 -m pip install --no-cache-dir torch==1.13.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# Install other libraries, you can add other libraries here as you want
#RUN pip install --upgrade pip
RUN python3 -m pip install --no-cache-dir emoji==1.6.1 transformers numpy scikit-learn tqdm lime seaborn pandas matplotlib torch_optimizer more-itertools

## STAGE 2 ##

FROM nvidia/cuda:11.4.0-devel-ubuntu18.04 AS build-images
COPY --from=compile-image /opt/conda /opt/conda
ENV PATH /opt/conda/bin:$PATH

# Install apt libraries
RUN apt-get update && \
    apt-get install -y curl git wget vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

RUN echo "source activate mrp" >> ~/.bashrc
CMD ["/bin/bash"]
