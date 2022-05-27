FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu18.04

# Set up locale to prevent bugs with encoding
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8
RUN apt-get update || true && apt-get install -y \
    wget curl git screen htop nano build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    python3.8 python3.8-distutils python3.8-dev \
    tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python3.8 get-pip.py
RUN cd /usr/bin \
    && ln -s python3.8 python \
    && ln -s pip3.8 pip

# Define workspace
WORKDIR /workspace
