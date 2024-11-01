# Install NVIDIA CONTAINER
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_22-04.html#rel_22-04
FROM nvcr.io/nvidia/pytorch:22.04-py3

ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER_NAME=dev_user
ARG REPO_NAME=samba-mixer

ENV CUDA_HOME=/usr/local/cuda

# ADD REQUIREMENTS FILE
ADD requirements.txt .

# INSTALL DEPENDENCIES
RUN apt update -y && \
pip install --no-cache-dir --upgrade pip && \
apt install libsm6 libxext6 -y && \
apt install graphviz -y && \
apt install sudo -y && \
pip install -r requirements.txt && \
groupadd -g ${GROUP_ID} ${USER_NAME} && \
useradd -m -l -u ${USER_ID} -g ${USER_NAME} -G sudo ${USER_NAME} && \
passwd -d ${USER_NAME}
# set empty password for user. password is required for sudo access.

USER ${USER_NAME}

# SET WORKDIR
WORKDIR /home/${USER_NAME}/${REPO_NAME}
