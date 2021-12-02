#!/bin/bash
CUDA_VERSION=$1
UBUNTU_VERSION=$2
MOUNT_DIR=$3

if [[ "$(docker images -q rapidsai/rapidsai)" == "" ]];
then
    docker pull rapidsai/rapidsai:21.10-$CUDA_VERSION-runtime-$UBUNTU_VERSION-py3.7
else 
    docker run --gpus all --rm -it --ipc=host \
    -p 8888:8888 -p 8787:8787 -p 8786:8786 \
    -v MOUNT_DIR:/mnt/ rapidsai/rapidsai:21.10-$CUDA_VERSION-runtime-$UBUNTU_VERSION-py3.7
fi
