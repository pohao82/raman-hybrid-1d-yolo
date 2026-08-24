#!/bin/bash

# first time use
# rename --name
# commit out side of docker env to a new image name
# change  nvcr.io/nvidia/pytorch:26.03-py3  to the new imagine name

GLOBAL_HF_CACHE="${HOME}/.cache/huggingface"

sudo chown -R $(id -u):$(id -g) "$GLOBAL_HF_CACHE"

docker run --gpus all -it --rm --ipc=host \
  --name torch_ml \
  --user $(id -u):$(id -g) \
  -v ${GLOBAL_HF_CACHE}:/hf_cache \
  -v ${PWD}:/workspace -w /workspace \
  -e HF_HOME=/hf_cache \
  nvcr.io/nvidia/pytorch:26.03-py3

