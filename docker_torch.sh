#!/bin/bash

GLOBAL_HF_CACHE="${HOME}/.cache/huggingface"

docker run --gpus all -it --rm --ipc=host \
  --name torch_ml_rapids \
  --user $(id -u):$(id -g) \
  -v ${GLOBAL_HF_CACHE}:/hf_cache \
  -v ${PWD}:/workspace -w /workspace \
  -e HF_HOME=/hf_cache \
  torch_ml_rapids

#  -p 8501:8501 \
#nvcr.io/nvidia/pytorch:26.03-py3

