#!/bin/bash

GLOBAL_HF_CACHE="${HOME}/.cache/huggingface"

docker run --gpus all -it --rm --ipc=host \
  --name torch_ml_rapids \
  --user $(id -u):$(id -g) \
  -p 8501:8501 \
  -v ${GLOBAL_HF_CACHE}:/hf_cache \
  -v ${PWD}:/workspace -w /workspace \
  -e HF_HOME=/hf_cache \
  torch_ml_rapids

