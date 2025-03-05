#!/bin/bash

# Exit if a command fails, i.e., if it outputs a non-zero exit status.
set -e

# Load environment
source activate env_addsavi

# Set extra environment paths so that Jax finds libraries from the conda environment
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX

# Run code
python -W ignore -m savi.main --config savi/configs/brno/brno_config.py --workdir ckpt_brno/