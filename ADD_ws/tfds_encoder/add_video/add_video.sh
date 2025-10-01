#!/bin/bash
# Exit if a command fails, i.e., if it outputs a non-zero exit status.
set -e

# Load environment
source activate env_tfds_build #TODO: change command according to your local setup of activating a virtual environment

srun tfds build --manual_dir /PATH/TO/INPUT/DATA --data_dir /PATH/TO/OUTPUT/TFDS