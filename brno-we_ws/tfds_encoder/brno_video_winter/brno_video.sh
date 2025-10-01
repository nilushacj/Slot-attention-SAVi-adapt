#!/bin/bash

# Exit if a command fails
set -e

# Load environment
source activate env_addsavi

srun tfds build --manual_dir /scratch/eng/t212-amlab/brno/brno-winter-tfds-intermediate --data_dir /scratch/eng/t212-amlab/brno/brno-winter-tfds-video #TODO: Change paths