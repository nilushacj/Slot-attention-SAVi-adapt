#!/bin/bash
# Exit if a command fails, i.e., if it outputs a non-zero exit status.
set -e

# Load environment
source activate env_addsavi

srun tfds build --manual_dir /scratch/eng/t212-amlab/waymo/waymo_ds_v_1_4_1 --data_dir /scratch/eng/t212-amlab/waymo/waymo-tfds-video #TODO: Change paths