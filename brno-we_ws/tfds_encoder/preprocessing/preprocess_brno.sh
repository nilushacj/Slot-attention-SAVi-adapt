#!/bin/bash

# Exit if a command fails
set -e

# Load environment
source activate env_addsavi

# Execute preprocessing scripts sequentially
echo "Running synchronization..."
srun python brno_data_synchronization.py && echo "Completed synchronization successfully."

echo "Running detections organization..."
srun python brno_dets_yolo_refine.py && echo "Completed detections organization successfully."

echo "Generating intermediate TFRecords..."
srun python brno_encoder_with_depth.py && echo "Completed generation of intermediate TFRecords."