"""
DESCRIPTION:
This script is used to reorganize the format of the yolo detections in the BRNO dataset

"""

import os
import csv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

brno_data_path = '/PATH/TO/DATASET' #TODO: set your path to the downloaded dataset
brno_raw_data_dir = 'brno-winter' 
brno_synched_data_dir = 'brno-synched'

ds_ids = ['3_1_1_1', '3_1_1_2', '3_1_2_1', '3_1_2_2', '3_1_3_1', '3_1_3_2', '3_1_3_3', 
          '3_1_3_4'] #TODO: set your desired subsets
out_dir = f'{brno_data_path}/{brno_synched_data_dir}' 

# TODO: flag for execution (set to False if sorting is already complete)
sort_yolo_files = True

# ---- Helper functions ----
def calculate_area(width, height, img_width, img_height):
  pix_width  = width  * img_width
  pix_height = height * img_height
  return pix_width * pix_height

def process_detections(input_file, output_file, img_width, img_height, lim_dets):
  detections_by_frame = {}
  with open(input_file, "r") as infile:
    reader = csv.reader(infile)
    for row in reader:
      frame_index = int(row[0])
      x, y, width, height, confidence, cls = map(float, row[1:])
      area = calculate_area(width, height, img_width, img_height)
      if frame_index not in detections_by_frame:
        detections_by_frame[frame_index] = []
      detections_by_frame[frame_index].append((frame_index, x, y, width, height, confidence, cls, area))
  # Sort and limit detections for each frame
  sorted_detections = []
  for frame_index in sorted(detections_by_frame.keys()):
    frame_detections = detections_by_frame[frame_index]
    # Sort by area in descending order
    frame_detections.sort(key=lambda det: det[-1], reverse=True)
    # Keep only the top n detections
    top_detections = frame_detections[:lim_dets]
    sorted_detections.extend(top_detections)
  # Write the sorted and limited detections to the output file
  with open(output_file, "w") as outfile:
    writer = csv.writer(outfile)
    for det in sorted_detections:
      writer.writerow(det[:-1])  # Exclude the area column


if sort_yolo_files:
  for dataset_id in ds_ids:
    # Sort RGB detections
    cam_dirs = ['camera_left_front', 'camera_right_front', 'camera_left_side', 'camera_right_side', 'camera_ir']
    cam_heights = [1200, 1200, 1200, 1200, 512]
    cam_widths  = [1920, 1920, 1920, 1920, 640]

    # -- loop each of the 4 cameras --
    for i, img_dir in enumerate(cam_dirs):
      # -- get current camera's yolo detections file --
      yolo_path = (os.path.join(f'{brno_data_path}/{brno_raw_data_dir}/{dataset_id}/yolo',f'{img_dir}.txt'))
      assert os.path.isfile(yolo_path)
      out_ds_dir = f'{out_dir}/{dataset_id}_synched/yolo_sorted'
      os.makedirs(out_ds_dir, exist_ok=True)
      out_path = f'{out_ds_dir}/{img_dir}.txt'
      process_detections(yolo_path, out_path, cam_widths[i], cam_heights[i], 20)
    logging.info(f'Completed format refinement for dataset with ID {dataset_id}')
