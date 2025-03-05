"""
DESCRIPTION:
This script is used to synchronize frame timesteps of the BRNO dataset's sensors
"""

import os
import glob
import numpy as np
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---- Helper functions ----
def get_all_timestamps(dataset_id):
  data_dir = f'{brno_data_path}/{brno_raw_data_dir}/{dataset_id}/{key}'
  ts_paths = glob.glob(os.path.join(data_dir,'*.txt'))
  return ts_paths

def get_some_timestamps(dataset_id, skip_list):
  data_dir = f'{brno_data_path}/{brno_raw_data_dir}/{dataset_id}/{key}'
  ts_paths = glob.glob(os.path.join(data_dir,'*.txt'))
  ts_paths_filtered = []
  for filepath in ts_paths:
    filename = os.path.basename(filepath)
    if filename not in skip_list:
      ts_paths_filtered.append(filepath)
  return ts_paths_filtered

def read_timestamps(file_path):
  with open(file_path, 'r') as file:
    lines = file.readlines()
  timestamps = [int(line.split(',')[0]) for line in lines]
  return timestamps, lines

# ---- Dataset ---
brno_data_path = '/PATH/TO/DATASET' #TODO: set your path to the downloaded dataset
brno_raw_data_dir = 'brno-winter'
brno_synched_data_dir = 'brno-synched'

ds_ids = ['3_1_1_1', '3_1_1_2', '3_1_2_1', '3_1_2_2', '3_1_3_1', '3_1_3_2', '3_1_3_3', 
          '3_1_3_4'] #TODO: set your desired subsets

for ds in ds_ids:
  # ---- Path to synched dataset ----
  synched_ds = f'{brno_data_path}/{brno_synched_data_dir}/{ds}_synched'

  # ---- Remove if synched dataset already exists ----
  if os.path.exists(synched_ds):
    logging.info('Path already exists and therefore removed')
    shutil.rmtree(synched_ds)

  # ---- List of sensor (strings and list of skip entries) data from which the timestamps must be synchronized ----
  sensor_info_dict = {
    'camera_ir': ('video_ir',[]),
    'camera_left_front': ('video', []),
    'camera_left_side': ('video', []),
    'camera_right_front': ('video', []),
    'camera_right_side': ('video', []),
    'gnss':('gnss',[]),
    'imu': ('imu', []),
    'lidar_center': ('lidar',[]),
    'lidar_left': ('lidar',[]),
    'lidar_right': ('lidar',[]),
    'radar_ti': ('radar', [])
  }

  # ---- Assert lengths of camera video frames ----
  base_key = 'camera_left_front'
  base_timestamp = f'{brno_data_path}/{brno_raw_data_dir}/{ds}/{base_key}/timestamps.txt'
  with open(base_timestamp, 'r') as f:
    lines = f.readlines()
  base_ts_count = len(lines)
  for key, value in sensor_info_dict.items():
    if value=='video':
      comp_timestamp = f'{brno_data_path}/{brno_raw_data_dir}/{ds}/{key}/timestamps.txt'
      with open(comp_timestamp, 'r') as f:
        lines = f.readlines()
        assert len(lines)==base_ts_count

  to_synch_dict = {}
  for key, value in sensor_info_dict.items():
    if (not value[0] == 'video'): # NOTE: video timestamps are already synched
      if not value[1]:
        # -- get all timestamps --
        ts_paths = get_all_timestamps(ds, key)
      else:
        # -- get the filtered timestamps --
        ts_paths = get_some_timestamps(ds, key, value[1])
      to_synch_dict[key] = (value[0], ts_paths)  

  # ---- Determine the common time range ----
  base_timestamps, base_lines = read_timestamps(base_timestamp)
  # ---- Loop each of the timestamp files which must be synchronized with the base key ----
  for key, value in to_synch_dict.items():
    # ---- Loop each of the timestamp files under this sensor ----
    for timestamp_file in value[1]:
      # -- Create new file to add the synchronized sensor data --
      synched_file_dir = os.path.join(synched_ds, key) 
      os.makedirs(synched_file_dir, exist_ok=True)
      synched_file_name = os.path.basename(timestamp_file)
      synched_file_path = os.path.join(synched_file_dir, synched_file_name)
      # -- Read the original sensor data --
      current_timestamps, current_lines = read_timestamps(timestamp_file)
      current_timestamps = np.array(current_timestamps)
      # -- Loop each timestamp of the base sensor --
      for i, base_ts in enumerate(base_timestamps):
        # -- Find the timestamp index with the closest time for the current base timestamp --
        nearest_frame_idx = np.argmin(np.abs(current_timestamps - base_ts))
        # -- Write new line to the synched data file --
        with open(synched_file_path, 'a') as file:
          new_line = current_lines[nearest_frame_idx]
          file.write(f'{new_line}')
  logging.info(f'Completed timestamp sychronization for dataset with ID {ds}')



