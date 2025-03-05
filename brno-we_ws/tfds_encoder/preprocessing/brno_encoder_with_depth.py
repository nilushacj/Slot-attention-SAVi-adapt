"""
DESCRIPTION:
This script is used to generate image-based TFRecords for BRNO
winter data (including depth features) 

"""
import os
import glob
import tensorflow as tf
import sys
from pypcd import pypcd
import zipfile
import numpy as np
import sys
from datetime import datetime
import torch
from depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

skip_tfrecords = False     #NOTE: set False for creating the tfrecords

brno_data_path = '/PATH/TO/DATASET' #TODO: set your path to the downloaded dataset
brno_raw_data_dir     = 'brno-winter' 
brno_synched_data_dir = 'brno-synched' 
brno_output_tfds = 'brno-winter-tfds-intermediate' #NOTE: path where the TFRecords will be saved

# -- datasets --
ds_ids = ['3_1_1_1', '3_1_1_2', '3_1_2_1', '3_1_2_2', '3_1_3_1', '3_1_3_2', '3_1_3_3', 
          '3_1_3_4'] #TODO: set your desired subsets

# -- camera frames --
camera_frames_dir_folder = f'brno-winter-frames' #NOTE: path to the camera frame data
camera_frames_path = os.path.join(brno_data_path, camera_frames_dir_folder)
os.makedirs(camera_frames_path, exist_ok=True) 

# -- initialize MDE model --
def init_depth_anything_model():
  DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
  model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
  }
  encoder = 'vitl'
  # ---- relative Depth Parameters ----
  model = DepthAnythingV2(**model_configs[encoder])
  model.load_state_dict(torch.load(f'./depth_anything_v2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
  model = model.to(DEVICE).eval()
  # --------------------------------------
  return model
rel_depth_model = init_depth_anything_model()

# -- helper function to read a LiDAR .pcd file from a zipped file and encode it --
def read_lidar_from_zip(zip_file, lidar_filename):
  with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    with zip_ref.open(lidar_filename) as lidar_file:
      pc = pypcd.PointCloud.from_fileobj(lidar_file)
      return pc.pc_data.tobytes()  # convert point cloud to bytes

# -- helper function to generate a float depth matrix from the RGB frame --
def generate_depth_from_rgb(rgb_img):
  rel_depth = rel_depth_model.infer_image(rgb_img) # HxW relative depth map in numpy form
  return rel_depth
    
# -- helper functions for defining TFRecords --
def image_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]))

def bytes_feature(value):
  """Returns a bytes_list from a byte (already encoded)."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_feature_encode(value):
  """Returns a bytes_list from a string."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

def float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def float_feature_list(value):
  """Returns a list of float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def depth_feature(value, min_depth=3.0, max_depth=80.0):
  """
  Encodes a depth tensor of type float as raw bytes for saving in TFRecords.
  Args:
    value: A numpy array or TensorFlow tensor representing the depth image.
  Returns:
    A tf.train.Feature containing the raw bytes.
  """

  # Convert to a tensor 
  depth_tensor = tf.convert_to_tensor(value, dtype=tf.float32)
  
  # Create a mask for valid values within the range
  valid_mask = (depth_tensor >= min_depth) & (depth_tensor <= max_depth)

  # Apply linear scaling to valid values
  scaled_values = ((depth_tensor - min_depth) / (max_depth - min_depth)) * 65535

  # Cast to uint16 and apply the mask
  uint16_depth = tf.where(valid_mask, tf.cast(scaled_values, tf.uint16), tf.zeros_like(scaled_values, dtype=tf.uint16))

  # Encode as PNG with uint16
  value_bytes = tf.io.encode_png(uint16_depth)
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value_bytes.numpy()]))

# -- helper function check if an array is flattened --
def is_flattened(input_list):
  # -- if the input is a numpy array, check its shape --
  if isinstance(input_list, np.ndarray):
    return input_list.ndim == 1
  # -- if it's a regular list, recursively check for any sublist --
  elif isinstance(input_list, list):
    return all(not isinstance(item, (list, np.ndarray)) for item in input_list)
  else:
    raise ValueError("Input should be a list or a NumPy array")

# -- common function to read every text file (assert return value to determine correct no of value are read) --
def file_reader(txt_file):
  with open(txt_file, 'r') as f:
    lines = f.readlines()
  return lines

# -- common function to filter lines from a text file --
def filter_lines_from_file(txt_file_lines, index_to_filter, filter_value, filtered_lines):
  """
  Args
    txt_file_lines: lines read from a text file which will be used for searching
    index_to_filter: the index of the value in each line which will be used for the filtering
    filter_value: value at the selected index for which the lines should be selected
    filtered_lines: variable which will be updated and returned post filtering  
  """
  # -- loop lines --
  for line in txt_file_lines:
    # -- get line values --
    #NOTE: The YOLO detections are in the format: <video frame index, x, y, width, height, detection confidence, class>
    values = line.strip().split(',')
    if int(values[index_to_filter])==filter_value:
      # -- add lines to nested list (i.e. corresponding camera) --
      (filtered_lines).append(values)
  return filtered_lines

# -- helper function to create a TFRecord example --
def create_example( timestamp, index,
                    imgs_rgb, imgs_depth, detections_rgb,
                    img_ir, index_ir, min_temp, max_temp, detections_ir,
                    pc_values,
                    acc_imu, vel_ang_imu, orientation_imu, field_imu, gnss_imu, d_quat_delta_imu, pressure_imu, utc_imu, temp_imu,
                    pose_gnss, utc_gnss,
                    count_radar, detections_radar ):

  cam_lf_depth = np.array(imgs_depth[0], dtype=float)
  cam_rf_depth = np.array(imgs_depth[1], dtype=float)
  cam_ls_depth = np.array(imgs_depth[2], dtype=float)
  cam_rs_depth = np.array(imgs_depth[3], dtype=float)

  cam_lf_dets = np.array(detections_rgb[0], dtype=float).flatten()
  cam_rf_dets = np.array(detections_rgb[1], dtype=float).flatten()
  cam_ls_dets = np.array(detections_rgb[2], dtype=float).flatten()
  cam_rs_dets = np.array(detections_rgb[3], dtype=float).flatten()

  detections_ir = np.array(detections_ir, dtype=float).flatten()
  detections_radar = np.array(detections_radar, dtype=float).flatten()

  # -- Check for flattened arrays --
  arrays_to_check = [
    cam_lf_dets,
    cam_rf_dets,
    cam_ls_dets,
    cam_rs_dets,
    acc_imu,
    vel_ang_imu,
    orientation_imu,
    field_imu,
    gnss_imu,
    d_quat_delta_imu,
    pose_gnss,
    detections_ir,
    detections_radar,
  ]

  for check_index, list_to_check in enumerate(arrays_to_check):
    flat_flag = is_flattened(list_to_check)
    if not flat_flag:
      logging.error(f'Array not flat at {check_index}')
      sys.exit()

  feature = {
    "timestamp": int64_feature(timestamp), # OK
    "index": int64_feature(index), # OK
    "cam_left_front/image" : image_feature(imgs_rgb[0]), # OK #TODO: encode original images if needed instead of the resized ones
    "cam_left_front/depth": depth_feature(cam_lf_depth), #TODO: check if it must be flattened
    "cam_left_front/detections" : float_feature_list(cam_lf_dets), #include DIM
    "cam_right_front/image": image_feature(imgs_rgb[1]), # OK
    "cam_right_front/depth": depth_feature(cam_rf_depth),
    "cam_right_front/detections": float_feature_list(cam_rf_dets), #include DIM
    "cam_left_side/image"  : image_feature(imgs_rgb[2]), # OK 
    "cam_left_side/depth": depth_feature(cam_ls_depth),
    "cam_left_side/detections"  : float_feature_list(cam_ls_dets), #include DIM
    "cam_right_side/image" : image_feature(imgs_rgb[3]), # OK
    "cam_right_side/depth": depth_feature(cam_rs_depth),
    "cam_right_side/detections" : float_feature_list(cam_rs_dets), #include DIM
    "cam_ir/image": image_feature(img_ir), # OK
    "cam_ir/index" : int64_feature(index_ir), # OK
    "cam_ir/min_temp": float_feature(min_temp), # OK
    "cam_ir/max_temp": float_feature(max_temp), # OK
    "cam_ir/detections": float_feature_list(detections_ir), # OK #include DIM
    "lidar_center/pc": bytes_feature(pc_values['lidar_center'][1]), # OK
    "lidar_center/index": int64_feature(pc_values['lidar_center'][0]),   # OK
    "lidar_left/pc": bytes_feature(pc_values['lidar_left'][1]), # OK
    "lidar_left/index": int64_feature(pc_values['lidar_left'][0]), # OK
    "lidar_right/pc": bytes_feature(pc_values['lidar_right'][1]), # OK
    "lidar_right/index": int64_feature(pc_values['lidar_right'][0]), # OK
    "imu/acc": float_feature_list(acc_imu), # OK #include DIM
    "imu/vel_ang": float_feature_list(vel_ang_imu), # OK #include DIM
    "imu/orientation": float_feature_list(orientation_imu), # OK #include DIM
    "imu/mag_field": float_feature_list(field_imu), # OK #include DIM
    "imu/gnss": float_feature_list(gnss_imu), # OK #include DIM
    "imu/d_quat_delta": float_feature_list(d_quat_delta_imu), # OK #include DIM
    "imu/pressure": int64_feature(pressure_imu), # OK
    "imu/utc": bytes_feature_encode(utc_imu), # OK
    "imu/temp": float_feature(temp_imu), # OK
    "gnss/pose": float_feature_list(pose_gnss), # OK #include DIM
    "gnss/utc": bytes_feature_encode(utc_gnss), # OK
    "radar/count": int64_feature(count_radar), # OK
    "radar/detections": float_feature_list(detections_radar) # OK #include DIM
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))


if not skip_tfrecords:

  for dataset_id in ds_ids:

    # -- sensor data to skip encoding --
    sensor_skip_encode = {}
    logging.info(f'=========== Processing dataset {dataset_id} =========== ')
    # -- ensure equal number of all camera images
    img_paths_lf = os.path.join(f'{camera_frames_path}/{dataset_id}','camera_left_front')
    img_paths_lf = sorted(glob.glob(os.path.join(img_paths_lf, '*.jpg')))
    img_paths_rf = os.path.join(f'{camera_frames_path}/{dataset_id}','camera_right_front')
    img_paths_rf = sorted(glob.glob(os.path.join(img_paths_rf, '*.jpg')))
    img_paths_ls = os.path.join(f'{camera_frames_path}/{dataset_id}','camera_left_side')
    img_paths_ls = sorted(glob.glob(os.path.join(img_paths_ls, '*.jpg')))
    img_paths_rs = os.path.join(f'{camera_frames_path}/{dataset_id}','camera_right_side')
    img_paths_rs = sorted(glob.glob(os.path.join(img_paths_rs, '*.jpg')))

    assert len(img_paths_lf)==len(img_paths_rf)==len(img_paths_ls)== len(img_paths_rs)!=0, f'{len(img_paths_lf)}, {len(img_paths_rf)}, {len(img_paths_ls)}, {len(img_paths_rs)} '
    logging.info(f" Number of rgb frames ({dataset_id}): {len(img_paths_lf)} ")
    
    # -- output directory --
    tfrecords_dir = f'{brno_data_path}/{brno_output_tfds}/{dataset_id}_brno_tfrecords'
    os.makedirs(tfrecords_dir, exist_ok=True)
    # -- number of data samples in each tfrecord file --
    num_samples = 199
    # -- total number of tfrecord files to create --
    num_tfrecords = len(img_paths_lf) // num_samples 
    if len(img_paths_lf) % num_samples: # remainder exists
      num_tfrecords += 1  # add one record if there are any remaining samples
    logging.info(f" Number of TFRecords pending creation ({dataset_id}): {num_tfrecords} ")

    # -- list to store tfrecord names --
    tfrecord_files = []
    for count in range(num_tfrecords):
      current_tfrecord_name = f'brno-train.tfrecord-{count:05d}-of-{num_tfrecords:05d}'
      tfrecord_files.append(os.path.join(tfrecords_dir, current_tfrecord_name))
    assert len(tfrecord_files)==num_tfrecords, f'{len(tfrecord_files)}, num_tfrecords'
    
    # ++++ Files: timestamps and index +++
    timestamps_path = f'{brno_data_path}/{brno_raw_data_dir}/{dataset_id}/camera_left_front/timestamps.txt'
    timestamp_lines = file_reader(timestamps_path)

    logging.info(f" Number of timestamps ({dataset_id}): {len(timestamp_lines)} ")

    # ++++ Files: rgb images +++
    rgb_dirs = ['camera_left_front', 'camera_right_front', 'camera_left_side', 'camera_right_side']

    # ++++ Files: laser scans +++
    laser_dirs = ['lidar_center', 'lidar_left', 'lidar_right']

    # -- loop TFrecord file names --
    for tfrec_num, tfrecord_file in enumerate(tfrecord_files):
      start_idx = tfrec_num * num_samples
      end_idx   = (tfrec_num + 1) * num_samples

      if end_idx > len(img_paths_lf):
        end_idx = len(img_paths_lf)

      logging.info(f' ---- Start index ({dataset_id}): {start_idx} ---- ')
      logging.info(f' ---- Final index ({dataset_id}): {end_idx} ---- ')

      # -- write data to TFRecord file --
      with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for tf_frame_index in range(start_idx, end_idx):
          # -- Get base timestamp (i.e. left front camera) for current frame --
          timestamp_line = timestamp_lines[tf_frame_index]


          # ++++ Feature: timestamps and index +++
          parts = timestamp_line.strip().split(',')
          # -- to ensure that the expected line is correct --
          assert len(parts)==3
          frame_timestamp = int(parts[0]) #NOTE: to TFRecord example
          frame_index = int(parts[1]) #NOTE: to TFRecord example


          # ++++ Feature: rgb images and depths +++
          # -- list to store images of the 4 cameras --
          frames_img_decoded = [] #NOTE: to TFRecord example
          # -- list to store yolo detection text files of the 4 cameras --
          all_detection_files = []
          # -- list to store corresponding depth representation --
          frames_depths_decoded = [] #NOTE: to TFRecord example
          # -- loop each of the 4 cameras --
          for img_dir in rgb_dirs:
            # -- get full path of image corresponding to current frame --
            image_path = (os.path.join(f'{camera_frames_path}/{dataset_id}/{img_dir}',f'frame-{frame_index}.jpg'))
            # -- assert if image exists --
            assert os.path.isfile(image_path)
            # -- add decoded image of current camera to list --
            temp_RGB_img = tf.io.decode_jpeg(tf.io.read_file(image_path)) 
            frames_img_decoded.append(temp_RGB_img)
            # -- get current camera's yolo detections file --
            yolo_path = (os.path.join(f'{brno_data_path}/{brno_synched_data_dir}/{dataset_id}_synched/yolo_sorted',f'{img_dir}.txt'))
            # -- assert if file exists --
            assert os.path.isfile(yolo_path)
            # -- add yolo detection file of current camera to list -- 
            all_detection_files.append(yolo_path)
            # -- generate the relavent depth file as a HxW float
            depth_npy = generate_depth_from_rgb(temp_RGB_img.numpy())
            depth_npy = np.expand_dims(depth_npy, axis=-1) 
            frames_depths_decoded.append(depth_npy) 
          # -- assert that each camera has a corresponding image file and yolo detections text file --
          assert len(frames_img_decoded)==len(frames_depths_decoded)==len(all_detection_files)==4


          # ++++ Feature: rgb detections +++ 
          all_detection_lines = [[],[],[],[]] #NOTE: to TFRecord example
          # -- loop the yolo detection file of each of the 4 cameras
          for cam_count, cam_detection_path in enumerate(all_detection_files):
            # -- read yolo detection file of camera --
            cam_detection_lines = file_reader(cam_detection_path)
            # -- call function to filter current frame's detections and add to list --
            all_detection_lines[cam_count] = filter_lines_from_file(cam_detection_lines, 0, frame_index, all_detection_lines[cam_count])
          

          # ++++ Feature: thermal index, min temp and max temp +++
          # -- read synched timestamp file of thermal camera and get current line --
          synched_ir_path = f'{brno_data_path}/{brno_synched_data_dir}/{dataset_id}_synched/camera_ir/timestamps.txt'
          assert os.path.isfile(synched_ir_path)
          synched_ir_lines = file_reader(synched_ir_path) 
          synched_ir_line = synched_ir_lines[frame_index]
          parts = synched_ir_line.strip().split(',')
          # -- to ensure that the expected line is correct --
          assert len(parts)==4
          # -- obtain index and temperature values of current frame from ir --
          frame_ir_index = int(parts[1]) #NOTE: to TFRecord example
          frame_ir_min_temp = float(parts[2]) #NOTE: to TFRecord example
          frame_ir_max_temp = float(parts[3]) #NOTE: to TFRecord example
        

          # ++++ Feature: thermal image +++
          # -- get full path of ir image corresponding to current frame --
          image_path = (os.path.join(f'{camera_frames_path}/{dataset_id}/camera_ir',f'frame-{frame_ir_index}.jpg'))
          # -- assert if ir image exists --
          assert os.path.isfile(image_path)
          # -- get decoded ir image -- 
          thermal_img_decoded = tf.io.decode_jpeg(tf.io.read_file(image_path)) #NOTE: to TFRecord example


          # ++++ Feature: thermal detections +++
          # -- get ir camera's yolo detections file --
          yolo_path = (os.path.join(f'{brno_data_path}/{brno_synched_data_dir}/{dataset_id}_synched/yolo_sorted','camera_ir.txt'))
          # -- assert if file exists --
          assert os.path.isfile(yolo_path)
          # -- read yolo detection file of ir camera --
          ir_detection_lines_all = file_reader(yolo_path)
          # -- call function to filter current frame's detections and add to list --
          ir_detection_lines = filter_lines_from_file(ir_detection_lines_all, 0, frame_ir_index, []) #NOTE: to TFRecord example 


          laser_tf_vals = {} # dict to store index and scan of each laser
          # ++++ Feature: laser index and laser scans +++
          for laser_dir in laser_dirs:
            # -- get path to synched timestamps of current laser and get current line --
            synched_laser_path = f'{brno_data_path}/{brno_synched_data_dir}/{dataset_id}_synched/{laser_dir}/timestamps.txt'
            assert os.path.isfile(synched_ir_path)
            synched_laser_lines = file_reader(synched_laser_path) 
            synched_laser_line = synched_laser_lines[frame_index]
            # -- extract laser frame index --
            parts = synched_laser_line.strip().split(',')
            assert ((len(parts)==2) or (len(parts)==3))
            frame_laser_index = int(parts[1]) #NOTE: to TFRecord example
            # -- read laser scan file of obtained index --
            frame_lidar_data = read_lidar_from_zip(f'{brno_data_path}/{brno_raw_data_dir}/{dataset_id}/{laser_dir}/scans.zip', f'scan{frame_laser_index:06d}.pcd')
            # -- set the two values (index and scan) to the dictionary --
            laser_tf_vals[laser_dir]=(frame_laser_index, frame_lidar_data) #NOTE: to TFRecord example
          assert len(laser_tf_vals.keys())<4


          # ++++ Feature: imu imu +++
          # -- read synched timestamp and get current line --
          synched_path = f'{brno_data_path}/{brno_synched_data_dir}/{dataset_id}_synched/imu/imu.txt'
          assert os.path.isfile(synched_path)
          synched_lines = file_reader(synched_path) 
          synched_line = synched_lines[frame_index]   
          parts = synched_line.strip().split(',')
          # -- to ensure that the expected line is correct --
          assert len(parts)==11
          # -- obtain all values, except timestamp, for current frame --
          frame_imu_acc = np.array(parts[1:4], dtype=float) #NOTE: to TFRecord example 
          frame_imu_ang_vel = np.array(parts[4:7], dtype=float) #NOTE: to TFRecord example 
          frame_imu_orientation = np.array(parts[7:10], dtype=float) #NOTE: to TFRecord example 


          # ++++ Feature: imu magnetic field +++
          # -- read synched timestamp and get current line --
          synched_path = f'{brno_data_path}/{brno_synched_data_dir}/{dataset_id}_synched/imu/mag.txt'
          assert os.path.isfile(synched_path)
          synched_lines = file_reader(synched_path) 
          synched_line = synched_lines[frame_index]   
          parts = synched_line.strip().split(',')
          # -- to ensure that the expected line is correct --
          assert len(parts)==4
          # -- obtain all values, except timestamp, for current frame --
          frame_imu_mag = np.array(parts[1:], dtype=float) #NOTE: to TFRecord example 


          # ++++ Feature: imu gnss +++
          # -- read synched timestamp and get current line --
          synched_path = f'{brno_data_path}/{brno_synched_data_dir}/{dataset_id}_synched/imu/gnss.txt'
          assert os.path.isfile(synched_path)
          synched_lines = file_reader(synched_path) 
          synched_line = synched_lines[frame_index]   
          parts = synched_line.strip().split(',')
          # -- to ensure that the expected line is correct --
          assert len(parts)==4
          # -- obtain all values, except timestamp, for current frame --
          frame_imu_gnss = np.array(parts[1:], dtype=float) #NOTE: to TFRecord example 


          # ++++ Feature: imu d quat +++
          # -- read synched timestamp and get current line --
          synched_path = f'{brno_data_path}/{brno_synched_data_dir}/{dataset_id}_synched/imu/d_quat.txt'
          assert os.path.isfile(synched_path)
          synched_lines = file_reader(synched_path) 
          synched_line = synched_lines[frame_index]   
          parts = synched_line.strip().split(',')
          # -- to ensure that the expected line is correct --
          assert len(parts)==5
          # -- obtain all values, except timestamp, for current frame --
          frame_imu_dquat = np.array(parts[1:], dtype=float) #NOTE: to TFRecord example
          

          # ++++ Feature: imu pressure +++
          # -- read synched timestamp and get current line --
          synched_path = f'{brno_data_path}/{brno_synched_data_dir}/{dataset_id}_synched/imu/pressure.txt'
          assert os.path.isfile(synched_path)
          synched_lines = file_reader(synched_path) 
          synched_line = synched_lines[frame_index]   
          parts = synched_line.strip().split(',')
          # -- to ensure that the expected line is correct --
          assert len(parts)==2
          # -- obtain the pressure for current frame --
          frame_imu_pressure = int(parts[1]) #NOTE: to TFRecord example


          # ++++ Feature: imu UTC time +++
          # -- read synched timestamp and get current line --
          synched_path = f'{brno_data_path}/{brno_synched_data_dir}/{dataset_id}_synched/imu/time.txt'
          assert os.path.isfile(synched_path)
          synched_lines = file_reader(synched_path) 
          synched_line = synched_lines[frame_index]   
          parts = synched_line.strip().split(',')
          # -- to ensure that the expected line is correct --
          assert len(parts)==8
          # -- get imu UTC as string for current frame --
          frame_imu_utc = '_'.join(parts[1:]) #NOTE: to TFRecord example


          # ++++ Feature: imu temp +++
          # -- read synched timestamp and get current line --
          synched_path = f'{brno_data_path}/{brno_synched_data_dir}/{dataset_id}_synched/imu/temp.txt'
          assert os.path.isfile(synched_path)
          synched_lines = file_reader(synched_path) 
          synched_line = synched_lines[frame_index]   
          parts = synched_line.strip().split(',')
          # -- to ensure that the expected line is correct --
          assert len(parts)==2
          # -- get imu temperature for current frame --
          frame_imu_temp = float(parts[1]) #NOTE: to TFRecord example


          # ++++ Feature: gnss pose +++
          # -- read synched timestamp and get current line --
          synched_path = f'{brno_data_path}/{brno_synched_data_dir}/{dataset_id}_synched/gnss/pose.txt'
          assert os.path.isfile(synched_path)
          synched_lines = file_reader(synched_path) 
          synched_line = synched_lines[frame_index]   
          parts = synched_line.strip().split(',')
          # -- to ensure that the expected line is correct --
          assert len(parts)==5
          # -- get the gnss pose for current frame --
          frame_gnss_pose = np.array(parts[1:], dtype=float) #NOTE: to TFRecord example


          # ++++ Feature: gnss time +++
          # -- read synched timestamp and get current line --
          synched_path = f'{brno_data_path}/{brno_synched_data_dir}/{dataset_id}_synched/gnss/time.txt'
          assert os.path.isfile(synched_path)
          synched_lines = file_reader(synched_path) 
          synched_line = synched_lines[frame_index]   
          parts = synched_line.strip().split(',')
          # -- to ensure that the expected line is correct --
          assert len(parts)==8
          # -- get gnss UTC as string for current frame --
          frame_gnss_utc = '_'.join(parts[1:]) #NOTE: to TFRecord example


          # ++++ Feature: radar scans +++
          # -- read synched timestamp and get current line --
          synched_path = f'{brno_data_path}/{brno_synched_data_dir}/{dataset_id}_synched/radar_ti/scans.txt'
          assert os.path.isfile(synched_path)
          synched_lines = file_reader(synched_path) 
          synched_line = synched_lines[frame_index]   
          parts = synched_line.strip().split(',')
          # -- to ensure that the expected line is correct --
          assert len(parts)==(2+(int(parts[1])*4))
          # -- get number of detections of radar for current frame --
          frame_radar_count = int(parts[1]) #NOTE: to TFRecord example
          # -- get flattened list of detections of radar for current frame --
          #NOTE: each detection is in the form x,y,z,vel
          frame_radar_detections = np.array(parts[2:], dtype=float) #.reshape((frame_radar_count,4))



          # **** Create TFRecord example from above features ****
          tfrecord_example = create_example(frame_timestamp, frame_index,
                                            frames_img_decoded, frames_depths_decoded, all_detection_lines,
                                            thermal_img_decoded, frame_ir_index, frame_ir_min_temp, frame_ir_max_temp, ir_detection_lines,
                                            laser_tf_vals,
                                            frame_imu_acc, frame_imu_ang_vel, frame_imu_orientation, frame_imu_mag, frame_imu_gnss, frame_imu_dquat, frame_imu_pressure, frame_imu_utc, frame_imu_temp,
                                            frame_gnss_pose, frame_gnss_utc,
                                            frame_radar_count, frame_radar_detections)

          # write the example to the TFRecord
          writer.write(tfrecord_example.SerializeToString())

          # Get current date and time
          current_time = datetime.now()

          # Format the output for date and time (HH:MM:SS)
          formatted_time = current_time.strftime("%Y-%m-%d_%H:%M:%S")
      
      logging.info(f' COMPLETED WRITING RECORD (({dataset_id})): {tfrec_num} ')
