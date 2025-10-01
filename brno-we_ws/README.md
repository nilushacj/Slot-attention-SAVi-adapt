## Description

This workspace is explicit to the creation of a custom TFDS, also SAVi++ compatibility, for the BRNO Urban Dataset- Winter Extension, along with a corresponding SAVi++ dataloader pipeline aimed at leveraging the data to create a corresponding prediction model for road user detection in autonomous driving.


## Data availability

The dataset can be downloaded from https://github.com/Robotics-BUT/Brno-Urban-Dataset. Session 3 corresponds to the winter extension.


## Pre-requisites for execution

- Directory structure for the input data must be similar to the one below w.r.t. the downloaded BRNO Urban: Winter Extension dataset:
```bash
├── 3_1_1_1
│   ├── calibrations
│   │   └── clone.sh
│   ├── camera_ir
│   │   ├── timestamps.txt
│   │   └── video.mp4
│   │   └── frame-0.jpg
│   │   └── ...
│   │   └── frame-n.jpg
│   ├── camera_left_front
│   │   ├── timestamps.txt
│   │   └── video.mp4
│   │   └── frame-0.jpg
│   │   └── ...
│   │   └── frame-n.jpg
│   ├── camera_left_side
│   │   ├── timestamps.txt
│   │   └── video.mp4
│   │   └── frame-0.jpg
│   │   └── ...
│   │   └── frame-n.jpg
│   ├── camera_right_front
│   │   ├── timestamps.txt
│   │   └── video.mp4
│   │   └── frame-0.jpg
│   │   └── ...
│   │   └── frame-n.jpg
│   ├── camera_right_side
│   │   ├── timestamps.txt
│   │   └── video.mp4
│   │   └── frame-0.jpg
│   │   └── ...
│   │   └── frame-n.jpg
│   ├── gnss
│   │   ├── pose.txt
│   │   └── time.txt
│   ├── imu
│   │   ├── d_quat.txt
│   │   ├── gnss.txt
│   │   ├── imu.txt
│   │   ├── mag.txt
│   │   ├── pressure.txt
│   │   ├── temp.txt
│   │   └── time.txt
│   ├── lidar_center
│   │   ├── scans.zip
│   │   └── timestamps.txt
│   ├── lidar_left
│   │   ├── scans.zip
│   │   └── timestamps.txt
│   ├── lidar_right
│   │   ├── scans.zip
│   │   └── timestamps.txt
│   ├── radar_ti
│   │   └── scans.txt
│   └── yolo
│       ├── camera_ir.txt
│       ├── camera_left_front.txt
│       ├── camera_left_side.txt
│       ├── camera_right_front.txt
│       └── camera_right_side.txt
├── 3_1_1_2
```

- Ensure environment (dependency) requirements are met (provided in the root README of the repo)

- In **tfds_encoder/brno_video_winter/brno_video.sh**: 
   - Set paths *manual_dir* and *data_dir*

- Update paths and information denoted by the **TODO** markers within all scripts of the current workspaces


## Execution

1. Run the preprocessing units (**do this once** i.e. to make read the input data):
   - Execute `preprocess_brno.sh` within the **preprocessing** sub-directory
      - This will synchronize the frames, reformat the 2D detections, and create intermediary TFRecords prior to the final encoding. The intermediary records correspond to the direct TFRecords representation of the raw data.
      - Synchronized information will be stored in the *brno-synched* folder under feature-specific directories. Within it will also be reformatted 2D detections within a *yolo_sorted* folder under each recording subset.
      - The generated intermediate TFRecords, which also comprises the monocular depth estimation for each RGB camera will be stored in the *brno-winter-tfds-intermediate* directory

2. Generate the custom TFDS dataset required for training and evaluating the SAVi++ prediction model on the BRNO dataset:
   - Execute `brno_video.sh` within the **brno_video_winter** sub-directory

3. Organization of the generated TFDS will be similar to what is shown below:
```bash
├── brno_video_winter
│   └── video6
│       └── 1.0.0
│           ├── brno_video_winter-test.tfrecord-00000-of-00256
│           ├── ...
│           ├── brno_video_winter-test.tfrecord-00255-of-00256
│           ├── brno_video_winter-train.tfrecord-00000-of-01024
│           ├── ...
│           ├── brno_video_winter-train.tfrecord-01023-of-01024
│           ├── brno_video_winter-validation.tfrecord-00000-of-00256
│           ├── ...
│           ├── brno_video_winter-validation.tfrecord-00255-of-00256
│           ├── dataset_info.json
│           └── features.json
└── downloads
    └── extracted
```

4. (Optional) Run ADDSAVi++:
   - Execute `run_brno.sh` within the **model** sub-directory (visualize progress on TensorBoard)