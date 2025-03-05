## Description

This workspace is explicit to the creation of a minimalist custom TFDS, with SAVi++ compatibilty, for the Waymo Open Perception Dataset, along with a corresponding SAVi++ dataloader pipeline aimed at leveraging the data to create a corresponding prediction model for road user detection in autonomous driving.


## Data availability

The dataset can be downloaded from https://waymo.com/open/.


## Pre-requisites for execution

- Directory structure for the input data must be similar to the one below w.r.t. the downloaded Waymo Open Perception Dataset:
```bash
waymo_ds_v_1_4_1/
├── training
│   ├── segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord
│   ├── segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord
│   └── ...
└── validation
│   ├── segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord
│   ├── segment-1024360143612057520_3580_000_3600_000_with_camera_labels.tfrecord
│   └── ...
```

- Ensure environment (dependency) requirements are met (provided in the root README of the repo)

- In **tfds_encoder/waymo_video/waymo_video.sh**: 
   - Set paths *manual_dir* and *data_dir*

- Update paths and information denoted by the **TODO** markers within all scripts of the current workspaces


## Execution

1. Generate the custom TFDS dataset required for training and evaluating the SAVi++ prediction model on the Waymo dataset:
   - Execute `waymo_video.sh` within the **waymo_video** sub-directory

2. Organization of the generated TFDS will be similar to what is shown below:
```bash
├── waymo-tfds-video/waymo_video
│   └── video6
│       └── 1.0.0
│           ├── waymo_video-test.tfrecord-00000-of-00256
│           ├── ...
│           ├── waymo_video-test.tfrecord-00255-of-00256
│           ├── waymo_video-train.tfrecord-00000-of-01024
│           ├── ...
│           ├── waymo_video-train.tfrecord-01023-of-01024
│           ├── waymo_video-validation.tfrecord-00000-of-00256
│           ├── ...
│           ├── waymo_video-validation.tfrecord-00255-of-00256
│           ├── dataset_info.json
│           └── features.json
└── downloads
    └── extracted
```

3. Run ADDSAVi++:
   - Execute `run_waymo.sh` within the **model** sub-directory (visualize progress on TensorBoard)