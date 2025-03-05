## Description

This workspace provides instructions to create a custom TFDS, with SAVi++ compatibilty, for a selected Autonomous Driving Dataset, along with a corresponding SAVi++ dataloader pipeline aimed at leveraging the data to create a corresponding prediction model for road user detection in autonomous driving.


## Pre-requisites for execution

- Create the template files (details outlined in https://www.tensorflow.org/datasets/add_dataset), e.g. `tfds new add_video`.

- Directory structure for the input data must ideally be as tfrecords (similar to Waymo). Otherwise refer to the BRNO dataset workspace (*brno-we_ws*) instructions for intermediate tfrecord creation OR extract the features within the script and set to the feature dictionary (refer to the instructions in **add_video.py**).

- Ensure environment (dependency) requirements are met (provided in the root README of the repo).

- In **tfds_encoder/add_video/add_video.sh**: 
   - Set paths *manual_dir* and *data_dir*

- Update paths and information denoted by the **TODO** markers within all scripts of the current workspaces.

- Update the **add** prefix with the dataset of choice in the files. Add the code from **add_video.py** and **add_video.sh** into the template files.  


## Execution

1. Generate the custom TFDS dataset required for training and evaluating the SAVi++ prediction model on the dataset:
   - Execute `add_video.sh` within the **add_video** sub-directory

2. Organization of the generated TFDS will be similar to what is shown below:
```bash
├── add-tfds-video/add_video
│   └── video6
│       └── 1.0.0
│           ├── add_video-test.tfrecord-00000-of-00256
│           ├── ...
│           ├── add_video-test.tfrecord-00255-of-00256
│           ├── add_video-train.tfrecord-00000-of-01024
│           ├── ...
│           ├── add_video-train.tfrecord-01023-of-01024
│           ├── add_video-validation.tfrecord-00000-of-00256
│           ├── ...
│           ├── add_video-validation.tfrecord-00255-of-00256
│           ├── dataset_info.json
│           └── features.json
└── downloads
    └── extracted
```

3. Run ADDSAVi++:
   - Execute `run_add.sh` within the **model** sub-directory (visualize progress on TensorBoard)