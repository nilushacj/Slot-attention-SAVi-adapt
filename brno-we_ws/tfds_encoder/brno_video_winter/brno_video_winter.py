"""brno_video_winter dataset."""
import dataclasses
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import os
# Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
The custom TensorFlow Dataset (TFDS) created here reformats the BRNO Urban dataset - Winter extension 
to be compatible with TensorFlow’s tf.data API and the SAVi++ object-centric learning model. This 
dataset provides a structured, reusable format for autonomous driving research, making it accessible to 
a wider research community. While optimized for SAVi++, the dataset can be integrated into open-source 
repositories, enabling other researchers to develop and evaluate machine learning models for autonomous 
driving tasks.

The dataset contains data from key sensors commonly utilized in the automotive industry, including:

  - Cameras: Four RGB cameras and a single IR (thermal) camera.
  - Depth estimates: Relative depths for each RGB camera.
  - 3D LiDARs: Data from three distinct LiDAR sensors
  - GNSS and IMU: RTK GNSS receiver with heading estimation, complemented by an IMU.
  - FMCW Radar: Radar data to detect objects in adverse weather and low-visibility scenarios.
  - Yolo detections for all five cameras

All data samples are timestamped, enabling synchronization and future offline data fusion for complex tasks 
such as multi-sensor integration and object tracking. The dataset captures driving scenarios in winter, recorded 
in Brno, Czechia. 
"""

# BibTeX citation
_CITATION = """\
@article{ligocki2022brno,
  title={Brno urban dataset: Winter extension},
  author={Ligocki, Adam and Jelinek, Ales and Zalud, Ludek},
  journal={Data in Brief},
  volume={40},
  pages={107667},
  year={2022},
  publisher={Elsevier}
}
"""

# Dictionary of features corresponding to the existing input features. TODO: update features as desired
_existing_features = tfds.features.FeaturesDict({
  'cam_ir': tfds.features.FeaturesDict({
    'detections': tfds.features.Tensor(shape=(None, 7), dtype=np.float32),
    'image': tfds.features.Scalar(dtype=np.bytes_), 
    'index': np.int64,
    'max_temp': np.float32,
    'min_temp': np.float32,
  }),
  'cam_left_front': tfds.features.FeaturesDict({
    'detections': tfds.features.Tensor(shape=(None, 7), dtype=np.float32),
    'image': tfds.features.Scalar(dtype=np.bytes_),
    'depth': tfds.features.Scalar(dtype=np.bytes_),
  }),
  'cam_left_side': tfds.features.FeaturesDict({
    'detections': tfds.features.Tensor(shape=(None, 7), dtype=np.float32),
    'image': tfds.features.Scalar(dtype=np.bytes_),
    'depth': tfds.features.Scalar(dtype=np.bytes_),
  }),
  'cam_right_front': tfds.features.FeaturesDict({
    'detections': tfds.features.Tensor(shape=(None, 7), dtype=np.float32),
    'image': tfds.features.Scalar(dtype=np.bytes_),
    'depth': tfds.features.Scalar(dtype=np.bytes_),
  }),
  'cam_right_side': tfds.features.FeaturesDict({
    'detections': tfds.features.Tensor(shape=(None, 7), dtype=np.float32),
    'image': tfds.features.Scalar(dtype=np.bytes_),
    'depth': tfds.features.Scalar(dtype=np.bytes_),
  }),
  'gnss': tfds.features.FeaturesDict({
    'pose': tfds.features.Tensor(shape=(4,), dtype=np.float32),
    'utc': tfds.features.Text(),
  }),
  'imu': tfds.features.FeaturesDict({
    'acc': tfds.features.Tensor(shape=(3,), dtype=np.float32),
    'd_quat_delta': tfds.features.Tensor(shape=(4,), dtype=np.float32),
    'gnss': tfds.features.Tensor(shape=(3,), dtype=np.float32),
    'mag_field': tfds.features.Tensor(shape=(3,), dtype=np.float32),
    'orientation': tfds.features.Tensor(shape=(3,), dtype=np.float32),
    'pressure': np.int64,
    'temp': np.float32,
    'utc': tfds.features.Text(),
    'vel_ang': tfds.features.Tensor(shape=(3,), dtype=np.float32),
  }),
  'index': np.int64,
  'lidar_center': tfds.features.FeaturesDict({
    'index': np.int64,
    'pc': tfds.features.Text(),
  }),
  'lidar_left': tfds.features.FeaturesDict({
    'index': np.int64,
    'pc': tfds.features.Text(),
  }),
  'lidar_right': tfds.features.FeaturesDict({
    'index': np.int64,
    'pc': tfds.features.Text(),
  }),
  'radar': tfds.features.FeaturesDict({
    'count': np.int64,
    'detections': tfds.features.Tensor(shape=(None, 4), dtype=np.float32),
  }),
  'timestamp': np.int64,
})

@dataclasses.dataclass
class VideoConfig(tfds.core.BuilderConfig):
  video_length: int = 1
  detections_length: int = 20 #TODO: set value based on input TFRecords
  shuffle: bool = True    

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for brno_video_winter dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
    '1.0.0': 'Initial release.',
  }

  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  Follow instructions of the README file for generating the input TFRecords
  """
  
  # TODO: Edit/add builder configs if preferred video lengths (comma separated)
  BUILDER_CONFIGS = [
    #VideoConfig(name='video1', description='Video of length 1', video_length=1),
    VideoConfig(name='video6', description='Video of length 6', video_length=6),
    #VideoConfig(name='video1-unshuffled', description='Video of length 1 (unshuffled)', video_length=1, shuffle=False),
    #VideoConfig(name='video6-unshuffled', description='Video of length 6 (unshuffled)', video_length=6, shuffle=False),
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""

    video_length = self.builder_config.video_length
    detections_length = self.builder_config.detections_length

    # TODO: ensure compatibility with _existing_features 
    batched_features = tfds.features.FeaturesDict({
      'cam_ir': tfds.features.FeaturesDict({
        'detections': tfds.features.Tensor(shape=(video_length,  detections_length, 7), dtype=np.float32),
        'image': tfds.features.Sequence(tfds.features.Image(shape=(512, 640, 3), dtype=np.uint8), length=video_length),
        'index': tfds.features.Tensor(shape=(video_length,), dtype=np.int64),
        'max_temp': tfds.features.Tensor(shape=(video_length,), dtype=np.float32),
        'min_temp': tfds.features.Tensor(shape=(video_length,), dtype=np.float32),
      }),
      'cam_left_front': tfds.features.FeaturesDict({
        'detections': tfds.features.Tensor(shape=(video_length,  detections_length, 7), dtype=np.float32),
        'image': tfds.features.Sequence(tfds.features.Image(shape=(1200, 1920, 3), dtype=np.uint8), length=video_length),
        'depth': tfds.features.Sequence(tfds.features.Image(shape=(1200, 1920, 1), dtype=np.uint16), length=video_length),
      }),
      'cam_left_side': tfds.features.FeaturesDict({
        'detections': tfds.features.Tensor(shape=(video_length,  detections_length, 7), dtype=np.float32),
        'image': tfds.features.Sequence(tfds.features.Image(shape=(1200, 1920, 3), dtype=np.uint8), length=video_length),
        'depth': tfds.features.Sequence(tfds.features.Image(shape=(1200, 1920, 1), dtype=np.uint16), length=video_length),
      }),
      'cam_right_front': tfds.features.FeaturesDict({
        'detections': tfds.features.Tensor(shape=(video_length,  detections_length, 7), dtype=np.float32),
        'image': tfds.features.Sequence(tfds.features.Image(shape=(1200, 1920, 3), dtype=np.uint8), length=video_length),
        'depth': tfds.features.Sequence(tfds.features.Image(shape=(1200, 1920, 1), dtype=np.uint16), length=video_length),
      }),
      'cam_right_side': tfds.features.FeaturesDict({
        'detections': tfds.features.Tensor(shape=(video_length,  detections_length, 7), dtype=np.float32),
        'image': tfds.features.Sequence(tfds.features.Image(shape=(1200, 1920, 3), dtype=np.uint8), length=video_length),
        'depth': tfds.features.Sequence(tfds.features.Image(shape=(1200, 1920, 1), dtype=np.uint16), length=video_length),
      }),
      'gnss': tfds.features.FeaturesDict({
        'pose': tfds.features.Tensor(shape=(video_length, 4,), dtype=np.float32),
        'utc': tfds.features.Sequence(tfds.features.Text(), length=video_length),      
      }),
      'imu': tfds.features.FeaturesDict({
        'acc': tfds.features.Tensor(shape=(video_length, 3,), dtype=np.float32),
        'd_quat_delta': tfds.features.Tensor(shape=(video_length, 4,), dtype=np.float32),
        'gnss': tfds.features.Tensor(shape=(video_length, 3,), dtype=np.float32),
        'mag_field': tfds.features.Tensor(shape=(video_length, 3,), dtype=np.float32),
        'orientation': tfds.features.Tensor(shape=(video_length, 3,), dtype=np.float32),
        'pressure': tfds.features.Tensor(shape=(video_length,), dtype=np.int64),
        'temp': tfds.features.Tensor(shape=(video_length,), dtype=np.float32),
        'utc': tfds.features.Sequence(tfds.features.Text(), length=video_length),
        'vel_ang': tfds.features.Tensor(shape=(video_length,  3,), dtype=np.float32),
      }),
      'index': tfds.features.Tensor(shape=(video_length,), dtype=np.int64),
      'lidar_center': tfds.features.FeaturesDict({
        'index': tfds.features.Tensor(shape=(video_length,), dtype=np.int64),
        'pc': tfds.features.Sequence(tfds.features.Text(), length=video_length),
      }),
      'lidar_left': tfds.features.FeaturesDict({
        'index': tfds.features.Tensor(shape=(video_length,), dtype=np.int64),
        'pc': tfds.features.Sequence(tfds.features.Text(), length=video_length),
      }),
      'lidar_right': tfds.features.FeaturesDict({
        'index': tfds.features.Tensor(shape=(video_length,), dtype=np.int64),
        'pc': tfds.features.Sequence(tfds.features.Text(), length=video_length),
      }),
      'radar': tfds.features.FeaturesDict({
        'count': tfds.features.Tensor(shape=(video_length,), dtype=np.int64),
        'detections': tfds.features.Tensor(shape=(video_length, detections_length, 4), dtype=np.float32),
      }),
      'timestamp': tfds.features.Tensor(shape=(video_length,), dtype=np.int64),
      })
      
    return self.dataset_info_from_configs(
      features=batched_features,
      # If there's a common (input, target) tuple from the
      # features, specify them here. They'll be used if
      # `as_supervised=True` in `builder.as_dataset`.
      supervised_keys=None,
      disable_shuffling=not self.builder_config.shuffle,
      homepage='https://github.com/nilushacj/Slot-attention-SAVi-adapt'
    )
      
  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    
    path = dl_manager.manual_dir
    #TODO: Update splits (paths and ratios) as desired 
    all_files = sorted(path.glob("*/*tfrecord*")) 
    train_files = all_files[:int(len(all_files) * 0.75)]
    val_files   = all_files[int(len(all_files) * 0.75):int(len(all_files) * 0.90)]
    test_files  = all_files[int(len(all_files) * 0.90):]
    return {
      'train': self._generate_examples(train_files),
      'validation': self._generate_examples(val_files),
      'test': self._generate_examples(test_files),
    }

  def _generate_examples(self, file_list):
    """Yields examples."""
    for current_file in file_list:
      yield from self._process_tfrecord(current_file)

  def _process_tfrecord(self, f):
    ds = tf.data.TFRecordDataset([f])
    experiment = int(os.path.basename(os.path.dirname(f)).strip('_brno_tfrecords').replace("_",""))
    tf_index = int(os.path.basename(f).split('-')[-3])

    batch_gen = (
      ds
      .map(_existing_features.deserialize_example)
      .map(self._pad_detections)
      .map(self._decode_images_depth)
      .batch(
        self.builder_config.video_length,
        drop_remainder=True,
      )
      # TODO (optional): add for testing purposes
      #.take(3)
    )
    
    for i, batch in enumerate(batch_gen):
      key = int(f"{experiment:04d}{tf_index:03d}{i:05d}")
      results = self._dict_to_numpy(batch)
      yield key, results
  
  @classmethod #TODO check what this means
  def _dict_to_numpy(cls, d):
    d_np = {}
    for k, v in d.items():
      if isinstance(v, dict):
        d_np[k] = cls._dict_to_numpy(v)
      else:
        d_np[k] = v.numpy()
    return d_np
  
  def _pad_detections(self, d):
    detections_length = self.builder_config.detections_length
    detection_features = ["cam_ir", "cam_left_front", "cam_left_side", "cam_right_front", "cam_right_side", "radar"]    
    for feature_name in detection_features:
      detections = d[feature_name]['detections']
      paddings = [[0, detections_length - tf.shape(detections)[0]], [0,0]]
      new_detections = tf.pad(detections, paddings)
      d[feature_name]['detections'] = new_detections
    return d

  def _decode_images_depth(self, d):

    image_features = ["cam_ir", "cam_left_front", "cam_left_side", "cam_right_front", "cam_right_side"]
    for feature_name in image_features:
      d[feature_name]['image'] = tf.io.decode_jpeg(d[feature_name]['image'])
      d[feature_name]['depth'] = tf.io.decode_png(d[feature_name]['depth'], dtype=tf.uint16)

    return d