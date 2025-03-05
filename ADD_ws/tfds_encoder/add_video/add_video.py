import tensorflow_datasets as tfds
import dataclasses
import numpy as np
import tensorflow as tf
import os
import torch
import json

_DESCRIPTION = """
  TODO: add description
"""

_CITATION = """
  TODO: add citation
"""

# TODO: update parameters for the required video length, and detections length (set integer values)
@dataclasses.dataclass
class VideoConfig(tfds.core.BuilderConfig):
  video_length: int = ... # TODO
  detections_length: int = ... # TODO
  shuffle: bool = True

class ADDVideo(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder"""

  # TODO (optional): update the release notes
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
    '1.0.0': 'Initial release.',
  }

  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  Follow instructions in TODO: add instructions
  """

  # TODO: replace "C" with preferred video length. Set multiple configs if needed (comma separated)
  BUILDER_CONFIGS = [
    VideoConfig(name='videoC', description='Video of length C', video_length=C),
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    video_length = self.builder_config.video_length
    detections_length = self.builder_config.detections_length
    # TODO: set feature hierarchy
    batched_features = tfds.features.FeaturesDict({
      ...
    })
    return self.dataset_info_from_configs(
      features=batched_features,
      supervised_keys=None,
      disable_shuffling=not self.builder_config.shuffle,
      homepage='https://github.com/nilushacj/Waymo-Custom-TFDS-dataloader'
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = dl_manager.manual_dir
    # TODO: set train, validation and test splits ~3 lines
    train_files = ...  
    val_files   = ...
    test_files  = ...
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

    # TODO: get index of the data sample by leveraging the format of the TFRecord naming convention
    tf_index = int(...)
    batch_gen = (
      ds
      .map(...) # TODO: add function to deserialize example
      .map(self._pad_detections) # TODO: implement function to zero pad the detections
      .map(self._add_depth) #TODO: implement function to add depths  
      .batch(
        self.builder_config.video_length,
        drop_remainder=True,
      )
      # TODO (optional): add for testing purposes
      #.take(3)
    )
    
    for i, batch in enumerate(batch_gen):
      key = int(f"{tf_index:03d}{i:05d}")
      results = self._dict_to_numpy(batch)
      yield key, results
  
  @classmethod 
  def _dict_to_numpy(cls, d):
    d_np = {}
    for k, v in d.items():
      if isinstance(v, dict):
        d_np[k] = cls._dict_to_numpy(v)
      else:
        # When working with tf.Tensor objects, convert to numpy.
        d_np[k] = v.numpy()
    return d_np
  
  # TODO: implement function to zero pad the detections and set to dictionary
  def _pad_detections(self, d):
    pass
    return d
  
  # TODO: implement function to add depth estimates to the dictionary
  def _add_depth(self, d):
    image_features = ["..."] # TODO: set camera name/s
    # TODO: (replace placeholders below)
    for feature_name in image_features:
      # TODO: implement the _get_depth_py function which returns the depth and depth_range for each camera in image_features
      # NOTE: tf.py_function can be used to execute code outside the graph mode of tensorflow (eager execution)
      d[feature_name]['depth'], d[feature_name]['depth_range'] = tf.py_function(
        func = self._get_depth_py, #NOTE: function to execute outside graph mode
        inp=[d[feature_name]['image']], #NOTE: inputs to the function (image in this case)
        Tout=[tf.uint16, tf.float32] #NOTE: dtypes of the return values (tf.uint16 for depth and tf.float32 for depth range in this case)
      )
    return d
