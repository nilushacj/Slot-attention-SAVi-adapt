import tensorflow_datasets as tfds
import dataclasses
import numpy as np
import tensorflow as tf
import os
import torch
import json
# Import the Waymo Open Dataset protos.
#from waymo_open_dataset.protos import open_dataset_pb2
from tensorflow_datasets.proto import waymo_dataset_pb2 as open_dataset_pb2
from depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2 

_DESCRIPTION = """
  Refer to the Waymo Open Perception page at https://waymo.com/open/
"""

_CITATION = """
  @InProceedings{Sun_2020_CVPR,
    author = {Sun, Pei and Kretzschmar, Henrik and Dotiwalla, Xerxes and Chouard, Aurelien and Patnaik, Vijaysai and Tsui, Paul and Guo, James and Zhou, Yin and Chai, Yuning and Caine, Benjamin and Vasudevan, Vijay and Han, Wei and Ngiam, Jiquan and Zhao, Hang and Timofeev, Aleksei and Ettinger, Scott and Krivokon, Maxim and Gao, Amy and Joshi, Aditya and Zhang, Yu and Shlens, Jonathon and Chen, Zhifeng and Anguelov, Dragomir},
    title = {Scalability in Perception for Autonomous Driving: Waymo Open Dataset},
    booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
  }
"""

_OBJECT_LABELS = [
  "TYPE_VEHICLE",
  "TYPE_PEDESTRIAN",
  "TYPE_CYCLIST",
]
_OBJECT_LABELS_INDEX = {label: idx for idx, label in enumerate(_OBJECT_LABELS)}

WAYMO_TYPE_MAPPING = {
  1: "TYPE_VEHICLE",    
  2: "TYPE_PEDESTRIAN",  
  4: "TYPE_CYCLIST"      
}

# -- initialize MDE model --
def init_depth_anything_model():
  DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
  model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
  }
  encoder = 'vitl'
  # ---- Relative Depth Parameters ----
  model = DepthAnythingV2(**model_configs[encoder])
  model.load_state_dict(torch.load(f'./depth_anything_v2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
  model = model.to(DEVICE).eval()
  # --------------------------------------
  return model

rel_depth_model = init_depth_anything_model()

@dataclasses.dataclass
class VideoConfig(tfds.core.BuilderConfig):
  video_length: int = 1
  detections_length: int = 150
  shuffle: bool = True

class WaymoVideo(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for waymo_video dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
    '1.0.0': 'Initial release.',
  }

  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  Follow instructions of the README file for generating the input TFRecords
  """
    
  BUILDER_CONFIGS = [
    VideoConfig(name='video20', description='Video of length 20', video_length=20),
    #VideoConfig(name='video6-unshuffled', description='Video of length 6 (unshuffled)', video_length=6, shuffle=False),
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    video_length = self.builder_config.video_length
    detections_length = self.builder_config.detections_length
    batched_features = tfds.features.FeaturesDict({
      'camera_FRONT': tfds.features.FeaturesDict({
        "image": tfds.features.Sequence(tfds.features.Image(shape=(1280, 1920, 3), dtype=np.uint8), length=video_length),
        'detections': tfds.features.Tensor(shape=(video_length,  detections_length, 5), dtype=np.float32),
        'depth': tfds.features.Sequence(tfds.features.Image(shape=(1280, 1920, 1), dtype=np.uint16), length=video_length),
        'depth_range': tfds.features.Tensor(shape=(video_length,  2,), dtype=tf.float32),
      })
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
    train_files = list(path.glob("training/*tfrecord*"))  
    val_test_files = list(path.glob("validation/*tfrecord*"))  
    val_files   = val_test_files[:int(len(val_test_files) * 0.50)]
    test_files  = val_test_files[int(len(val_test_files) * 0.50):]
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
    tf_index = int(os.path.basename(f).split('-')[1].split('_')[0])
    # Instead of returning a dictionary from _parse_waymo_frame,
    # we return a tuple (image, labels_json)
    batch_gen = (
      ds
      .map(lambda serialized: tf.py_function(
        func=self._parse_waymo_frame,
        inp=[serialized],
        Tout=(tf.uint8, tf.string)   # Return a tuple instead of a nested dict.
      ))
      # Now convert the tuple into the desired dictionary structure.
      .map(lambda img, labels: {"camera_FRONT": {"image": img, "labels": labels}})
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
  
  def _parse_waymo_frame(self, serialized_example):
    """Custom parser for a Waymo Frame protobuf.
    
    This function:
      - Parses the serialized Waymo Frame.
      - Extracts the front camera image and decodes it.
      - Extracts the front-camera labels and JSON-encodes them.
    """
    # Because this function is called via tf.py_function,
    # serialized_example is a tf.Tensor with dtype=string.
    frame = open_dataset_pb2.Frame()
    frame.ParseFromString(serialized_example.numpy())
    
    # Extract the front camera image.
    front_img = None
    for image in frame.images:
      # Compare with the FRONT enum defined in the proto.
      if image.name == open_dataset_pb2.CameraName.FRONT:
        front_img = image.image
        break
    if front_img is None:
      raise ValueError("No front camera image found in the Waymo Frame.")
    # Decode JPEG image.
    img = tf.io.decode_jpeg(front_img, channels=3)
    
    # Extract front-camera labels.
    labels = []
    for cam_label in frame.camera_labels:
      if cam_label.name == open_dataset_pb2.CameraName.FRONT:
        # Now iterate over each label in the repeated field.
        for label in cam_label.labels:
          # Convert the center-based box to [ymin, xmin, ymax, xmax].
          cx = label.box.center_x
          cy = label.box.center_y
          l = label.box.length
          w = label.box.width
          ymin=(cy - (w / 2)) / img.shape[0]
          ymax=(cy + (w / 2)) / img.shape[0]
          xmin=(cx - (l / 2)) / img.shape[1]
          xmax=(cx + (l / 2)) / img.shape[1]
          bbox = [ymin, ymax, xmin, xmax]
          # Convert the label type to int
          label_type = int(label.type)
          labels.append({'bbox': bbox, 'type': label_type})
        # Once we process the front camera labels, we can break.
        break
    
    # If no labels were found, ensure we still have an empty list.
    if labels is None or len(labels) == 0:
      labels = []

    # JSON-encode the list of label dictionaries.
    labels_json = json.dumps(labels)
    
    # Return a tuple instead of a dictionary.
    return img, labels_json

  def _pad_detections_py(self, labels_json, detections_length):
    """Python function to decode JSON and pad detections.

    Args:
      labels_json: a byte string (numpy bytes) containing the JSON-encoded labels.
      detections_length: an integer for the fixed number of detections.
    
    Returns:
      A NumPy array of shape (detections_length, 5) of type np.float32.
    """
        # If labels_json is not already a bytes object, try to convert it.
    if hasattr(labels_json, 'numpy'):
      labels_json = labels_json.numpy()
    # If it comes in as a numpy array, extract the scalar bytes.
    if isinstance(labels_json, np.ndarray):
      labels_json = labels_json.item()

    # Convert the input byte string to a Python string.
    labels_str = labels_json.decode('utf-8')
    # Load the JSON into a Python list.
    labels = json.loads(labels_str)
    # Create an empty array for detections.
    detections = np.zeros((detections_length, 5), dtype=np.float32)
    # Filter and process the labels: only use those labels whose 'type' is within the valid range.
    valid_labels = [
      label for label in labels
      if label.get('type') in [1, 2, 4]    # based on label.proto file in waymo dataset
    ]
    for i, label in enumerate(valid_labels):
      if i >= detections_length:  # Only keep the first detections_length items.
        break
      # Extract the bounding box and class index.
      ymin, ymax, xmin, xmax = label['bbox']
      object_type = _OBJECT_LABELS_INDEX[WAYMO_TYPE_MAPPING[label['type']]]
      detections[i] = [ymin, ymax, xmin, xmax, float(object_type)]
    return detections

  # In your builder class, update _pad_detections as follows:
  def _pad_detections(self, d):
    """Pads detection labels to a fixed length using a tf.py_function."""
    detections_length = self.builder_config.detections_length
    feature_name = "camera_FRONT"

    # If 'labels' is missing, assign an empty JSON list.
    if 'labels' not in d[feature_name]:
      d[feature_name]['labels'] = tf.constant(b'[]', dtype=tf.string)
    
    # Extract the JSON-encoded labels (a tf.string tensor)
    labels_tensor = d[feature_name]['labels']
    # Wrap the python function; note that tf.py_function expects numpy arrays.
    detections = tf.py_function(
      func=self._pad_detections_py,
      inp=[labels_tensor, detections_length],
      Tout=tf.float32
    )
    # Explicitly set the shape so later operations know what to expect.
    detections.set_shape((detections_length, 5))
    d[feature_name]['detections'] = detections
    del d[feature_name]['labels']

    return d

  """Computes relative depth for the image using the depth-anything model."""
  def _get_depth_py(self, rgb_img):
    rgb_img = rgb_img.numpy()
    # Execute the depth-anything model inference.
    rel_depth = rel_depth_model.infer_image(rgb_img)  # HxW relative depth map in numpy
    # Expand dims to get shape HxWx1.
    rel_depth = np.expand_dims(rel_depth, axis=-1)
    # Convert to a tensor.
    rel_depth_tensor = tf.convert_to_tensor(rel_depth, dtype=tf.float32)
    min_val = tf.reduce_min(rel_depth_tensor)
    max_val = tf.reduce_max(rel_depth_tensor)
    range_val = max_val - min_val
    # Normalize using TensorFlow operations.
    rel_depth_normalized = tf.cond(
      tf.greater(range_val, 0),
      lambda: (rel_depth_tensor - min_val) / range_val,
      lambda: tf.zeros_like(rel_depth_tensor)
    )
    # Scale to uint16 range and cast.
    rel_depth_scaled = rel_depth_normalized * 65535.0
    depth_uint16_tensor = tf.cast(rel_depth_scaled, tf.uint16)
    depth_float_range = tf.stack([min_val, max_val])
    return depth_uint16_tensor, depth_float_range
  
  def _decode_images_depth(self, d):
    image_features = ["camera_FRONT"]
    for feature_name in image_features:
      d[feature_name]['depth'], d[feature_name]['depth_range'] = tf.py_function(
        func = self._get_depth_py,
        inp=[d[feature_name]['image']],
        Tout=[tf.uint16, tf.float32]
      )
    return d
