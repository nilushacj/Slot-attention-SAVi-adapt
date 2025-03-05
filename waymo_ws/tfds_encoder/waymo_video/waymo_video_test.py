"""waymo_video dataset."""

import tensorflow_datasets as tfds
from . import waymo_video


class WaymoVideoTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for waymo_video dataset."""
  # TODO(waymo_video):
  DATASET_CLASS = waymo_video.WaymoVideo
  SPLITS = {
      'train': 3,  # Number of fake train example
      'test': 1,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
  tfds.testing.test_main()
