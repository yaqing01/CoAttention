# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MS COCO."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import json
import os
import random
import zipfile

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.data_generators import imagenet
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry

import tensorflow as tf

# URLs and filenames for MSCOCO data.
_MSCOCO_ROOT_URL = "http://msvocds.blob.core.windows.net/"
_MSCOCO_URLS = [
    "coco2014/train2014.zip", "coco2014/val2014.zip", "coco2014/test2014.zip",
    "annotations-1-0-3/captions_train-val2014.zip"
]
_MSCOCO_TRAIN_PREFIX = "train2014"
_MSCOCO_EVAL_PREFIX = "val2014"
_MSCOCO_TRAIN_CAPTION_FILE = "annotations/captions_train2014.json"
_MSCOCO_EVAL_CAPTION_FILE = "annotations/captions_val2014.json"


def _get_mscoco(directory):
  """Download and extract MSCOCO datasets to directory unless it is there."""
  for url in _MSCOCO_URLS:
    filename = os.path.basename(url)
    download_url = os.path.join(_MSCOCO_ROOT_URL, url)
    path = generator_utils.maybe_download(directory, filename, download_url)
    unzip_dir = os.path.join(directory, filename.strip(".zip"))
    if not tf.gfile.Exists(unzip_dir):
      zipfile.ZipFile(path, "r").extractall(directory)


def mscoco_generator(data_dir,
                     tmp_dir,
                     training,
                     how_many,
                     start_from=0,
                     eos_list=None,
                     vocab_filename=None):
  """Image generator for MSCOCO captioning problem with token-wise captions.
  Args:
    data_dir: path to the data directory.
    tmp_dir: path to temporary storage directory.
    training: a Boolean; if true, we use the train set, otherwise the test set.
    how_many: how many images and labels to generate.
    start_from: from which image to start.
    eos_list: optional list of end of sentence tokens, otherwise use default
      value `1`.
    vocab_filename: file within `tmp_dir` to read vocabulary from.
  Yields:
    A dictionary representing the images with the following fields:
    * image/encoded: the string encoding the image as JPEG,
    * image/format: the string "jpeg" representing image format,
    * image/class/label: a list of integers representing the caption,
    * image/height: an integer representing the height,
    * image/width: an integer representing the width.
    Every field is actually a list of the corresponding type.
  """
  eos_list = [1] if eos_list is None else eos_list
  def get_vocab():
    """Get vocab for caption text encoder."""
    if data_dir is not None and vocab_filename is not None:
      vocab_filepath = os.path.join(data_dir, vocab_filename)
      if tf.gfile.Exists(vocab_filepath):
        tf.logging.info("Found vocab file: %s", vocab_filepath)
        vocab_symbolizer = text_encoder.SubwordTextEncoder(vocab_filepath)
        return vocab_symbolizer
      else:
        raise ValueError("Vocab file does not exist: %s", vocab_filepath)
    return None

  vocab_symbolizer = get_vocab()
  _get_mscoco(tmp_dir)
  caption_filepath = (
      _MSCOCO_TRAIN_CAPTION_FILE if training else _MSCOCO_EVAL_CAPTION_FILE)
  caption_filepath = os.path.join(tmp_dir, caption_filepath)
  prefix = _MSCOCO_TRAIN_PREFIX if training else _MSCOCO_EVAL_PREFIX
  caption_file = io.open(caption_filepath)
  caption_json = json.load(caption_file)
  # Dictionary from image_id to ((filename, height, width), captions).
  image_dict = dict()
  for image in caption_json["images"]:
    image_dict[image["id"]] = [(image["file_name"], image["height"],
                                image["width"]), []]
  annotations = caption_json["annotations"]
  annotation_count = len(annotations)
  image_count = len(image_dict)
  tf.logging.info("Processing %d images and %d labels\n" % (image_count,
                                                            annotation_count))
  for annotation in annotations:
    image_id = annotation["image_id"]
    image_dict[image_id][1].append(annotation["caption"])

  data = list(image_dict.values())[start_from:start_from + how_many]
  random.shuffle(data)
    
  sample_count = 0
  num_neg = 3
  data_len = len(data)
  max_length = 32

  for image_info, labels in data:
    image_filename = image_info[0]
    image_filepath = os.path.join(tmp_dir, prefix, image_filename)
    with tf.gfile.Open(image_filepath, "r") as f:
      encoded_image_data = f.read()
      height, width = image_info[1], image_info[2]
      for label in labels:
        
        training_pair = []
        if vocab_filename is None or vocab_symbolizer is None:
          label = [ord(c) for c in label] + eos_list
        else:
          label = vocab_symbolizer.encode(label) + eos_list
        training_pair.append([label,1])
                             
        for negIdx in range(num_neg):
            # choose a negative image
          neg_Id = random.randint(0,data_len-2)
          neg_Id = neg_Id if neg_Id<sample_count else neg_Id+1
          neg_image_info, neg_labels = data[neg_Id]
        
          neg_label = random.choice(neg_labels)
          if vocab_filename is None or vocab_symbolizer is None:
            neg_label = [ord(c) for c in neg_label] + eos_list
          else:
            neg_label = vocab_symbolizer.encode(neg_label) + eos_list
          training_pair.append([neg_label,0])
        
        for textlabel,pairlabel in training_pair:
          if len(textlabel)<max_length: # pad
            for padIdx in range(max_length-len(textlabel)):
                textlabel.append(0)
          else:
            textlabel = textlabel[0:max_length] 
          assert len(textlabel)==max_length
          yield {
            "image/encoded": [encoded_image_data],
            "image/format": ["jpeg"],
            "image/class/label": textlabel,
            "image/height": [height],
            "image/width": [width],
            "image/pairlabel": [pairlabel]
           }
    sample_count+=1


@registry.register_problem
class ImageMsCocoMatchCharacters(image_utils.Image2TextProblem):
  """MSCOCO, character level."""

  @property
  def is_character_level(self):
    return True

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_CHR

  @property
  def train_shards(self):
    return 100

  @property
  def dev_shards(self):
    return 10

  def preprocess_example(self, example, mode, _):
    return imagenet.imagenet_preprocess_example(example, mode)

  def generator(self, data_dir, tmp_dir, is_training):
    if is_training:
      return mscoco_generator(data_dir, tmp_dir, True, 80000)
    else:
      return mscoco_generator(data_dir, tmp_dir, False, 40000)
    raise NotImplementedError()


@registry.register_problem
class ImageMsCocoMatchTokens32k(ImageMsCocoMatchCharacters):
  """MSCOCO, 8k tokens vocab."""

  @property
  def is_character_level(self):
    return False

  @property
  def targeted_vocab_size(self):
    return 2**15  # 32768

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def train_shards(self):
    return 100

  @property
  def dev_shards(self):
    return 10

  def generator(self, data_dir, tmp_dir, is_training):
    # We use the translate vocab file as the vocabulary for captions.
    # This requires having the vocab file present in the data_dir for the
    # generation pipeline to succeed.
    vocab_filename = "vocab.ende.%d" % self.targeted_vocab_size
    if is_training:
      return mscoco_generator(
          data_dir,
          tmp_dir,
          True,
          80000,
          vocab_filename=vocab_filename)
    else:
      return mscoco_generator(
          data_dir,
          tmp_dir,
          False,
          40000,
          vocab_filename=vocab_filename)