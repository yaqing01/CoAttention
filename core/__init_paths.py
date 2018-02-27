from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf
import time
import os.path as osp
import sys
import shutil

from tensorflow.contrib import slim
import tqdm
import os
from nets import inception
from preprocessing import inception_preprocessing
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
from tensorflow.python.client import device_lib
from tensorflow.python.ops import image_ops

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from nets.nasnet import nasnet
from nets.nasnet import nasnet_utils
from utils.utils import add_timing_signal_nd, Attention
from core.feature_extractors import *
import sys
import math