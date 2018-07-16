from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import time
# Main slim library
from tensorflow.contrib import slim
import random
from datetime import datetime
import os
import sys
import threading
import random

import os
import sys

import glob

# import cv2
import numpy as np
from matplotlib import pyplot as plt

import random

#import png
import itertools

# from skimage.io import imread,imsave
import shutil
import getopt
from multiprocessing import Pool 
import time

sys.path.append("../slim/")
sys.path.append("..")

from eval_cuhk03 import *

import math
import numpy as np
import tensorflow as tf
import time
from core import models
# Main slim library
from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.data.prefetch_queue import prefetch_queue

from nets import inception
from preprocessing import inception_preprocessing
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
from deployment import model_deploy
import json
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import weights_broadcast_ops
from utils import learning_schedules
import tqdm

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_integer('target_width', 112, 'target width of images')
tf.app.flags.DEFINE_integer('target_height', 224, 'target height of images')
tf.app.flags.DEFINE_string('phase', 'train', 'Training data directory')
tf.app.flags.DEFINE_integer('num_threads', 8, 'Number of threads to preprocess the images.')
tf.app.flags.DEFINE_boolean('resize_to_ratio', False, 'if True, resize and keep aspedct ratio')
tf.app.flags.DEFINE_integer('batch_size', 100, 'Number of batch_size')
tf.app.flags.DEFINE_string('weights', None, 'pre-trained models')
tf.app.flags.DEFINE_string('model','MultiHeadAttentionBaseModel_set_share', 'the model name')
tf.app.flags.DEFINE_string('gpu_id',"4", 'gpu id')
tf.app.flags.DEFINE_integer('num_heads', 4, 'the number of heads')
tf.app.flags.DEFINE_float('weight_decay', 0.0002, 'weight_decay')
tf.app.flags.DEFINE_integer('rand_times', 10, 'random times for testing')
tf.app.flags.DEFINE_integer('set_no', 1, 'random times for testing')
FLAGS = tf.app.flags.FLAGS

remaining_args = FLAGS([sys.argv[0]] + [flag for flag in sys.argv if flag.startswith("--")])
assert(remaining_args == [sys.argv[0]])
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu_id

HNM = 0


set_no =FLAGS.set_no
    
# sess = tf.Session("", graph=graph)
# saver.restore(sess,'../experiments_set/coattention_set_scratch_30epoch/model.ckpt-876881')

cmc=calCMC(set_no,rand_times=FLAGS.rand_times)
print(cmc)