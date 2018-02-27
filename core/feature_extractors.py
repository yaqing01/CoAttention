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
import sys
import math


FLAGS = tf.app.flags.FLAGS

class FeatureExtractor:
    
    def __init__(self, reuse=False):
        self.reuse = reuse

    def __call__(self, image_input, training=False, keep_prob=1.0):
        """
        Runs the CNN producing the embeddings and the gradients.
        :param image_input: Image input to produce embeddings for. [batch_size, image_size, image_size, 1]
        :param training: A flag indicating training or evaluation
        :param keep_prob: A tf placeholder of type tf.float32 indicating the amount of dropout applied
        :return: Embeddings of size [batch_size, 2048]
        """
        weight_decay=FLAGS.weight_decay
        activation_fn=tf.nn.relu 
            
        end_points = {}
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope(weight_decay=FLAGS.weight_decay)):
            with tf.variable_scope("", reuse=self.reuse):
                with tf.variable_scope(None, 'InceptionResnetV2', [image_input]) as scope:
                    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=training):
                        net, end_points = inception.inception_resnet_v2_base(image_input, scope=scope, activation_fn=activation_fn)
                        feature_map = end_points['PreAuxLogits']

                        self.reuse = True

        return feature_map

class FeatureExtractor_inv1:
    
    def __init__(self, reuse=False):
        self.reuse = reuse

    def __call__(self, image_input, training=False, keep_prob=1.0, endpoint_name = 'Mixed_5c'):
        weight_decay=FLAGS.weight_decay
        activation_fn=tf.nn.relu 
            
        end_points = {}
        with slim.arg_scope(inception.inception_v1_arg_scope(weight_decay=FLAGS.weight_decay)):
            with tf.variable_scope("", reuse=self.reuse):
                with tf.variable_scope(None, 'InceptionV1', [image_input]) as scope:
                    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=training):
                        net, end_points = inception.inception_v1_base(image_input, scope=scope)
                        feature_map = end_points[endpoint_name]

                        self.reuse = True

        return feature_map
class FeatureExtractor_nas:
    
    def __init__(self, reuse=False):
        self.reuse = reuse

    def __call__(self, image_input, training=False, keep_prob=1.0):
        """
        Runs the CNN producing the embeddings and the gradients.
        :param image_input: Image input to produce embeddings for. [batch_size, image_size, image_size, 1]
        :param training: A flag indicating training or evaluation
        :param keep_prob: A tf placeholder of type tf.float32 indicating the amount of dropout applied
        :return: Embeddings of size [batch_size, 2048]
        """
        weight_decay=FLAGS.weight_decay
        activation_fn=tf.nn.relu 
            
        end_points = {}

        with tf.variable_scope('NAS', reuse=self.reuse) as scope:
            with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=training):
                with slim.arg_scope(nasnet.nasnet_large_arg_scope(weight_decay=FLAGS.weight_decay)):
                    _, endpoints = nasnet.build_nasnet_large(
                      image_input, num_classes=None,
                      is_training=training,
                      final_endpoint='global_pool')
                feature_map = endpoints['Cell_17']
                print(feature_map.shape)
                self.reuse = True
        return feature_map
