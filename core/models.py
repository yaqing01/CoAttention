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

from utils.utils import add_timing_signal_nd, Attention
from core.feature_extractors import *
from core.CoAttention import *

# sys.path.insert(0, './deeplab/')
# from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, prepare_label

FLAGS = tf.app.flags.FLAGS

class BaseModel(object):
    """Inherit from this class when implementing new models."""

    def create_model(self, unused_model_input, **unused_params):
        raise NotImplementedError()

    

class KeyValueExtraction:
    
    def __init__(self, reuse=False):
        self.reuse = reuse

    def __call__(self, feature_map, training=True, keep_prob=1.0):
        weight_decay=FLAGS.weight_decay
        activation_fn=tf.nn.relu 
            
        end_points = {}
        with slim.arg_scope(inception.inception_v1_arg_scope(weight_decay=FLAGS.weight_decay)):
            with tf.variable_scope("", reuse=self.reuse):
                with tf.variable_scope(None, 'KeyValueModule', [feature_map]) as scope:
                    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=training):
                        key = slim.conv2d(feature_map, 256, [1, 1], scope='KeyRep')
                        key = tf.reshape(key,[key.get_shape().as_list()[0],-1,key.get_shape().as_list()[3]])
                        
                        value = slim.conv2d(feature_map, 1024, [1, 1], scope='ValueRep')
                        value = tf.reshape(value,[value.get_shape().as_list()[0],-1,value.get_shape().as_list()[3]])

                        self.reuse = True
        return key,value


class MatchNetwork:
    def __init__(self, reuse=False):
        self.reuse = reuse

    def __call__(self, feat_a, feat_b, training=False, spatial_matching=True):
        """
        This module calculates the cosine distance between each of the support set embeddings and the target
        image embeddings.
        :param support_set: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, 64]
        :param input_image: The embedding of the target image, tensor of shape [batch_size, 64]
        :param name: Name of the op to appear on the graph
        :param training: Flag indicating training or evaluation (True/False)
        :return: A tensor with cosine similarities of shape [batch_size, sequence_length, 1]
        """
            
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope(use_batch_norm=True,weight_decay=FLAGS.weight_decay)):
            with tf.variable_scope("", reuse=self.reuse):
                with slim.arg_scope([slim.batch_norm, slim.dropout],
                    is_training=training):
                    eps = 1e-10

                    if spatial_matching:
                        concated_representation = tf.concat([feat_a,feat_b],axis=3)
                        net = slim.conv2d(concated_representation, 1024, [3, 3], scope='Match_1a_3x3')
                        net = slim.conv2d(net, 1024, [3, 3], scope='Match_1b_3x3')
                        net = slim.max_pool2d(net, [3, 3], scope='Match_pool1')
                        net = slim.conv2d(net, 1024, [3, 3], scope='Match_2a_3x3')
                        net = slim.conv2d(net, 1024, [3, 3], scope='Match_2b_3x3')
#                         net = slim.max_pool2d(net, [2, 2], scope='Match_pool2')
                        net = slim.flatten(net)
                    else: 
                        net = tf.concat([feat_a,feat_b],axis=1)                    
#                         aux_logits = slim.dropout(aux_logits, 0.75, scope='Dropout_match',is_training=training)
                    net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='Match_Prelogits1')
                    net = slim.dropout(net, 0.50, scope='Dropout_match',is_training=training)
                    net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='Match_Prelogits2')
                    net = slim.dropout(net, 0.50, scope='Dropout_match',is_training=training)
                    net = slim.fully_connected(net, 2, activation_fn=None, scope='Match_logits')
                    self.reuse=True
        return net

class MatchNetwork_small:
    def __init__(self, reuse=False):
        self.reuse = reuse

    def __call__(self, feat_a, feat_b, training=False, spatial_matching=True):
        """
        This module calculates the cosine distance between each of the support set embeddings and the target
        image embeddings.
        :param support_set: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, 64]
        :param input_image: The embedding of the target image, tensor of shape [batch_size, 64]
        :param name: Name of the op to appear on the graph
        :param training: Flag indicating training or evaluation (True/False)
        :return: A tensor with cosine similarities of shape [batch_size, sequence_length, 1]
        """
            
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope(use_batch_norm=True,weight_decay=FLAGS.weight_decay)):
            with tf.variable_scope("", reuse=self.reuse):
                with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=training):
                    eps = 1e-10

                    if spatial_matching:
                        concated_representation = tf.concat([feat_a,feat_b],axis=3)
                        net = slim.conv2d(concated_representation, 1024, [1, 1], scope='Match_1a_3x3')
                        net = slim.conv2d(net, 1024, [3, 3], scope='Match_1b_3x3')
                        net = slim.flatten(net)
                    else: 
                        net = tf.concat([feat_a,feat_b],axis=1)                    
                    net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='Match_Prelogits2')
                    net = slim.dropout(net, 0.50, scope='Dropout_match',is_training=training)
                    net = slim.fully_connected(net, 2, activation_fn=None, scope='Match_logits')
                    self.reuse=True
        return net
    
class MatchNetwork_nas:
    def __init__(self, reuse=False):
        self.reuse = reuse

    def __call__(self, feat_a, feat_b, training=False, spatial_matching=True):
        """
        This module calculates the cosine distance between each of the support set embeddings and the target
        image embeddings.
        :param support_set: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, 64]
        :param input_image: The embedding of the target image, tensor of shape [batch_size, 64]
        :param name: Name of the op to appear on the graph
        :param training: Flag indicating training or evaluation (True/False)
        :return: A tensor with cosine similarities of shape [batch_size, sequence_length, 1]
        """
            
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope(use_batch_norm=True,weight_decay=FLAGS.weight_decay)):
            with tf.variable_scope("", reuse=self.reuse):
                with slim.arg_scope([slim.batch_norm, slim.dropout],
                    is_training=training):
                    eps = 1e-10

                    if spatial_matching:
                        concated_representation = tf.concat([feat_a,feat_b],axis=3)
                        net = slim.conv2d(concated_representation, 2048, [3, 3], scope='Match_1a_3x3')
                        net = slim.conv2d(net, 2048, [3, 3], scope='Match_1b_3x3')
                        if net.get_shape().as_list()[1]>2 and net.get_shape().as_list()[2]>2:
                            net = slim.max_pool2d(net, [3, 3], scope='Match_pool1')
                        net = slim.conv2d(net, 2048, [3, 3], scope='Match_2a_3x3')
                        net = slim.conv2d(net, 2048, [3, 3], scope='Match_2b_3x3')
#                         net = slim.max_pool2d(net, [2, 2], scope='Match_pool2')
                        net = slim.flatten(net)
                    else: 
                        net = tf.concat([feat_a,feat_b],axis=1)                    
#                         aux_logits = slim.dropout(aux_logits, 0.75, scope='Dropout_match',is_training=training)
                    net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='Match_Prelogits1')
                    net = slim.dropout(net, 0.50, scope='Dropout_match',is_training=training)
                    net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='Match_Prelogits2')
                    net = slim.dropout(net, 0.50, scope='Dropout_match',is_training=training)
                    net = slim.fully_connected(net, 2, activation_fn=None, scope='Match_logits')
                    self.reuse=True
        return net
    
    
class DCSL(BaseModel):
    def create_model(self, input_a, input_b, reuse, is_training=True, **unused_params):
        
        fn_extraction = FeatureExtractor(reuse = reuse)
        fn_match = MatchNetwork(reuse = reuse)
        feat_map_a = fn_extraction(input_a,training=is_training)
        feat_map_b = fn_extraction(input_b,training=is_training)

        score = fn_match(feat_map_a,feat_map_b,training=is_training)
        
        return score
    
class DCSL_inception_v1(BaseModel):
    def create_model(self, input_a, input_b, reuse, is_training=True, **unused_params):
        
        fn_extraction = FeatureExtractor_inv1(reuse = reuse)
        fn_match = MatchNetwork(reuse = reuse)
        feat_map_a = fn_extraction(input_a,training=is_training)
        feat_map_b = fn_extraction(input_b,training=is_training)

        score = fn_match(feat_map_a,feat_map_b,training=is_training)
        
        return score       

class DCSL_NAS(BaseModel):
    def create_model(self, input_a, input_b, reuse, is_training=True, **unused_params):
        
        fn_extraction = FeatureExtractor_nas(reuse = reuse)
        fn_match = MatchNetwork(reuse = reuse)
        feat_map_a = fn_extraction(input_a,training=is_training)
        feat_map_b = fn_extraction(input_b,training=is_training)

        score = fn_match(feat_map_a,feat_map_b,training=is_training)
        
        return score  
    



class MultiHeadAttention:
    def __init__(self, reuse=False):
        self.reuse = reuse

    def __call__(self, Q, K, V, num_heads, training=False, scope = ""):
        """
        This module calculates the cosine distance between each of the support set embeddings and the target
        image embeddings.
        :param support_set: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, 64]
        :param input_image: The embedding of the target image, tensor of shape [batch_size, 64]
        :param name: Name of the op to appear on the graph
        :param training: Flag indicating training or evaluation (True/False)
        :return: A tensor with cosine similarities of shape [batch_size, sequence_length, 1]
        """
        
        d_model = Q.get_shape().as_list()[-1]
        d_key = d_model / num_heads
        d_value = d_model / num_heads
        
        heads = []
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope(use_batch_norm=True,weight_decay=FLAGS.weight_decay)):
            with tf.variable_scope(scope, reuse=self.reuse):
                with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=training):
                    
                    for HeadIdx in range(num_heads):
                        query = slim.conv2d(Q, d_key, [1, 1], scope='QueryRep-'+str(HeadIdx))
                        query = tf.reshape(query,[query.get_shape().as_list()[0],-1,query.get_shape().as_list()[3]])
                        
                        key = slim.conv2d(K, d_key, [1, 1], scope='KeyRep-'+str(HeadIdx))
                        key = tf.reshape(key,[key.get_shape().as_list()[0],-1,key.get_shape().as_list()[3]])

                        value = slim.conv2d(V, d_value, [1, 1], scope='ValueRep-'+str(HeadIdx))
                        value = tf.reshape(value,[value.get_shape().as_list()[0],-1,value.get_shape().as_list()[3]])
                        
                        head = Attention(query,key,value)
                        heads.append(head)

                    heads = tf.concat(heads,axis=2)
#                     heads = slim.fully_connected(heads, d_model) 
        return heads
class MultiHeadAttention_v2:
    def __init__(self, reuse=False):
        self.reuse = reuse

    def __call__(self, Q, K, V, num_heads, training=False, scope = ""):
        """
        This module calculates the cosine distance between each of the support set embeddings and the target
        image embeddings.
        :param support_set: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, 64]
        :param input_image: The embedding of the target image, tensor of shape [batch_size, 64]
        :param name: Name of the op to appear on the graph
        :param training: Flag indicating training or evaluation (True/False)
        :return: A tensor with cosine similarities of shape [batch_size, sequence_length, 1]
        """
        
        d_model = V.get_shape().as_list()[-1]
        d_key = d_model / num_heads
        d_value = d_model / num_heads
        
        batch_norm_params = {
            'decay': 0.9997,
            'epsilon': 0.001,
            'fused': None,  # Use fused batch norm if possible.
        }
        normalizer_fn = slim.batch_norm
        normalizer_params = batch_norm_params

        heads = []
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope(use_batch_norm=True,weight_decay=FLAGS.weight_decay)):
            with tf.variable_scope(scope, reuse=self.reuse):
                with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=training):
                    
                    for HeadIdx in range(num_heads):
                        query = slim.conv2d(Q, d_key, [1, 1], scope='QueryRep-'+str(HeadIdx))
                        query = tf.reshape(query,[query.get_shape().as_list()[0],-1,query.get_shape().as_list()[3]])
                        
                        key = slim.conv2d(K, d_key, [1, 1], scope='KeyRep-'+str(HeadIdx))
                        key = tf.reshape(key,[key.get_shape().as_list()[0],-1,key.get_shape().as_list()[3]])

                        value = slim.conv2d(V, d_value, [1, 1], scope='ValueRep-'+str(HeadIdx))
                        value = tf.reshape(value,[value.get_shape().as_list()[0],-1,value.get_shape().as_list()[3]])
                        
                        head = Attention(query,key,value)
                        heads.append(head)

                    heads = tf.concat(heads,axis=2)
#                     heads = slim.fully_connected(heads, d_model,activation_fn=tf.nn.relu,
#                                                  normalizer_fn=normalizer_fn, normalizer_params=normalizer_params) # use conv2d?
        return heads
    
# class CoAttention(BaseModel):
#     def create_model(self, input_a, input_b, reuse, is_training=True, **unused_params):
        
#         fn_extraction = FeatureExtractor_inv1(reuse = reuse)
#         fn_kv_extraction = KeyValueExtraction(reuse = reuse)
#         fn_match = MatchNetwork(reuse = reuse)
        
#         feat_map_a = fn_extraction(input_a,training=is_training)
#         feat_map_b = fn_extraction(input_b,training=is_training)
        
#         # location embedding
#         feat_map_a_pos_enc = add_timing_signal_nd(feat_map_a)
#         feat_map_b_pos_enc = add_timing_signal_nd(feat_map_b)
        
#         key_a, value_a = fn_kv_extraction(feat_map_a_pos_enc,training=is_training)
#         key_b, value_b = fn_kv_extraction(feat_map_b_pos_enc,training=is_training)
        
#         att_a2b = Attention(key_a,key_b,value_b)
#         att_a2b = tf.reshape(att_a2b, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
#         value_a = tf.reshape(value_a, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        
# #         att_b2a = Attention(key_b,key_a,value_a)
        
#         score = fn_match(att_a2b,value_a,training=is_training)
        
#         return score
    
# class MultiHeadCoAttention(BaseModel):
#     def create_model(self, input_a, input_b, reuse, is_training=True, **unused_params):
        
#         fn_extraction = FeatureExtractor_inv1(reuse = reuse)
#         fn_match = MatchNetwork(reuse = reuse)
#         fn_coattention = MultiHeadAttention(reuse = reuse)
        
#         feat_map_a = fn_extraction(input_a,training=is_training)
#         feat_map_b = fn_extraction(input_b,training=is_training)
        
#         # location embedding
#         feat_map_a_pos_enc = add_timing_signal_nd(feat_map_a)
#         feat_map_b_pos_enc = add_timing_signal_nd(feat_map_b)
        
#         A2AAtt = fn_coattention(feat_map_a_pos_enc,feat_map_a_pos_enc,feat_map_a_pos_enc,FLAGS.num_heads,training=is_training,scope='SelfAtt1')
#         A2BAtt = fn_coattention(feat_map_a_pos_enc,feat_map_b_pos_enc,feat_map_b_pos_enc,FLAGS.num_heads,training=is_training,scope='CoAtt1')
        
#         B2BAtt = fn_coattention(feat_map_b_pos_enc,feat_map_b_pos_enc,feat_map_b_pos_enc,FLAGS.num_heads,training=is_training,scope='SelfAtt2')
#         B2AAtt = fn_coattention(feat_map_b_pos_enc,feat_map_a_pos_enc,feat_map_a_pos_enc,FLAGS.num_heads,training=is_training,scope='CoAtt2')
        
#         A2AAtt = tf.reshape(A2AAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
#         A2BAtt = tf.reshape(A2BAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
#         B2BAtt = tf.reshape(B2BAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
#         B2AAtt = tf.reshape(B2AAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        
#         score = fn_match(A2AAtt,A2BAtt,training=is_training)
        
#         return score 
    
class MultiHeadCoAttention(BaseModel):
    def create_model(self, input_a, input_b, reuse, is_training=True, **unused_params):
        
        fn_extraction = FeatureExtractor_inv1(reuse = reuse)
        fn_match = MatchNetwork(reuse = reuse)
        fn_coattention = MultiHeadAttention(reuse = reuse)
        
        feat_map_a = fn_extraction(input_a,training=is_training)
        feat_map_b = fn_extraction(input_b,training=is_training)
        
        # location embedding
        feat_map_a_pos_enc = add_timing_signal_nd(feat_map_a)
        feat_map_b_pos_enc = add_timing_signal_nd(feat_map_b)
        
        A2AAtt = fn_coattention(feat_map_a_pos_enc,feat_map_a_pos_enc,feat_map_a_pos_enc,FLAGS.num_heads,training=is_training,scope='SelfAtt1')
        A2BAtt = fn_coattention(feat_map_a_pos_enc,feat_map_b_pos_enc,feat_map_b_pos_enc,FLAGS.num_heads,training=is_training,scope='CoAtt1')
        
        B2BAtt = fn_coattention(feat_map_b_pos_enc,feat_map_b_pos_enc,feat_map_b_pos_enc,FLAGS.num_heads,training=is_training,scope='SelfAtt2')
        B2AAtt = fn_coattention(feat_map_b_pos_enc,feat_map_a_pos_enc,feat_map_a_pos_enc,FLAGS.num_heads,training=is_training,scope='CoAtt2')
        
        A2AAtt = tf.reshape(A2AAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        A2BAtt = tf.reshape(A2BAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        
        B2BAtt = tf.reshape(B2BAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        B2AAtt = tf.reshape(B2AAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        
        A = tf.concat([A2AAtt,B2AAtt],axis=3)
        B = tf.concat([B2BAtt,A2BAtt],axis=3)
        
        score = fn_match(A,B,training=is_training)
        
        return score 