from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf

from tensorflow.contrib import slim
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
from tensorflow.python.client import device_lib
from tensorflow.python.ops import image_ops

import numpy as np
from tensorflow.contrib.rnn import BasicLSTMCell
import math

def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in xrange(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret

def Attention(Q,K,V):
    # Q,K,V
    # Q: query batch_size x query_num x query_dim
    # K: key batch_size x key_num x query_dim
    # V: value batch_size x value_num x value_dim
    scale = tf.sqrt(Q.get_shape().as_list()[2]*1.0)
    dot_product = tf.matmul(Q,K,transpose_b=True)/scale
    softmax_attention = tf.nn.softmax(dot_product)
    attention_value = tf.matmul(softmax_attention,V)

    return attention_value,softmax_attention
def add_timing_signal_nd(x, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase in one of the positional dimensions.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(a+b) and cos(a+b) can be
    experessed in terms of b, sin(a) and cos(a).
    x is a Tensor with n "positional" dimensions, e.g. one dimension for a
    sequence or two dimensions for an image
    We use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels // (n * 2). For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
    x: a Tensor with shape [batch, d1 ... dn, channels]
    min_timescale: a float
    max_timescale: a float
    Returns:
    a Tensor the same shape as x.
    """
    num_dims = len(x.get_shape().as_list()) - 2
    channels = shape_list(x)[-1]
    num_timescales = channels // (num_dims * 2)
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    for dim in xrange(num_dims):
        length = shape_list(x)[dim + 1]
        position = tf.to_float(tf.range(length))
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        prepad = dim * 2 * num_timescales
        postpad = channels - (dim + 1) * 2 * num_timescales
        signal = tf.pad(signal, [[0, 0], [prepad, postpad]])
        for _ in xrange(1 + dim):
            signal = tf.expand_dims(signal, 0)
        for _ in xrange(num_dims - 1 - dim):
            signal = tf.expand_dims(signal, -2)
        x += signal
    return x

def get_model_gradient_multipliers(last_layer_gradient_multiplier):
    """Gets the gradient multipliers.
    The gradient multipliers will adjust the learning rates for model
    variables. For the task of semantic segmentation, the models are
    usually fine-tuned from the models trained on the task of image
    classification. To fine-tune the models, we usually set larger (e.g.,
    10 times larger) learning rate for the parameters of last layer.
    Args:
    last_layers: Scopes of last layers.
    last_layer_gradient_multiplier: The gradient multiplier for last layers.
    Returns:
    The gradient multiplier map with variables as key, and multipliers as value.
    """
    gradient_multipliers = {}

    for var in slim.get_model_variables():
        # Double the learning rate for biases.
        if 'biases' in var.op.name:
            gradient_multipliers[var.op.name] = 2.

        # Use larger learning rate for last layer variables.
        if 'Score' in var.op.name:
            if 'biases' in var.op.name:
                gradient_multipliers[var.op.name] = 2 * last_layer_gradient_multiplier
                print(var.op.name)
            elif 'weights' in var.op.name:
                gradient_multipliers[var.op.name] = last_layer_gradient_multiplier
                print(var.op.name)

    return gradient_multipliers