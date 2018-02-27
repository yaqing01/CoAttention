from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append("slim/")
import math
import numpy as np
import tensorflow as tf
import time
from core import models
# Main slim library
from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.data.prefetch_queue import prefetch_queue
import os

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

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_string('set', 'validate', '')
tf.app.flags.DEFINE_string('checkpoints_dir', 'model/trained_models', 'pre-trained models')
tf.app.flags.DEFINE_integer('num_readers', 1, 'Number of threads to preprocess the images.')
tf.app.flags.DEFINE_integer('batch_size', 1, 'Number of batch_size')
tf.app.flags.DEFINE_string('dataset_dir','/data/reid_data_test/validate/', 'directory of saving the training data')
tf.app.flags.DEFINE_integer('num_epoches',20, 'Number of epoches for training')
tf.app.flags.DEFINE_string('train_dir','experiments/dcsl3', 'the experiment name')
tf.app.flags.DEFINE_string('eval_dir','experiments/dcsl3_eval', 'the experiment name')
tf.app.flags.DEFINE_string('eval_weight',None, 'the experiment model for eval once')

tf.app.flags.DEFINE_string('model','DCSL', 'the model name')
tf.app.flags.DEFINE_string('weights', None, 'pre-trained models')

tf.app.flags.DEFINE_integer('target_height', 224, 'the target input image size')
tf.app.flags.DEFINE_integer('target_width', 112, 'the target input image size')

tf.app.flags.DEFINE_integer('num_clones', 1, 'the number of gpus for training')
tf.app.flags.DEFINE_float('weight_decay', 0.00004, 'weight_decay')

tf.app.flags.DEFINE_float('learning_rate',0.01 , 'Initial learning rate.')
tf.app.flags.DEFINE_string('optimizer','momentum', 'optimizer : adam or momentum')
tf.app.flags.DEFINE_string('gpu_id',"0", 'gpu id')
tf.app.flags.DEFINE_float('gpu_memory_fraction', 1.0, 'Initial learning rate.')

tf.app.flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')
tf.app.flags.DEFINE_integer('task', 0, 'task id')
tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                     'Force clones to be deployed on CPU.  Note that even if '
                     'set to False (allowing ops to run on gpu), some ops may '
                     'still be run on the CPU if they have no GPU kernel.')
tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker+trainer '
                     'replicas.')
tf.app.flags.DEFINE_integer('ps_tasks', 0,
                     'Number of parameter server tasks. If None, does not use '
                     'a parameter server.')

tf.app.flags.DEFINE_float('moving_average_decay', None, 'moving_average_decay')
tf.app.flags.DEFINE_integer('num_heads', 8, 'the number of heads')

FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu_id
eval_dir = FLAGS.train_dir + '_eval'

_FILE_PATTERN = '%s-*'

_SPLITS_TO_SIZES = {
    'validate': 2000,
}

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'label': 'The label id of the image, integer between 0 and 999',
}

_NUM_CLASSES = 2

def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading ImageNet.
    Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.
    Returns:
    A `Dataset` namedtuple.
    Raises:
    ValueError: if `split_name` is not a valid train/test split.
    """
    if split_name not in _SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)
    print ('reading dataset from: ' + file_pattern)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
      'image/encoded_a': tf.FixedLenFeature((), tf.string),
      'image/encoded_b': tf.FixedLenFeature((100,), tf.string),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'image/label': tf.FixedLenFeature((), dtype=tf.int64),
    }

    items_to_handlers = {
      'image_a': slim.tfexample_decoder.Image('image/encoded_a', 'image/format'),
      'image_b': slim.tfexample_decoder.Image('image/encoded_b', 'image/format',repeated=True),  
      'label': slim.tfexample_decoder.Tensor('image/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=_SPLITS_TO_SIZES[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=_NUM_CLASSES)

def process_image(image):
#     bbox = tf.constant([0.0, 0.0, 1.0, 1.0],dtype=tf.float32,shape=[1, 1, 4])
#     sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
#         tf.shape(image),
#         bounding_boxes=bbox,
#         min_object_covered=0.1,
#         aspect_ratio_range=(0.80, 1.20),
#         area_range=(0.80, 1.0),
#         max_attempts=100,
#         use_image_if_no_bounding_boxes=True)
#     bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
#     distorted_image = tf.slice(image, bbox_begin, bbox_size)
#     distorted_image.set_shape([None, None, 3])
#     distorted_image = tf.image.random_flip_left_right(distorted_image)

    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [FLAGS.target_height, FLAGS.target_width], align_corners=False)
    image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)  
    return image

def find_class_by_name(name, modules):
    """Searches the provided modules for the named class and returns it."""
    modules = [getattr(module, name, None) for module in modules]
    return next(a for a in modules if a)

def main(_):
    
    with tf.Graph().as_default() as graph:
        tf_global_step = slim.get_or_create_global_step()
        
        
        ######################
        # Select the dataset #
        ######################
        # Train Process
        dataset = get_split('validate',FLAGS.dataset_dir)
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=FLAGS.num_readers,
            common_queue_capacity=FLAGS.batch_size * 20,
            common_queue_min=FLAGS.batch_size * 10)
        [image_a, image_b, label] = provider.get(['image_a','image_b','label'])
        probe = image_a
        galleries = tf.unstack(image_b)
        
        galleries_process = []
        
        probe = process_image(probe)
        probe.set_shape([FLAGS.target_height, FLAGS.target_width, 3])
        
        for gallery in galleries:
            gallery = process_image(gallery)
            gallery.set_shape([FLAGS.target_height, FLAGS.target_width, 3])
            galleries_process.append(gallery)
        
        galleries_process = tf.stack(galleries_process)
                 
        probe_batch, galleries_batch, labels = tf.train.batch(
            [probe, galleries_process, label],
            batch_size=FLAGS.batch_size,
            num_threads=8,
            capacity=FLAGS.batch_size* 10)

        inputs_queue = prefetch_queue([probe_batch, galleries_batch, labels])
            
        ######################
        # Select the network #
        ######################

        probe_batch, galleries_batch, labels = inputs_queue.dequeue()
        probe_batch = tf.squeeze(probe_batch)
        galleries_batch = tf.squeeze(galleries_batch)
        
        model = find_class_by_name(FLAGS.model, [models])()

        probe_batch = tf.expand_dims(probe_batch,0)
        probe_batch = tf.tile(probe_batch,[100,1,1,1])
        logits = model.create_model(probe_batch, galleries_batch, reuse=False, is_training = False) 
        

        if FLAGS.moving_average_decay:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, global_step)
        else:
            moving_average_variables, variable_averages = None, None
        variables_to_restore = slim.get_variables_to_restore()
            
        predictions = tf.argmax(logits, 0)
        predictions = tf.slice(predictions, [1],[1])
        predictions = tf.squeeze(predictions)

        labels = tf.squeeze(labels)
        
        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        })
        
        # Print the summaries to screen.
        for name, value in names_to_values.items():
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
            

        num_samples = _SPLITS_TO_SIZES[FLAGS.set]
        num_batches = math.ceil(num_samples/float(FLAGS.batch_size))    
        
        if FLAGS.eval_weight is not None:
            slim.evaluation.evaluate_once(
                FLAGS.master,
                FLAGS.eval_weight,
                eval_dir,
                num_evals=num_batches,
                eval_op=list(names_to_updates.values()),
                variables_to_restore=variables_to_restore
                )
        else:
            slim.evaluation.evaluation_loop(
                FLAGS.master,
                FLAGS.train_dir,
                eval_dir,
                num_evals=num_batches,
                eval_op=list(names_to_updates.values()),
                eval_interval_secs=600,
                variables_to_restore=variables_to_restore
                )
        
if __name__ == '__main__':
    tf.app.run()