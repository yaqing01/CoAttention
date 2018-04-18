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
from utils import variables_helper
from utils import utils
tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_string('set', 'train', '')
tf.app.flags.DEFINE_string('checkpoints_dir', 'model/trained_models', 'pre-trained models')
tf.app.flags.DEFINE_integer('num_readers', 8, 'Number of threads to preprocess the images.')
tf.app.flags.DEFINE_integer('batch_size', 5, 'Number of batch_size')
tf.app.flags.DEFINE_string('dataset_dir','/data/reid_data_train_set_extreme_data/train_shuffle/', 'directory of saving the training data')
tf.app.flags.DEFINE_integer('num_epoches',10, 'Number of epoches for training')
tf.app.flags.DEFINE_string('train_dir','experiments/dcsl3', 'the experiment name')
tf.app.flags.DEFINE_string('model','DCSL', 'the model name')
tf.app.flags.DEFINE_string('weights', None, 'pre-trained models')

tf.app.flags.DEFINE_integer('target_height', 224, 'the target input image size')
tf.app.flags.DEFINE_integer('target_width', 112, 'the target input image size')

tf.app.flags.DEFINE_integer('num_clones', 4, 'the number of gpus for training')
tf.app.flags.DEFINE_integer('num_heads', 8, 'the number of heads')
tf.app.flags.DEFINE_float('weight_decay', 0.00004, 'weight_decay')

tf.app.flags.DEFINE_float('learning_rate',0.01 , 'Initial learning rate.')
tf.app.flags.DEFINE_string('optimizer','momentum', 'optimizer : adam or momentum')

tf.app.flags.DEFINE_float('gpu_memory_fraction', 1.0, 'Initial learning rate.')
tf.app.flags.DEFINE_string('gpu_id',"4,5,6,7", 'gpu id')
tf.app.flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')
tf.app.flags.DEFINE_integer('task', 0, 'task id')
tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                     'Force clones to be deployed on CPU.  Note that even if '
                     'set to False (allowing ops to run on gpu), some ops may '
                     'still be run on the CPU if they have no GPU kernel.')
tf.app.flags.DEFINE_boolean('norm_input', True, 'norm input [-1:1].')
tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker+trainer '
                     'replicas.')
tf.app.flags.DEFINE_integer('ps_tasks', 0,
                     'Number of parameter server tasks. If None, does not use '
                     'a parameter server.')

tf.app.flags.DEFINE_float('moving_average_decay', None, 'moving_average_decay')
flags.DEFINE_float('last_layer_gradient_multiplier', 1,
                   'The gradient multiplier for last layers, which is used to '
                   'boost the gradient of last layers if the value > 1.')

FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu_id

_FILE_PATTERN = '%s-*'

_SPLITS_TO_SIZES = {
    'train': 1252200,
}

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'label': 'The label id of the image, integer between 0 and 999',
}

_NUM_CLASSES = 2

num_samples = _SPLITS_TO_SIZES[FLAGS.set]
num_batches = int(num_samples/FLAGS.batch_size*FLAGS.num_epoches)
logdir = FLAGS.train_dir
num_epoches = FLAGS.num_epoches
    
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
      'image/encoded_b': tf.FixedLenFeature((10,), tf.string),
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
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0],dtype=tf.float32,shape=[1, 1, 4])
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(0.80, 1.20),
        area_range=(0.80, 1.0),
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
    distorted_image = tf.slice(image, bbox_begin, bbox_size)
    distorted_image.set_shape([None, None, 3])
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    image = tf.expand_dims(distorted_image, 0)
    image = tf.image.resize_bilinear(image, [FLAGS.target_height, FLAGS.target_width], align_corners=False)
    image = tf.squeeze(image, [0])
    
    if FLAGS.norm_input:
        image = tf.divide(image, 255)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)        

    return image
def get_variables_to_restore():
    """Returns a function run by the chief worker to warm-start the training."""
#     checkpoint_exclude_scopes=["distance-module", "InceptionV4/Logits"]
    checkpoint_exclude_scopes=["DeInceptionV4"]
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return variables_to_restore

def find_class_by_name(name, modules):
    """Searches the provided modules for the named class and returns it."""
    modules = [getattr(module, name, None) for module in modules]
    return next(a for a in modules if a)

def rotate(tf, points, theta):
    rotation_matrix = tf.pack([tf.cos(theta),-tf.sin(theta), 0, tf.sin(theta), tf.cos(theta), 0, 0,0,1])
    rotation_matrix = tf.reshape(rotation_matrix, (3,3))
    return tf.matmul(points, rotation_matrix)
   

def main(_):

    with tf.Graph().as_default() as graph:
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        global_summaries = set([])

        num_batches_epoch = num_samples//(FLAGS.batch_size*FLAGS.num_clones)
        print(num_batches_epoch)

        #######################
        # Config model_deploy #
        #######################
        config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=FLAGS.clone_on_cpu,
            replica_id=FLAGS.task,
            num_replicas=FLAGS.worker_replicas,
            num_ps_tasks=FLAGS.ps_tasks)

        # Create global_step
        with tf.device(config.variables_device()):
            global_step = slim.create_global_step()
    
        ######################
        # Select the dataset #
        ######################
        with tf.device(config.inputs_device()): 
            # Train Process
            dataset = get_split('train',FLAGS.dataset_dir)
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
        def model_fn(inputs_queue):
            probe_batch, galleries_batch, labels = inputs_queue.dequeue()
            probe_batch_tile = tf.tile(tf.expand_dims(probe_batch,axis=1),[1,10,1,1,1])
            shape = probe_batch_tile.get_shape().as_list()
            probe_batch_reshape = tf.reshape(probe_batch_tile,[-1,shape[2],shape[3],shape[4]])
            galleries_batch_reshape = tf.reshape(galleries_batch,[-1,shape[2],shape[3],shape[4]])
            images_a = probe_batch_reshape
            images_b = galleries_batch_reshape
            
            model = find_class_by_name(FLAGS.model, [models])()

            logits = model.create_model(images_a, images_b, reuse=False, is_training = True) 
            logits = tf.reshape(logits,[FLAGS.batch_size,-1])
            label_onehot = tf.one_hot(labels,10)
            crossentropy_loss = tf.losses.softmax_cross_entropy(onehot_labels=label_onehot,logits=logits)

            tf.summary.histogram('images_a',images_a)
        clones = model_deploy.create_clones(config, model_fn, [inputs_queue])
        first_clone_scope = clones[0].scope

        #################################
        # Configure the moving averages #
        #################################
        if FLAGS.moving_average_decay:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, global_step)
        else:
            moving_average_variables, variable_averages = None, None

            
         #########################################
        # Configure the optimization procedure. #
        #########################################
        with tf.device(config.optimizer_device()):
                  
            learning_rate_step_boundaries = [int(num_batches_epoch*num_epoches*0.50), int(num_batches_epoch*num_epoches*0.65), int(num_batches_epoch*num_epoches*0.80)]
            learning_rate_sequence = [FLAGS.learning_rate]
            learning_rate_sequence += [FLAGS.learning_rate*0.1, FLAGS.learning_rate*0.01, FLAGS.learning_rate*0.001]
            learning_rate = learning_schedules.manual_stepping(
                global_step, learning_rate_step_boundaries,
                learning_rate_sequence)
#             learning_rate = learning_schedules.exponential_decay_with_burnin(global_step,
#                                   FLAGS.learning_rate,num_batches_epoch*num_epoches,0.001/FLAGS.learning_rate,
#                                   burnin_learning_rate=0.01,
#                                   burnin_steps=5000)
            if FLAGS.optimizer=='adam':
                opt = tf.train.AdamOptimizer(learning_rate)   
            if FLAGS.optimizer=='momentum':
                opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
            summaries.add(tf.summary.scalar('learning_rate', learning_rate))
                
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)
        with tf.device(config.optimizer_device()):
            training_optimizer = opt
        

        
        

        # Create ops required to initialize the model from a given checkpoint. TODO!!
        init_fn = None
        if FLAGS.model=='DCSL':
            if FLAGS.weights is None:
                # if not FLAGS.moving_average_decay:
                variables = slim.get_model_variables('InceptionResnetV2')
                init_fn = slim.assign_from_checkpoint_fn(
                    os.path.join(FLAGS.checkpoints_dir, 'inception_resnet_v2.ckpt'),
                    slim.get_model_variables('InceptionResnetV2'))
        if FLAGS.model=='DCSL_inception_v1':
            if FLAGS.weights is None:
                # if not FLAGS.moving_average_decay:
                variables = slim.get_model_variables('InceptionV1')
                init_fn = slim.assign_from_checkpoint_fn(
                    os.path.join(FLAGS.checkpoints_dir, 'inception_v1.ckpt'),
                    slim.get_model_variables('InceptionV1'))
        if FLAGS.model=='DCSL_NAS':
#             if FLAGS.weights is None:
#                 # if not FLAGS.moving_average_decay:
#                 variables = slim.get_model_variables('NAS')
#                 init_fn = slim.assign_from_checkpoint_fn(
#                     os.path.join(FLAGS.checkpoints_dir, 'nasnet-a_large_04_10_2017/model.ckpt'),
#                     slim.get_model_variables('NAS'))
            def restore_map():
                variables_to_restore = {}
                for variable in tf.global_variables():
                    for scope_name in ['NAS']:
                        if variable.op.name.startswith(scope_name):
                            var_name = variable.op.name.replace(scope_name + '/', '')
#                             var_name = variable.op.name
                            variables_to_restore[var_name+'/ExponentialMovingAverage'] = variable
#                             variables_to_restore[var_name] = variable
                return variables_to_restore
            var_map = restore_map()
            # restore_var = [v for v in tf.global_variables() if 'global_step' not in v.name]
            available_var_map = (variables_helper.get_variables_available_in_checkpoint(
                                   var_map, FLAGS.weights))
            init_saver = tf.train.Saver(available_var_map)
            def initializer_fn(sess):
                init_saver.restore(sess, FLAGS.weights)
            init_fn = initializer_fn  
            
        if FLAGS.model=='MultiHeadAttentionBaseModel_set':
            if FLAGS.weights is None:
                # if not FLAGS.moving_average_decay:
                variables = slim.get_model_variables('InceptionV1')
                init_fn = slim.assign_from_checkpoint_fn(
                    os.path.join(FLAGS.checkpoints_dir, 'inception_v1.ckpt'),
                    slim.get_model_variables('InceptionV1'))
            else: 
                restore_var = [v for v in slim.get_model_variables() if 'Score' not in v.name]
                init_fn = slim.assign_from_checkpoint_fn(FLAGS.weights,restore_var)   
        if FLAGS.model=='MultiHeadAttentionBaseModel_set_share':
            if FLAGS.weights is None:
                # if not FLAGS.moving_average_decay:
                variables = slim.get_model_variables('InceptionV1')
                init_fn = slim.assign_from_checkpoint_fn(
                    os.path.join(FLAGS.checkpoints_dir, 'inception_v1.ckpt'),
                    slim.get_model_variables('InceptionV1'))
            else: 
                restore_var = [v for v in slim.get_model_variables() if 'Score' not in v.name]
                init_fn = slim.assign_from_checkpoint_fn(FLAGS.weights,restore_var)   
                
        if FLAGS.model=='MultiHeadAttentionBaseModel_set_share_res50':
            if FLAGS.weights is None:
                # if not FLAGS.moving_average_decay:
                variables = slim.get_model_variables('resnet_v2_50')
                init_fn = slim.assign_from_checkpoint_fn(
                    os.path.join(FLAGS.checkpoints_dir, 'resnet_v2_50.ckpt'),
                    slim.get_model_variables('resnet_v2_50'))   
        if FLAGS.model=='MultiHeadAttentionBaseModel_set_inv3':
            # if not FLAGS.moving_average_decay:
            variables = slim.get_model_variables('InceptionV3')
            init_fn = slim.assign_from_checkpoint_fn(
                os.path.join(FLAGS.checkpoints_dir, 'inception_v3.ckpt'),
                slim.get_model_variables('InceptionV3'))
            
        # compute and update gradients
        with tf.device(config.optimizer_device()):
            if FLAGS.moving_average_decay:
                update_ops.append(variable_averages.apply(moving_average_variables))

            # Variables to train.
            all_trainable = tf.trainable_variables()

            #  and returns a train_tensor and summary_op
            total_loss, grads_and_vars = model_deploy.optimize_clones(clones, training_optimizer, regularization_losses=None, var_list=all_trainable)
            
            grad_mult = utils.get_model_gradient_multipliers(FLAGS.last_layer_gradient_multiplier)
            grads_and_vars = slim.learning.multiply_gradients(grads_and_vars, grad_mult)
            # Optionally clip gradients
            # with tf.name_scope('clip_grads'):
            #     grads_and_vars = slim.learning.clip_gradient_norms(grads_and_vars, 10)

            total_loss = tf.check_numerics(total_loss, 'LossTensor is inf or nan.')

            # Create gradient updates.
            grad_updates = training_optimizer.apply_gradients(grads_and_vars,global_step=global_step)
            update_ops.append(grad_updates)

            update_op = tf.group(*update_ops)
            with tf.control_dependencies([update_op]):
                train_tensor = tf.identity(total_loss, name='train_op')   

        # Add summaries.
        for loss_tensor in tf.losses.get_losses():
            global_summaries.add(tf.summary.scalar(loss_tensor.op.name, loss_tensor))
        global_summaries.add(tf.summary.scalar('TotalLoss', tf.losses.get_total_loss()))

        # Add the summaries from the first clone. These contain the summaries
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))
        summaries |= global_summaries
        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        # GPU settings
        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = False
        # Save checkpoints regularly.
        keep_checkpoint_every_n_hours = 2.0

        saver = tf.train.Saver(keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

        ###########################
        # Kicks off the training. #
        ###########################
        slim.learning.train(
            train_tensor,
            logdir=logdir,
            master=FLAGS.master,
            is_chief=(FLAGS.task == 0),
            session_config=session_config,
            startup_delay_steps=10,
            summary_op=summary_op,
            init_fn=init_fn,
            number_of_steps=num_batches_epoch*FLAGS.num_epoches,
            save_summaries_secs=240,
            sync_optimizer=None,
            saver=saver)  

                    
if __name__ == '__main__':
    tf.app.run()
