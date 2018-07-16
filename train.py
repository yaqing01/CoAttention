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
tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_string('set', 'train', '')
tf.app.flags.DEFINE_string('checkpoints_dir', 'model/trained_models', 'pre-trained models')
tf.app.flags.DEFINE_integer('num_readers', 8, 'Number of threads to preprocess the images.')
tf.app.flags.DEFINE_integer('batch_size', 50, 'Number of batch_size')
tf.app.flags.DEFINE_string('dataset_dir','/data/reid_data_large/train/', 'directory of saving the training data')
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
tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker+trainer '
                     'replicas.')
tf.app.flags.DEFINE_integer('ps_tasks', 0,
                     'Number of parameter server tasks. If None, does not use '
                     'a parameter server.')
tf.app.flags.DEFINE_boolean('norm_input', True, 'norm input [-1:1].')
tf.app.flags.DEFINE_boolean('random_erase', False, 'random_erase input')
tf.app.flags.DEFINE_boolean('random_crop', False, 'random_crop')
tf.app.flags.DEFINE_boolean('random_rotate', False, 'random_rotate')
tf.app.flags.DEFINE_boolean('random_flip', True, 'random_flip')

tf.app.flags.DEFINE_float('moving_average_decay', None, 'moving_average_decay')


FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu_id

_FILE_PATTERN = '%s-*'

_SPLITS_TO_SIZES = {
    'train': 1292196,
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
      'image/encoded_a': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/encoded_b': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'image/label': tf.FixedLenFeature((), dtype=tf.int64),
    }

    items_to_handlers = {
      'image_a': slim.tfexample_decoder.Image('image/encoded_a', 'image/format'),
      'image_b': slim.tfexample_decoder.Image('image/encoded_b', 'image/format'),  
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

def erasing(img,mean,sl,sh,r1):
#     for attempt in range(100):
    area = img.get_shape().as_list()[0]*img.get_shape().as_list()[1]
    target_area = tf.squeeze(tf.random_uniform([1], minval=sl , maxval=sh, dtype=tf.float32)) * area
    aspect_ratio = tf.squeeze(tf.random_uniform([1], minval=r1 , maxval=1.0/r1, dtype=tf.float32) )

    h = tf.cast(tf.round(tf.sqrt(target_area * aspect_ratio)),tf.int32)
    w = tf.cast(tf.round(tf.sqrt(target_area / aspect_ratio)),tf.int32)

    cond_h = (h < img.get_shape().as_list()[0])
    cond_w = (w < img.get_shape().as_list()[1])
    
    def process(image):
        x1 = tf.squeeze(tf.random_uniform([1], minval=0 , maxval=img.get_shape().as_list()[0]-h, dtype=tf.int32))
        y1 = tf.squeeze(tf.random_uniform([1], minval=0 , maxval=img.get_shape().as_list()[1]-w, dtype=tf.int32))
        shape = image.get_shape().as_list()
        coord_w, coord_h = tf.meshgrid(tf.range(shape[1]),tf.range(shape[0]))
        ind_w = tf.logical_and(coord_w>=y1,coord_w<y1+w)
        ind_h = tf.logical_and(coord_h>=x1,coord_h<x1+h)
        mask = tf.cast(tf.logical_and(ind_w,ind_h),tf.float32)
        masks = []
        values = []
        for idx in range(len(mean)):
            masks.append(mask)
            values.append(mask*mean[idx]*255)
        masks = tf.stack(masks,axis=2)
        values = tf.stack(values,axis=2)
        image = tf.multiply(1-masks,image) + values
        return image
        
        
    img = tf.cond(tf.logical_and(cond_h,cond_w),
                  lambda: process(img),
                  lambda: img)
    return img
class random_erasing(object):
    def __init__(self, probability = 0.4, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.5, 0.5, 0.5]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
    def __call__(self, img):

        rand_num = tf.random_uniform([1], minval=0 , maxval=1, dtype=tf.float32)
        rand_num = tf.squeeze(rand_num)

        img = tf.cond(rand_num>self.probability,
                lambda: erasing(img,self.mean,self.sl,self.sh,self.r1),
                lambda: img)
        return img
def process_image(image):
    if FLAGS.random_crop:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0],dtype=tf.float32,shape=[1, 1, 4])
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=0.1,
            aspect_ratio_range=(0.45, 0.55),
            area_range=(0.70, 1.0),
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
        
        image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(tf.divide(tf.cast(image,tf.float32), 255), 0), distort_bbox)
        tf.summary.image('images_with_box', image_with_box)
        
        image = tf.slice(image, bbox_begin, bbox_size)
        image.set_shape([None, None, 3])
    if FLAGS.random_flip:
        image = tf.image.random_flip_left_right(image)
 
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [FLAGS.target_height, FLAGS.target_width], align_corners=False)
    image = tf.squeeze(image, [0])

    if FLAGS.random_erase:
        re = random_erasing()
        image = re(image)
        
    if FLAGS.random_rotate:
        pi = 3.14
        rand_angle = tf.random_uniform([1], minval=-pi/16, maxval=pi/16, dtype=tf.float32)
        image = tf.contrib.image.rotate(image, rand_angle)
        
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
            [image_a, image_b, label] = provider.get(['image_a','image_b', 'label'])
            
            image_a = process_image(image_a)
            image_b = process_image(image_b)
            image_a.set_shape([FLAGS.target_height, FLAGS.target_width, 3])
            image_b.set_shape([FLAGS.target_height, FLAGS.target_width, 3])
            images_a, images_b, labels = tf.train.batch(
                [image_a, image_b, label],
                batch_size=FLAGS.batch_size,
                num_threads=8,
                capacity=FLAGS.batch_size* 10)

            inputs_queue = prefetch_queue([images_a, images_b, labels])
        
        ######################
        # Select the network #
        ######################
        def model_fn(inputs_queue):
            images_a, images_b, labels = inputs_queue.dequeue()
            model = find_class_by_name(FLAGS.model, [models])()
            if 'ContrastiveModel' in FLAGS.model:
                vec_a,vec_b = model.create_model(images_a, images_b, reuse=False, is_training = True) 
                contrastive_loss = tf.contrib.losses.metric_learning.contrastive_loss(labels,vec_a,vec_b)
                tf.losses.add_loss(contrastive_loss)
            else:
                
                logits = model.create_model(images_a, images_b, reuse=False, is_training = True) 
                label_onehot = tf.one_hot(labels,2)
                crossentropy_loss = tf.losses.softmax_cross_entropy(onehot_labels=label_onehot,logits=logits)
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
                  
            learning_rate_step_boundaries = [int(num_batches_epoch*num_epoches*0.50), int(num_batches_epoch*num_epoches*0.75), int(num_batches_epoch*num_epoches*0.90)]
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
            
        if FLAGS.model=='CoAttention':
            if FLAGS.weights is None:
                # if not FLAGS.moving_average_decay:
                variables = slim.get_model_variables('InceptionV1')
                init_fn = slim.assign_from_checkpoint_fn(
                    os.path.join(FLAGS.checkpoints_dir, 'inception_v1.ckpt'),
                    slim.get_model_variables('InceptionV1'))
        if FLAGS.model=='AttentionBaseModel':
            if FLAGS.weights is None:
                # if not FLAGS.moving_average_decay:
                variables = slim.get_model_variables('InceptionV1')
                init_fn = slim.assign_from_checkpoint_fn(
                    os.path.join(FLAGS.checkpoints_dir, 'inception_v1.ckpt'),
                    slim.get_model_variables('InceptionV1'))
        if FLAGS.model=='CoAttentionBaseModel':
            if FLAGS.weights is None:
                # if not FLAGS.moving_average_decay:
                variables = slim.get_model_variables('InceptionV1')
                init_fn = slim.assign_from_checkpoint_fn(
                    os.path.join(FLAGS.checkpoints_dir, 'inception_v1.ckpt'),
                    slim.get_model_variables('InceptionV1'))
                
        if FLAGS.model=='MultiHeadCoAttention':
            if FLAGS.weights is None:
                # if not FLAGS.moving_average_decay:
                variables = slim.get_model_variables('InceptionV1')
                init_fn = slim.assign_from_checkpoint_fn(
                    os.path.join(FLAGS.checkpoints_dir, 'inception_v1.ckpt'),
                    slim.get_model_variables('InceptionV1'))
        if FLAGS.model=='MultiHeadAttentionBaseModel':
            if FLAGS.weights is None:
                # if not FLAGS.moving_average_decay:
                variables = slim.get_model_variables('InceptionV1')
                init_fn = slim.assign_from_checkpoint_fn(
                    os.path.join(FLAGS.checkpoints_dir, 'inception_v1.ckpt'),
                    slim.get_model_variables('InceptionV1'))
        if FLAGS.model=='MultiHeadAttentionBaseModel_fixed':
            if FLAGS.weights is None:
                # if not FLAGS.moving_average_decay:
                variables = slim.get_model_variables('InceptionV1')
                init_fn = slim.assign_from_checkpoint_fn(
                    os.path.join(FLAGS.checkpoints_dir, 'inception_v1.ckpt'),
                    slim.get_model_variables('InceptionV1'))
        if FLAGS.model=='MultiHeadAttentionBaseModel_res':
            if FLAGS.weights is None:
                # if not FLAGS.moving_average_decay:
                variables = slim.get_model_variables('InceptionV1')
                init_fn = slim.assign_from_checkpoint_fn(
                    os.path.join(FLAGS.checkpoints_dir, 'inception_v1.ckpt'),
                    slim.get_model_variables('InceptionV1'))
        if FLAGS.model=='MultiHeadAttentionBaseModel_set_share_softmax':
            if FLAGS.weights is None:
                # if not FLAGS.moving_average_decay:
                variables = slim.get_model_variables('InceptionV1')
                init_fn = slim.assign_from_checkpoint_fn(
                    os.path.join(FLAGS.checkpoints_dir, 'inception_v1.ckpt'),
                    slim.get_model_variables('InceptionV1'))          
        if FLAGS.model=='CoAttentionBaseModel_v2':
            if FLAGS.weights is None:
                # if not FLAGS.moving_average_decay:
                variables = slim.get_model_variables('InceptionV1')
                init_fn = slim.assign_from_checkpoint_fn(
                    os.path.join(FLAGS.checkpoints_dir, 'inception_v1.ckpt'),
                    slim.get_model_variables('InceptionV1'))
                
        if 'ParallelAttentionBaseModel' in FLAGS.model:
            if FLAGS.weights is None:
                # if not FLAGS.moving_average_decay:
                variables = slim.get_model_variables('InceptionV1')
                init_fn = slim.assign_from_checkpoint_fn(
                    os.path.join(FLAGS.checkpoints_dir, 'inception_v1.ckpt'),
                    slim.get_model_variables('InceptionV1'))
        if 'ContrastiveModel' in FLAGS.model:
            if FLAGS.weights is None:
                # if not FLAGS.moving_average_decay:
                variables = slim.get_model_variables('InceptionV1')
                init_fn = slim.assign_from_checkpoint_fn(
                    os.path.join(FLAGS.checkpoints_dir, 'inception_v1.ckpt'),
                    slim.get_model_variables('InceptionV1'))
                
        if FLAGS.model=='MultiHeadCoAttention_inv4':
            if FLAGS.weights is None:
                # if not FLAGS.moving_average_decay:
                variables = slim.get_model_variables('InceptionV4')
                init_fn = slim.assign_from_checkpoint_fn(
                    os.path.join(FLAGS.checkpoints_dir, 'inception_v4.ckpt'),
                    slim.get_model_variables('InceptionV4'))
        if FLAGS.model=='MultiLayerMultiHeadCoAttention':
            if FLAGS.weights is None:
                # if not FLAGS.moving_average_decay:
                variables = slim.get_model_variables('InceptionV1')
                init_fn = slim.assign_from_checkpoint_fn(
                    os.path.join(FLAGS.checkpoints_dir, 'inception_v1.ckpt'),
                    slim.get_model_variables('InceptionV1'))
        if FLAGS.model=='DCSL_inception_v4':
            if FLAGS.weights is None:
                # if not FLAGS.moving_average_decay:
                variables = slim.get_model_variables('InceptionV4')
                init_fn = slim.assign_from_checkpoint_fn(
                    os.path.join(FLAGS.checkpoints_dir, 'inception_v4.ckpt'),
                    slim.get_model_variables('InceptionV4'))
                
        # compute and update gradients
        with tf.device(config.optimizer_device()):
            if FLAGS.moving_average_decay:
                update_ops.append(variable_averages.apply(moving_average_variables))

            # Variables to train.
            all_trainable = tf.trainable_variables()

            #  and returns a train_tensor and summary_op
            total_loss, grads_and_vars = model_deploy.optimize_clones(clones, training_optimizer, regularization_losses=None, var_list=all_trainable)

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
