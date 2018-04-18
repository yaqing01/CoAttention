from core.__init_paths import *
from utils.utils import add_timing_signal_nd, Attention
tf.app.flags.DEFINE_integer('feature_dim',256, 'Dimension of feature embeddings')
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
FLAGS = tf.app.flags.FLAGS

class BaseModel(object):
    """Inherit from this class when implementing new models."""

    def create_model(self, unused_model_input, **unused_params):
        raise NotImplementedError()
    
class StemBlock:
    def __init__(self, reuse=False):
        self.reuse = reuse

    def __call__(self, net, embedding_length, training=False, scope = ""):
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope(use_batch_norm=True,weight_decay=FLAGS.weight_decay)):
            with tf.variable_scope(scope, reuse=self.reuse):
                with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=training):
                    net = slim.conv2d(net, embedding_length, [1, 1],scope='Conv1x1')
                    net = slim.conv2d(net, embedding_length, [3, 3],scope='Conv3x3a')
                    net = slim.conv2d(net, embedding_length, [3, 3],scope='Conv3x3b')
            self.reuse=True
        return net
    
class StemBlock_res:
    def __init__(self, reuse=False):
        self.reuse = reuse

    def __call__(self, net_in, embedding_length, reuse=False, training=False, scope = ""):
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope(use_batch_norm=True,weight_decay=FLAGS.weight_decay)):
            with tf.variable_scope(scope, reuse=reuse):
                with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=training):
                    net_in = slim.conv2d(net_in, embedding_length, [1, 1],scope='Conv1x1')
                    net = slim.conv2d(net_in, embedding_length, [3, 3],scope='Conv3x3a')
                    net = slim.conv2d(net, embedding_length, [3, 3],scope='Conv3x3b')
                    net = net + net_in
        return net

class AttentionBlock:
    def __init__(self, reuse=False):
        self.reuse = reuse

    def __call__(self, net, atten=None, reuse=False, training=False, scope = ""):
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope(use_batch_norm=True,weight_decay=FLAGS.weight_decay)):
            with tf.variable_scope(scope, reuse=reuse):
                with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=training):
                    feature_len = FLAGS.feature_dim
                    if atten is not None:
                        net = tf.multiply(net,atten)
                        
                    net = slim.conv2d(net, feature_len, [3, 3],scope='Conv3x3a')
                    net = slim.conv2d(net, feature_len, [3, 3],scope='Conv3x3b')
                    net = slim.conv2d(net, 1, [1, 1],scope='Conv1x1',activation_fn=None)
                    net = tf.sigmoid(net)
            self.reuse=True
        return net
    
class CompareBlock:
    def __init__(self, reuse=False):
        self.reuse = reuse

    def __call__(self, net_a, net_b, training=False,reuse=False, scope = ""):
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope(use_batch_norm=True,weight_decay=FLAGS.weight_decay)):
            with tf.variable_scope(scope, reuse=reuse):
                with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=training):
                    feature_len = FLAGS.feature_dim
                    net = tf.concat([net_a,net_b],axis=3)
                    
                    net = slim.conv2d(net, feature_len, [1, 1],scope='Conv1x1')    
                    net = slim.conv2d(net, feature_len, [3, 3],scope='Conv3x3a')
                    net = slim.conv2d(net, feature_len, [3, 3],scope='Conv3x3b')
        return net

class QueryBlock:
    def __init__(self, reuse=False):
        self.reuse = reuse

    def __call__(self, net, atten, training=False,reuse=False, scope = ""):
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope(use_batch_norm=True,weight_decay=FLAGS.weight_decay)):
            with tf.variable_scope(scope, reuse=reuse):
                with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=training):
                    feature_len = FLAGS.feature_dim
                    
                    net = tf.multiply(net,atten)
                    net = slim.conv2d(net, feature_len, [3, 3],scope='Conv3x3a')
                    net = slim.conv2d(net, feature_len, [3, 3],scope='Conv3x3b')
        return net

class ClassificationBlock:
    def __init__(self, reuse=False):
        self.reuse = reuse

    def __call__(self, net, training=False, scope = ""):
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope(use_batch_norm=True,weight_decay=FLAGS.weight_decay)):
            with tf.variable_scope(scope, reuse=self.reuse):
                with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=training):
                    feature_len = FLAGS.feature_dim
                    
                    net = slim.flatten(net)
                    net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='Match_Prelogits1')
                    net = slim.dropout(net, 0.50, scope='Dropout_match',is_training=training)
                    net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='Match_Prelogits2')
                    net = slim.dropout(net, 0.50, scope='Dropout_match',is_training=training)
                    net = slim.fully_connected(net, 2, activation_fn=None, scope='Match_logits')
                    
            self.reuse=True
        return net

class ScoreBlock:
    def __init__(self, reuse=False):
        self.reuse = reuse

    def __call__(self, net, training=False, scope = "Score"):
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope(use_batch_norm=True,weight_decay=FLAGS.weight_decay)):
            with tf.variable_scope(scope, reuse=self.reuse):
                with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=training):
                    with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_initializer=trunc_normal(0.01)):
                        feature_len = FLAGS.feature_dim

                        net = slim.flatten(net)
                        net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='Score_Prelogits1')
                        net = slim.dropout(net, 0.50, scope='Dropout_match',is_training=training)
                        net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='Score_Prelogits2')
                        net = slim.dropout(net, 0.50, scope='Dropout_match',is_training=training)
                        net = slim.fully_connected(net, 1, activation_fn=None, scope='Score_logits')
                    
            self.reuse=True
        return net
    
def SigmoidAttention(Q,K,V):
    scale = tf.sqrt(Q.get_shape().as_list()[2]*1.0)
    dot_product = tf.matmul(Q,K,transpose_b=True)/scale
    sigmoid_attention = tf.sigmoid(dot_product)
    sigmoid_attention_mean = tf.reduce_mean(sigmoid_attention,axis=2,keepdims=True)
    
    attention_value = tf.matmul(sigmoid_attention,V)/sigmoid_attention_mean
    
    return attention_value,sigmoid_attention

class MultiHeadAttentionBlock:
    def __init__(self, reuse=False):
        self.reuse = reuse

    def __call__(self, Q, K, V, num_heads, training=False,evaluation=False,reuse=False, scope = ""):
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

        attentions = []
            
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope(use_batch_norm=True,weight_decay=FLAGS.weight_decay)):
            with tf.variable_scope(scope, reuse=reuse):
                with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=training):
                    
                    for HeadIdx in range(num_heads):
                        query = slim.conv2d(Q, d_key, [1, 1], scope='QueryRep-'+str(HeadIdx))
                        query = slim.conv2d(query, d_key, [3, 3],scope='QueryRepb-'+str(HeadIdx), padding='SAME', )
#                             activation_fn = None, normalizer_fn=None, normalizer_params=None)
                        query = tf.reshape(query,[query.get_shape().as_list()[0],-1,query.get_shape().as_list()[3]])
                        
                        key = slim.conv2d(K, d_key, [1, 1], scope='KeyRep-'+str(HeadIdx))
                        key = slim.conv2d(key, d_key, [3, 3],scope='KeyRepb-'+str(HeadIdx), padding='SAME', )
#                             activation_fn = None, normalizer_fn=None, normalizer_params=None)
                        key = tf.reshape(key,[key.get_shape().as_list()[0],-1,key.get_shape().as_list()[3]])

                        value = slim.conv2d(V, d_value, [1, 1], scope='ValueRep-'+str(HeadIdx))
                        value = slim.conv2d(value, d_value, [3, 3],scope='ValueRepb-'+str(HeadIdx), padding='SAME', )
#                             activation_fn = None, normalizer_fn=None, normalizer_params=None)
                        value = tf.reshape(value,[value.get_shape().as_list()[0],-1,value.get_shape().as_list()[3]])
                        
                        head,attention = Attention(query,key,value)
                        
                        heads.append(head)
                        attentions.append(attention)
                        

                    heads = tf.concat(heads,axis=2)
                    attentions = tf.stack(attentions,axis=1)
#                     heads = slim.fully_connected(heads, d_model) 
        return heads,attentions

class MultiHeadAttentionBlock_v2:
    def __init__(self, reuse=False):
        self.reuse = reuse

    def __call__(self, Q, K, V, num_heads, training=False,evaluation=False,reuse=False, scope = ""):
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

        attentions = []
            
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope(use_batch_norm=True,weight_decay=FLAGS.weight_decay)):
            with tf.variable_scope(scope, reuse=reuse):
                with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=training):
                    
                    for HeadIdx in range(num_heads):
                        with tf.variable_scope('Query', reuse=False):
                            query = slim.conv2d(Q, d_key, [1, 1], scope='QueryRep-'+str(HeadIdx))
                            query = slim.conv2d(query, d_key, [3, 3],scope='QueryRepb-'+str(HeadIdx), padding='SAME', )
    #                             activation_fn = None, normalizer_fn=None, normalizer_params=None)
                            query = tf.reshape(query,[query.get_shape().as_list()[0],-1,query.get_shape().as_list()[3]])
                        with tf.variable_scope('Query', reuse=True):
                            key = slim.conv2d(K, d_key, [1, 1], scope='QueryRep-'+str(HeadIdx))
                            key = slim.conv2d(key, d_key, [3, 3],scope='QueryRepb-'+str(HeadIdx), padding='SAME', )
    #                             activation_fn = None, normalizer_fn=None, normalizer_params=None)
                            key = tf.reshape(key,[key.get_shape().as_list()[0],-1,key.get_shape().as_list()[3]])

                        value = slim.conv2d(V, d_value, [1, 1], scope='ValueRep-'+str(HeadIdx))
                        value = slim.conv2d(value, d_value, [3, 3],scope='ValueRepb-'+str(HeadIdx), padding='SAME', )
#                             activation_fn = None, normalizer_fn=None, normalizer_params=None)
                        value = tf.reshape(value,[value.get_shape().as_list()[0],-1,value.get_shape().as_list()[3]])
                        
                        head,attention = Attention(query,key,value)
                        
                        heads.append(head)
                        attentions.append(attention)
                        

                    heads = tf.concat(heads,axis=2)
                    attentions = tf.stack(attentions,axis=1)
#                     heads = slim.fully_connected(heads, d_model) 
        return heads,attentions

def Attention_noscale(Q,K,V):
    # Q,K,V
    # Q: query batch_size x query_num x query_dim
    # K: key batch_size x key_num x query_dim
    # V: value batch_size x value_num x value_dim
    scale = tf.sqrt(Q.get_shape().as_list()[2]*1.0)
    dot_product = tf.matmul(Q,K,transpose_b=True)
    softmax_attention = tf.nn.softmax(dot_product)
    attention_value = tf.matmul(softmax_attention,V)

    return attention_value,softmax_attention
class MultiHeadAttentionBlock_v3:
    def __init__(self, reuse=False):
        self.reuse = reuse

    def __call__(self, Q, K, V, num_heads, training=False,evaluation=False,reuse=False, scope = ""):
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

        attentions = []
            
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope(use_batch_norm=True,weight_decay=FLAGS.weight_decay)):
            with tf.variable_scope(scope, reuse=reuse):
                with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=training):
                    
                    for HeadIdx in range(num_heads):
                        with tf.variable_scope('Query', reuse=False):
                            query = slim.conv2d(Q, d_key, [1, 1], scope='QueryRep-'+str(HeadIdx))
                            query = slim.conv2d(query, d_key, [3, 3],scope='QueryRepb-'+str(HeadIdx), padding='SAME',
                                activation_fn = None, )
#                             normalizer_fn=None, normalizer_params=None)
                            query = tf.reshape(query,[query.get_shape().as_list()[0],-1,query.get_shape().as_list()[3]])
                            query = query/tf.expand_dims(tf.norm(query,axis=2),axis=2)
                            
                        with tf.variable_scope('Query', reuse=True):
                            key = slim.conv2d(K, d_key, [1, 1], scope='QueryRep-'+str(HeadIdx))
                            key = slim.conv2d(key, d_key, [3, 3],scope='QueryRepb-'+str(HeadIdx), padding='SAME',
                                activation_fn = None, )
#                             normalizer_fn=None, normalizer_params=None)
                            key = tf.reshape(key,[key.get_shape().as_list()[0],-1,key.get_shape().as_list()[3]])
                            key = key/tf.expand_dims(tf.norm(key,axis=2),axis=2)

                        value = slim.conv2d(V, d_value, [1, 1], scope='ValueRep-'+str(HeadIdx))
                        value = slim.conv2d(value, d_value, [3, 3],scope='ValueRepb-'+str(HeadIdx), padding='SAME',
                            activation_fn = None, )
#                         normalizer_fn=None, normalizer_params=None)
                        value = tf.reshape(value,[value.get_shape().as_list()[0],-1,value.get_shape().as_list()[3]])
                        
                        head,attention = Attention_noscale(query,key,value)
                        
                        heads.append(head)
                        attentions.append(attention)
                        

                    heads = tf.concat(heads,axis=2)
                    attentions = tf.stack(attentions,axis=1)
#                     heads = slim.fully_connected(heads, d_model) 
        return heads,attentions

class MultiHeadAttentionBaseModel(BaseModel):
    def create_model(self, input_a, input_b, reuse, is_training=True,is_evaluation=False, **unused_params):
        
        fn_extraction = FeatureExtractor_inv1(reuse = reuse)
        stem_block = StemBlock(reuse = reuse)
        attention_block = AttentionBlock(reuse = reuse)
        query_block = QueryBlock(reuse = reuse)
        compare_block = CompareBlock(reuse = reuse)
        classification_block = ClassificationBlock(reuse = reuse)
        fn_coattention = MultiHeadAttentionBlock(reuse = reuse)
        
        embedding_length = FLAGS.feature_dim
        
        feat_map_a = fn_extraction(input_a,training=is_training, endpoint_name = 'Mixed_4f')
        feat_map_b = fn_extraction(input_b,training=is_training, endpoint_name = 'Mixed_4f')
        
        feat_map_a = stem_block(feat_map_a,embedding_length, scope='stem_embedding')
        feat_map_b = stem_block(feat_map_b,embedding_length, scope='stem_embedding')
        
        # location embedding
        feat_map_a = add_timing_signal_nd(feat_map_a)
        feat_map_b = add_timing_signal_nd(feat_map_b)
        
        # ATTENTION
        A2AAtt,attentions_1 = fn_coattention(feat_map_a,feat_map_a,feat_map_a,
                                             FLAGS.num_heads,training=is_training,evaluation=is_evaluation,scope='SelfAtt1')
        A2BAtt,attentions_2 = fn_coattention(feat_map_a,feat_map_a,feat_map_b,
                                             FLAGS.num_heads,training=is_training,evaluation=is_evaluation,scope='CoAtt1')
        
        A2AAtt = tf.reshape(A2AAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        A2BAtt = tf.reshape(A2BAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        # COMPARE
        compare_map_ab = compare_block(A2AAtt,A2BAtt,scope='compare',reuse=False)

        # ATTENTION, reuse = True
        B2BAtt,attentions_3 = fn_coattention(feat_map_b,feat_map_b,feat_map_b,FLAGS.num_heads,
                                             training=is_training,evaluation=is_evaluation,reuse=True, scope='SelfAtt1')
        B2AAtt,attentions_4 = fn_coattention(feat_map_b,feat_map_b,feat_map_a,FLAGS.num_heads,
                                             training=is_training,evaluation=is_evaluation,reuse=True, scope='CoAtt1')
        
        B2BAtt = tf.reshape(B2BAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        B2AAtt = tf.reshape(B2AAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        # COMPARE
        compare_map_ba = compare_block(B2BAtt,B2AAtt,scope='compare',reuse=True)   
        

        # Compare Final
        compmap = compare_block(compare_map_ab,compare_map_ba,scope='compare-final',reuse=False)  
        score = classification_block(compmap)
        
        # summary
        
        attentions_1 = tf.reshape(attentions_1, [attentions_1.get_shape()[0],attentions_1.get_shape()[1],attentions_1.get_shape()[2],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2] ])
        
        slice_a = tf.slice(attentions_1, [0,0,31,0,0],[-1,1,1,-1,-1])
        slice_a = tf.squeeze(slice_a)
        tf.summary.image('attentions_5_3_1',tf.expand_dims(slice_a,axis=3))
        slice_a = tf.slice(attentions_1, [0,0,39,0,0],[-1,1,1,-1,-1])
        slice_a = tf.squeeze(slice_a)
        tf.summary.image('attentions_6_4_1',tf.expand_dims(slice_a,axis=3))
        
        tf.summary.image('feat_map_a',tf.subtract(tf.multiply(input_a, 0.5),-0.5))
        tf.summary.image('feat_map_b',tf.subtract(tf.multiply(input_b, 0.5),-0.5))
        
        return score
 
class MultiHeadAttentionBaseModel_fixed(BaseModel):
    def create_model(self, input_a, input_b, reuse, is_training=True,is_evaluation=False, **unused_params):
        
        fn_extraction = FeatureExtractor_inv1(reuse = reuse)
        stem_block = StemBlock(reuse = reuse)
        attention_block = AttentionBlock(reuse = reuse)
        query_block = QueryBlock(reuse = reuse)
        compare_block = CompareBlock(reuse = reuse)
        classification_block = ClassificationBlock(reuse = reuse)
        fn_coattention = MultiHeadAttentionBlock_v2(reuse = reuse)
        
        embedding_length = FLAGS.feature_dim
        
        feat_map_a = fn_extraction(input_a,training=is_training, endpoint_name = 'Mixed_4f')
        feat_map_b = fn_extraction(input_b,training=is_training, endpoint_name = 'Mixed_4f')
        
        feat_map_a = stem_block(feat_map_a,embedding_length,training=is_training, scope='stem_embedding')
        feat_map_b = stem_block(feat_map_b,embedding_length,training=is_training, scope='stem_embedding')
        
        print(feat_map_a.shape)
        # location embedding
        feat_map_a = add_timing_signal_nd(feat_map_a)
        feat_map_b = add_timing_signal_nd(feat_map_b)
        
        # ATTENTION
        A2AAtt,attentions_1 = fn_coattention(feat_map_a,feat_map_a,feat_map_a,
                                             FLAGS.num_heads,training=is_training,evaluation=is_evaluation,scope='SelfAtt1')
        A2BAtt,attentions_2 = fn_coattention(feat_map_a,feat_map_b,feat_map_b,
                                             FLAGS.num_heads,training=is_training,evaluation=is_evaluation,scope='CoAtt1')
        
        A2AAtt = tf.reshape(A2AAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        A2BAtt = tf.reshape(A2BAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        # COMPARE
        compare_map_ab = compare_block(A2AAtt,A2BAtt,training=is_training,scope='compare',reuse=False)

        # ATTENTION, reuse = True
        B2BAtt,attentions_3 = fn_coattention(feat_map_b,feat_map_b,feat_map_b,FLAGS.num_heads,
                                             training=is_training,evaluation=is_evaluation,reuse=True, scope='SelfAtt1')
        B2AAtt,attentions_4 = fn_coattention(feat_map_b,feat_map_a,feat_map_a,FLAGS.num_heads,
                                             training=is_training,evaluation=is_evaluation,reuse=True, scope='CoAtt1')
        
        B2BAtt = tf.reshape(B2BAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        B2AAtt = tf.reshape(B2AAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        # COMPARE
        compare_map_ba = compare_block(B2BAtt,B2AAtt,training=is_training,scope='compare',reuse=True)   
        

        # Compare Final
        compmap = compare_block(compare_map_ab,compare_map_ba,training=is_training,scope='compare-final',reuse=False)  
        score = classification_block(compmap,training=is_training)
        
        # summary
        
        attentions_1 = tf.reshape(attentions_1, [attentions_1.get_shape()[0],attentions_1.get_shape()[1],attentions_1.get_shape()[2],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2] ])
        
        slice_a = tf.slice(attentions_1, [0,0,31,0,0],[-1,1,1,-1,-1])
        slice_a = tf.squeeze(slice_a)
        tf.summary.image('attentions_5_3_1',tf.expand_dims(slice_a,axis=3))
        slice_a = tf.slice(attentions_1, [0,0,39,0,0],[-1,1,1,-1,-1])
        slice_a = tf.squeeze(slice_a)
        tf.summary.image('attentions_6_4_1',tf.expand_dims(slice_a,axis=3))
        
        attentions_2 = tf.reshape(attentions_2, [attentions_2.get_shape()[0],attentions_2.get_shape()[1],attentions_2.get_shape()[2],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2] ])
        
        slice_a = tf.slice(attentions_2, [0,0,31,0,0],[-1,1,1,-1,-1])
        slice_a = tf.squeeze(slice_a)
        tf.summary.image('attentions_5_3_1_b',tf.expand_dims(slice_a,axis=3))
        slice_a = tf.slice(attentions_2, [0,0,39,0,0],[-1,1,1,-1,-1])
        slice_a = tf.squeeze(slice_a)
        tf.summary.image('attentions_6_4_1_b',tf.expand_dims(slice_a,axis=3))
        
        tf.summary.image('feat_map_a',tf.subtract(tf.multiply(input_a, 0.5),-0.5))
        tf.summary.image('feat_map_b',tf.subtract(tf.multiply(input_b, 0.5),-0.5))
        
        return score
    
class MultiHeadAttentionBaseModel_res(BaseModel):
    def create_model(self, input_a, input_b, reuse, is_training=True,is_evaluation=False, **unused_params):
        
        fn_extraction = FeatureExtractor_inv1(reuse = reuse)
        stem_block = StemBlock_res(reuse = reuse)
        attention_block = AttentionBlock(reuse = reuse)
        query_block = QueryBlock(reuse = reuse)
        compare_block = CompareBlock(reuse = reuse)
        classification_block = ClassificationBlock(reuse = reuse)
        fn_coattention = MultiHeadAttentionBlock_v2(reuse = reuse)
        
        embedding_length = FLAGS.feature_dim
        
        feat_map_a = fn_extraction(input_a,training=is_training, endpoint_name = 'Mixed_4f')
        feat_map_b = fn_extraction(input_b,training=is_training, endpoint_name = 'Mixed_4f')
        
        feat_map_a = stem_block(feat_map_a,embedding_length,training=is_training, reuse=False, scope='stem_embedding')
        feat_map_b = stem_block(feat_map_b,embedding_length,training=is_training, reuse=True, scope='stem_embedding')
        
        print(feat_map_a.shape)
        # location embedding
        feat_map_a = add_timing_signal_nd(feat_map_a)
        feat_map_b = add_timing_signal_nd(feat_map_b)
        
        # ATTENTION
        A2AAtt,attentions_1 = fn_coattention(feat_map_a,feat_map_a,feat_map_a,
                                             FLAGS.num_heads,training=is_training,evaluation=is_evaluation,scope='SelfAtt1')
        A2BAtt,attentions_2 = fn_coattention(feat_map_a,feat_map_b,feat_map_b,
                                             FLAGS.num_heads,training=is_training,evaluation=is_evaluation,scope='CoAtt1')
        
        A2AAtt = tf.reshape(A2AAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        A2BAtt = tf.reshape(A2BAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        A2AAtt = A2AAtt + feat_map_a
        A2BAtt = A2BAtt + feat_map_b
        A2AAtt = stem_block(A2AAtt,embedding_length,training=is_training,reuse=False, scope='stem_embedding_self')
        A2BAtt = stem_block(A2BAtt,embedding_length,training=is_training,reuse=False, scope='stem_embedding_co')
        # COMPARE
        compare_map_ab = compare_block(A2AAtt,A2BAtt,training=is_training,scope='compare',reuse=False)

        # ATTENTION, reuse = True
        B2BAtt,attentions_3 = fn_coattention(feat_map_b,feat_map_b,feat_map_b,FLAGS.num_heads,
                                             training=is_training,evaluation=is_evaluation,reuse=True, scope='SelfAtt1')
        B2AAtt,attentions_4 = fn_coattention(feat_map_b,feat_map_a,feat_map_a,FLAGS.num_heads,
                                             training=is_training,evaluation=is_evaluation,reuse=True, scope='CoAtt1')
        
        B2BAtt = tf.reshape(B2BAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        B2AAtt = tf.reshape(B2AAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        B2BAtt = B2BAtt + feat_map_b
        B2AAtt = B2AAtt + feat_map_a
        B2BAtt = stem_block(B2BAtt,embedding_length,training=is_training,reuse=True, scope='stem_embedding_self')
        B2AAtt = stem_block(B2AAtt,embedding_length,training=is_training,reuse=True, scope='stem_embedding_co')
        
        # COMPARE
        compare_map_ba = compare_block(B2BAtt,B2AAtt,training=is_training,scope='compare',reuse=True)   
        

        # Compare Final
        compmap = compare_block(compare_map_ab,compare_map_ba,training=is_training,scope='compare-final',reuse=False)  
        score = classification_block(compmap,training=is_training)
        
        # summary
        
        attentions_1 = tf.reshape(attentions_1, [attentions_1.get_shape()[0],attentions_1.get_shape()[1],attentions_1.get_shape()[2],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2] ])
        
        slice_a = tf.slice(attentions_1, [0,0,31,0,0],[-1,1,1,-1,-1])
        slice_a = tf.squeeze(slice_a)
        tf.summary.image('attentions_5_3_1',tf.expand_dims(slice_a,axis=3))
        slice_a = tf.slice(attentions_1, [0,0,39,0,0],[-1,1,1,-1,-1])
        slice_a = tf.squeeze(slice_a)
        tf.summary.image('attentions_6_4_1',tf.expand_dims(slice_a,axis=3))
        
        attentions_2 = tf.reshape(attentions_2, [attentions_2.get_shape()[0],attentions_2.get_shape()[1],attentions_2.get_shape()[2],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2] ])
        
        slice_a = tf.slice(attentions_2, [0,0,31,0,0],[-1,1,1,-1,-1])
        slice_a = tf.squeeze(slice_a)
        tf.summary.image('attentions_5_3_1_b',tf.expand_dims(slice_a,axis=3))
        slice_a = tf.slice(attentions_2, [0,0,39,0,0],[-1,1,1,-1,-1])
        slice_a = tf.squeeze(slice_a)
        tf.summary.image('attentions_6_4_1_b',tf.expand_dims(slice_a,axis=3))
        
        tf.summary.image('feat_map_a',tf.subtract(tf.multiply(input_a, 0.5),-0.5))
        tf.summary.image('feat_map_b',tf.subtract(tf.multiply(input_b, 0.5),-0.5))
        
        return score
    
class MultiHeadAttentionBaseModel_set(BaseModel):
    def create_model(self, input_a, input_b, reuse, is_training=True,is_evaluation=False, **unused_params):
        
        fn_extraction = FeatureExtractor_inv1(reuse = reuse)
        stem_block = StemBlock(reuse = reuse)
        attention_block = AttentionBlock(reuse = reuse)
        query_block = QueryBlock(reuse = reuse)
        compare_block = CompareBlock(reuse = reuse)
        score_block = ScoreBlock(reuse = reuse)
        fn_coattention = MultiHeadAttentionBlock_v2(reuse = reuse)
        
        embedding_length = FLAGS.feature_dim
        
        feat_map_a = fn_extraction(input_a,training=is_training, endpoint_name = 'Mixed_4f')
        feat_map_b = fn_extraction(input_b,training=is_training, endpoint_name = 'Mixed_4f')
        
        feat_map_a = stem_block(feat_map_a,embedding_length, training=is_training, scope='stem_embedding')
        feat_map_b = stem_block(feat_map_b,embedding_length, training=is_training, scope='stem_embedding')
        
        print(feat_map_a.shape)
        # location embedding
        feat_map_a = add_timing_signal_nd(feat_map_a)
        feat_map_b = add_timing_signal_nd(feat_map_b)
        
        # ATTENTION
        A2AAtt,attentions_1 = fn_coattention(feat_map_a,feat_map_a,feat_map_a,
                                             FLAGS.num_heads,training=is_training,evaluation=is_evaluation,scope='SelfAtt1')
        A2BAtt,attentions_2 = fn_coattention(feat_map_a,feat_map_b,feat_map_b,
                                             FLAGS.num_heads,training=is_training,evaluation=is_evaluation,scope='CoAtt1')
        
        A2AAtt = tf.reshape(A2AAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        A2BAtt = tf.reshape(A2BAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        # COMPARE
        compare_map_ab = compare_block(A2AAtt,A2BAtt,training=is_training,scope='compare',reuse=False)

        # ATTENTION, reuse = True
        B2BAtt,attentions_3 = fn_coattention(feat_map_b,feat_map_b,feat_map_b,FLAGS.num_heads,
                                             training=is_training,evaluation=is_evaluation,reuse=True, scope='SelfAtt1')
        B2AAtt,attentions_4 = fn_coattention(feat_map_b,feat_map_a,feat_map_a,FLAGS.num_heads,
                                             training=is_training,evaluation=is_evaluation,reuse=True, scope='CoAtt1')
        
        B2BAtt = tf.reshape(B2BAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        B2AAtt = tf.reshape(B2AAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        # COMPARE
        compare_map_ba = compare_block(B2BAtt,B2AAtt,training=is_training,scope='compare',reuse=True)   
        

        # Compare Final
        compmap = compare_block(compare_map_ab,compare_map_ba,training=is_training,scope='compare-final',reuse=False)  
        score = score_block(compmap,training=is_training)
        
        # summary
        


        
        if not is_evaluation:
            attentions_1 = tf.reshape(attentions_1, [attentions_1.get_shape()[0],attentions_1.get_shape()[1],attentions_1.get_shape()[2],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2] ])
            slice_a = tf.slice(attentions_1, [0,0,31,0,0],[-1,1,1,-1,-1])
            slice_a = tf.squeeze(slice_a)
            tf.summary.image('attentions_5_3_1',tf.expand_dims(slice_a,axis=3))
            slice_a = tf.slice(attentions_1, [0,0,39,0,0],[-1,1,1,-1,-1])
            slice_a = tf.squeeze(slice_a)
            tf.summary.image('attentions_6_4_1',tf.expand_dims(slice_a,axis=3))

            attentions_2 = tf.reshape(attentions_2, [attentions_2.get_shape()[0],attentions_2.get_shape()[1],attentions_2.get_shape()[2],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2] ])

            slice_a = tf.slice(attentions_2, [0,0,31,0,0],[-1,1,1,-1,-1])
            slice_a = tf.squeeze(slice_a)
            tf.summary.image('attentions_5_3_1_b',tf.expand_dims(slice_a,axis=3))
            slice_a = tf.slice(attentions_2, [0,0,39,0,0],[-1,1,1,-1,-1])
            slice_a = tf.squeeze(slice_a)
            tf.summary.image('attentions_6_4_1_b',tf.expand_dims(slice_a,axis=3))

            tf.summary.image('feat_map_a',tf.subtract(tf.multiply(input_a, 0.5),-0.5))
            tf.summary.image('feat_map_b',tf.subtract(tf.multiply(input_b, 0.5),-0.5))
            return score
        else:
            attentions = []
            attentions.append(attentions_1)
            attentions.append(attentions_2)
            attentions.append(attentions_3)
            attentions.append(attentions_4)
            return score,attentions
        
class MultiHeadAttentionBaseModel_set_share(BaseModel):
    def create_model(self, input_a, input_b, reuse, is_training=True,is_evaluation=False, **unused_params):
        
        fn_extraction = FeatureExtractor_inv1(reuse = reuse)
        stem_block = StemBlock(reuse = reuse)
        attention_block = AttentionBlock(reuse = reuse)
        query_block = QueryBlock(reuse = reuse)
        compare_block = CompareBlock(reuse = reuse)
        score_block = ScoreBlock(reuse = reuse)
        fn_coattention = MultiHeadAttentionBlock_v2(reuse = reuse)
        
        embedding_length = FLAGS.feature_dim
        
        feat_map_a = fn_extraction(input_a,training=is_training, endpoint_name = 'Mixed_4f')
        feat_map_b = fn_extraction(input_b,training=is_training, endpoint_name = 'Mixed_4f')
        
        feat_map_a = stem_block(feat_map_a,embedding_length, training=is_training, scope='stem_embedding')
        feat_map_b = stem_block(feat_map_b,embedding_length, training=is_training, scope='stem_embedding')
        
        print(feat_map_a.shape)
        # location embedding
        feat_map_a = add_timing_signal_nd(feat_map_a)
        feat_map_b = add_timing_signal_nd(feat_map_b)
        
        # ATTENTION
        A2AAtt,attentions_1 = fn_coattention(feat_map_a,feat_map_a,feat_map_a,FLAGS.num_heads,
                                             training=is_training,evaluation=is_evaluation,scope='CoAtt1')
        A2BAtt,attentions_2 = fn_coattention(feat_map_a,feat_map_b,feat_map_b,FLAGS.num_heads,
                                             training=is_training,evaluation=is_evaluation,reuse=True, scope='CoAtt1')
        
        A2AAtt = tf.reshape(A2AAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        A2BAtt = tf.reshape(A2BAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        # COMPARE
        compare_map_ab = compare_block(A2AAtt,A2BAtt,training=is_training,scope='compare',reuse=False)

        # ATTENTION, reuse = True
        B2BAtt,attentions_3 = fn_coattention(feat_map_b,feat_map_b,feat_map_b,FLAGS.num_heads,
                                             training=is_training,evaluation=is_evaluation,reuse=True, scope='CoAtt1')
        B2AAtt,attentions_4 = fn_coattention(feat_map_b,feat_map_a,feat_map_a,FLAGS.num_heads,
                                             training=is_training,evaluation=is_evaluation,reuse=True, scope='CoAtt1')
        
        B2BAtt = tf.reshape(B2BAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        B2AAtt = tf.reshape(B2AAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        # COMPARE
        compare_map_ba = compare_block(B2BAtt,B2AAtt,training=is_training,scope='compare',reuse=True)   
        

        # Compare Final
        compmap = compare_block(compare_map_ab,compare_map_ba,training=is_training,scope='compare-final',reuse=False)  
        score = score_block(compmap,training=is_training)
        
        # summary
        


        
        if not is_evaluation:
            attentions_1 = tf.reshape(attentions_1, [attentions_1.get_shape()[0],attentions_1.get_shape()[1],attentions_1.get_shape()[2],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2] ])
            slice_a = tf.slice(attentions_1, [0,0,31,0,0],[-1,1,1,-1,-1])
            slice_a = tf.squeeze(slice_a)
            tf.summary.image('attentions_5_3_1',tf.expand_dims(slice_a,axis=3))
            slice_a = tf.slice(attentions_1, [0,0,39,0,0],[-1,1,1,-1,-1])
            slice_a = tf.squeeze(slice_a)
            tf.summary.image('attentions_6_4_1',tf.expand_dims(slice_a,axis=3))

            attentions_2 = tf.reshape(attentions_2, [attentions_2.get_shape()[0],attentions_2.get_shape()[1],attentions_2.get_shape()[2],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2] ])

            slice_a = tf.slice(attentions_2, [0,0,31,0,0],[-1,1,1,-1,-1])
            slice_a = tf.squeeze(slice_a)
            tf.summary.image('attentions_5_3_1_b',tf.expand_dims(slice_a,axis=3))
            slice_a = tf.slice(attentions_2, [0,0,39,0,0],[-1,1,1,-1,-1])
            slice_a = tf.squeeze(slice_a)
            tf.summary.image('attentions_6_4_1_b',tf.expand_dims(slice_a,axis=3))

            tf.summary.image('feat_map_a',tf.subtract(tf.multiply(input_a, 0.5),-0.5))
            tf.summary.image('feat_map_b',tf.subtract(tf.multiply(input_b, 0.5),-0.5))
            return score
        else:
            attentions = []
            attentions.append(attentions_1)
            attentions.append(attentions_2)
            attentions.append(attentions_3)
            attentions.append(attentions_4)
            return score,attentions 
        
class MultiHeadAttentionBaseModel_set_share_feature_extract(BaseModel):
    def create_model(self, input_a, reuse, is_training=True,is_evaluation=False, **unused_params):
        
        fn_extraction = FeatureExtractor_inv1(reuse = reuse)
        stem_block = StemBlock(reuse = reuse)
        attention_block = AttentionBlock(reuse = reuse)
        query_block = QueryBlock(reuse = reuse)
        compare_block = CompareBlock(reuse = reuse)
        score_block = ScoreBlock(reuse = reuse)
        fn_coattention = MultiHeadAttentionBlock_v2(reuse = reuse)
        embedding_length = FLAGS.feature_dim
        feat_map_a = fn_extraction(input_a,training=is_training, endpoint_name = 'Mixed_4f')
        # location embedding
        feat_map_a = stem_block(feat_map_a,embedding_length, training=is_training, scope='stem_embedding')
        feat_map_a = add_timing_signal_nd(feat_map_a)
        return feat_map_a 
    
class MultiHeadAttentionBaseModel_set_share_match(BaseModel):
    def create_model(self, feat_map_a, feat_map_b, reuse, is_training=True,is_evaluation=False, **unused_params):
        
        fn_extraction = FeatureExtractor_inv1(reuse = reuse)
        stem_block = StemBlock(reuse = reuse)
        attention_block = AttentionBlock(reuse = reuse)
        query_block = QueryBlock(reuse = reuse)
        compare_block = CompareBlock(reuse = reuse)
        score_block = ScoreBlock(reuse = reuse)
        fn_coattention = MultiHeadAttentionBlock_v2(reuse = reuse)
        
        embedding_length = FLAGS.feature_dim
        
        # ATTENTION
        A2AAtt,attentions_1 = fn_coattention(feat_map_a,feat_map_a,feat_map_a,FLAGS.num_heads,
                                             training=is_training,evaluation=is_evaluation,scope='CoAtt1')
        A2BAtt,attentions_2 = fn_coattention(feat_map_a,feat_map_b,feat_map_b,FLAGS.num_heads,
                                             training=is_training,evaluation=is_evaluation,reuse=True, scope='CoAtt1')
        
        A2AAtt = tf.reshape(A2AAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        A2BAtt = tf.reshape(A2BAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        # COMPARE
        compare_map_ab = compare_block(A2AAtt,A2BAtt,training=is_training,scope='compare',reuse=False)

        # ATTENTION, reuse = True
        B2BAtt,attentions_3 = fn_coattention(feat_map_b,feat_map_b,feat_map_b,FLAGS.num_heads,
                                             training=is_training,evaluation=is_evaluation,reuse=True, scope='CoAtt1')
        B2AAtt,attentions_4 = fn_coattention(feat_map_b,feat_map_a,feat_map_a,FLAGS.num_heads,
                                             training=is_training,evaluation=is_evaluation,reuse=True, scope='CoAtt1')
        
        B2BAtt = tf.reshape(B2BAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        B2AAtt = tf.reshape(B2AAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        # COMPARE
        compare_map_ba = compare_block(B2BAtt,B2AAtt,training=is_training,scope='compare',reuse=True)   
        

        # Compare Final
        compmap = compare_block(compare_map_ab,compare_map_ba,training=is_training,scope='compare-final',reuse=False)  
        score = score_block(compmap,training=is_training)

        return score

class MultiHeadAttentionBaseModel_set_inv3(BaseModel):
    def create_model(self, input_a, input_b, reuse, is_training=True,is_evaluation=False, **unused_params):
        
        fn_extraction = FeatureExtractor_inv3(reuse = reuse)
        stem_block = StemBlock(reuse = reuse)
        attention_block = AttentionBlock(reuse = reuse)
        query_block = QueryBlock(reuse = reuse)
        compare_block = CompareBlock(reuse = reuse)
        score_block = ScoreBlock(reuse = reuse)
        fn_coattention = MultiHeadAttentionBlock_v2(reuse = reuse)
        
        embedding_length = FLAGS.feature_dim
        
        feat_map_a = fn_extraction(input_a,training=is_training, endpoint_name = 'Mixed_6e')
        feat_map_b = fn_extraction(input_b,training=is_training, endpoint_name = 'Mixed_6e')
        
        feat_map_a = stem_block(feat_map_a,embedding_length, training=is_training, scope='stem_embedding')
        feat_map_b = stem_block(feat_map_b,embedding_length, training=is_training, scope='stem_embedding')
        
        print(feat_map_a.shape)
        # location embedding
        feat_map_a = add_timing_signal_nd(feat_map_a)
        feat_map_b = add_timing_signal_nd(feat_map_b)
        
        # ATTENTION
        A2AAtt,attentions_1 = fn_coattention(feat_map_a,feat_map_a,feat_map_a,
                                             FLAGS.num_heads,training=is_training,evaluation=is_evaluation,scope='SelfAtt1')
        A2BAtt,attentions_2 = fn_coattention(feat_map_a,feat_map_b,feat_map_b,
                                             FLAGS.num_heads,training=is_training,evaluation=is_evaluation,scope='CoAtt1')
        
        A2AAtt = tf.reshape(A2AAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        A2BAtt = tf.reshape(A2BAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        # COMPARE
        compare_map_ab = compare_block(A2AAtt,A2BAtt,training=is_training,scope='compare',reuse=False)

        # ATTENTION, reuse = True
        B2BAtt,attentions_3 = fn_coattention(feat_map_b,feat_map_b,feat_map_b,FLAGS.num_heads,
                                             training=is_training,evaluation=is_evaluation,reuse=True, scope='SelfAtt1')
        B2AAtt,attentions_4 = fn_coattention(feat_map_b,feat_map_a,feat_map_a,FLAGS.num_heads,
                                             training=is_training,evaluation=is_evaluation,reuse=True, scope='CoAtt1')
        
        B2BAtt = tf.reshape(B2BAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        B2AAtt = tf.reshape(B2AAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        # COMPARE
        compare_map_ba = compare_block(B2BAtt,B2AAtt,training=is_training,scope='compare',reuse=True)   
        

        # Compare Final
        compmap = compare_block(compare_map_ab,compare_map_ba,training=is_training,scope='compare-final',reuse=False)  
        score = score_block(compmap,training=is_training)
        
        # summary
        
        attentions_1 = tf.reshape(attentions_1, [attentions_1.get_shape()[0],attentions_1.get_shape()[1],attentions_1.get_shape()[2],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2] ])
        
        slice_a = tf.slice(attentions_1, [0,0,31,0,0],[-1,1,1,-1,-1])
        slice_a = tf.squeeze(slice_a)
        tf.summary.image('attentions_5_3_1',tf.expand_dims(slice_a,axis=3))
        slice_a = tf.slice(attentions_1, [0,0,39,0,0],[-1,1,1,-1,-1])
        slice_a = tf.squeeze(slice_a)
        tf.summary.image('attentions_6_4_1',tf.expand_dims(slice_a,axis=3))
        
        attentions_2 = tf.reshape(attentions_2, [attentions_2.get_shape()[0],attentions_2.get_shape()[1],attentions_2.get_shape()[2],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2] ])
        
        slice_a = tf.slice(attentions_2, [0,0,31,0,0],[-1,1,1,-1,-1])
        slice_a = tf.squeeze(slice_a)
        tf.summary.image('attentions_5_3_1_b',tf.expand_dims(slice_a,axis=3))
        slice_a = tf.slice(attentions_2, [0,0,39,0,0],[-1,1,1,-1,-1])
        slice_a = tf.squeeze(slice_a)
        tf.summary.image('attentions_6_4_1_b',tf.expand_dims(slice_a,axis=3))
        
        tf.summary.image('feat_map_a',tf.subtract(tf.multiply(input_a, 0.5),-0.5))
        tf.summary.image('feat_map_b',tf.subtract(tf.multiply(input_b, 0.5),-0.5))
        
        return score
    
class CoAttentionBaseModel_v2(BaseModel):
    def create_model(self, input_a, input_b, reuse, is_training=True,is_evaluation=False, **unused_params):
        
        fn_extraction = FeatureExtractor_inv1(reuse = reuse)
        stem_block = StemBlock(reuse = reuse)
        attention_block = AttentionBlock(reuse = reuse)
        query_block = QueryBlock(reuse = reuse)
        compare_block = CompareBlock(reuse = reuse)
        classification_block = ClassificationBlock(reuse = reuse)
        fn_coattention = MultiHeadAttentionBlock_v2(reuse = reuse)
        
        embedding_length = FLAGS.feature_dim
        
        feat_map_a = fn_extraction(input_a,training=is_training, endpoint_name = 'Mixed_4f')
        feat_map_b = fn_extraction(input_b,training=is_training, endpoint_name = 'Mixed_4f')
        
        feat_map_a = stem_block(feat_map_a,embedding_length, scope='stem_embedding')
        feat_map_b = stem_block(feat_map_b,embedding_length, scope='stem_embedding')
        
        # location embedding
        feat_map_a = add_timing_signal_nd(feat_map_a)
        feat_map_b = add_timing_signal_nd(feat_map_b)
        
        # ATTENTION
        preAttention = None
        compare_maps = []
        attention_maps_a = []

        
        for attenIdx in range(FLAGS.num_heads):
            # ATTENTION
            attention_a = attention_block(feat_map_a,preAttention,scope='attention-'+str(attenIdx),reuse=False)
            # QUERY
            query_a = query_block(feat_map_a,attention_a,scope='query-'+str(attenIdx),reuse=False) 
            query_a_vec = tf.reduce_max(tf.reduce_max(query_a,axis=1),axis=1)
            query_a_vec = tf.expand_dims(tf.expand_dims(query_a_vec,axis=1),axis=1)
            query_a_tile = tf.tile(query_a_vec,[1,query_a.get_shape().as_list()[1],query_a.get_shape().as_list()[2],1])
            concated = tf.concat([query_a_tile,feat_map_b],axis=3)
            
            # ATTENTION
            attention_b = attention_block(concated,preAttention,scope='attentionB-'+str(attenIdx),reuse=False)
            # QUERY
            query_b = query_block(feat_map_b,attention_b,scope='query-'+str(attenIdx),reuse=True)

            # COMPARE
            compare_map = compare_block(query_a,query_b,scope='compare-'+str(attenIdx),reuse=False) # share or not
            tf.summary.image('attention_a'+str(attenIdx),attention_a)
            tf.summary.image('attention_b'+str(attenIdx),attention_b)
            compare_maps.append(compare_map)

        
        comp_a = tf.concat(compare_maps[0:FLAGS.num_heads/2],axis=3)
        comp_b = tf.concat(compare_maps[FLAGS.num_heads/2:],axis=3)
        comp2nd = compare_block(comp_a,comp_b,scope='compare2nd',reuse=False)

        
        # CLASSIFICATION
        score = classification_block(comp2nd)
        
        tf.summary.image('feat_map_a',tf.subtract(tf.multiply(input_a, 0.5),-0.5))
        tf.summary.image('feat_map_b',tf.subtract(tf.multiply(input_b, 0.5),-0.5))
        
        return score
    
class MultiHeadAttentionBaseModel_set_share_res50(BaseModel):
    def create_model(self, input_a, input_b, reuse, is_training=True,is_evaluation=False, **unused_params):
        
        fn_extraction = FeatureExtractor_res50(reuse = reuse)
        stem_block = StemBlock(reuse = reuse)
        attention_block = AttentionBlock(reuse = reuse)
        query_block = QueryBlock(reuse = reuse)
        compare_block = CompareBlock(reuse = reuse)
        score_block = ScoreBlock(reuse = reuse)
        fn_coattention = MultiHeadAttentionBlock_v2(reuse = reuse)
        
        embedding_length = FLAGS.feature_dim
        
        feat_map_a = fn_extraction(input_a,reuse=False,training=is_training)
        feat_map_b = fn_extraction(input_b,reuse=True,training=is_training)
        
        feat_map_a = stem_block(feat_map_a,embedding_length, training=is_training, scope='stem_embedding')
        feat_map_b = stem_block(feat_map_b,embedding_length, training=is_training, scope='stem_embedding')
        
        # location embedding
        feat_map_a = add_timing_signal_nd(feat_map_a)
        feat_map_b = add_timing_signal_nd(feat_map_b)
        
        # ATTENTION
        A2AAtt,attentions_1 = fn_coattention(feat_map_a,feat_map_a,feat_map_a,FLAGS.num_heads,
                                             training=is_training,evaluation=is_evaluation,scope='CoAtt1')
        A2BAtt,attentions_2 = fn_coattention(feat_map_a,feat_map_b,feat_map_b,FLAGS.num_heads,
                                             training=is_training,evaluation=is_evaluation,reuse=True, scope='CoAtt1')
        
        A2AAtt = tf.reshape(A2AAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        A2BAtt = tf.reshape(A2BAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        # COMPARE
        compare_map_ab = compare_block(A2AAtt,A2BAtt,training=is_training,scope='compare',reuse=False)

        # ATTENTION, reuse = True
        B2BAtt,attentions_3 = fn_coattention(feat_map_b,feat_map_b,feat_map_b,FLAGS.num_heads,
                                             training=is_training,evaluation=is_evaluation,reuse=True, scope='CoAtt1')
        B2AAtt,attentions_4 = fn_coattention(feat_map_b,feat_map_a,feat_map_a,FLAGS.num_heads,
                                             training=is_training,evaluation=is_evaluation,reuse=True, scope='CoAtt1')
        
        B2BAtt = tf.reshape(B2BAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        B2AAtt = tf.reshape(B2AAtt, [feat_map_a.get_shape()[0],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2],-1])
        # COMPARE
        compare_map_ba = compare_block(B2BAtt,B2AAtt,training=is_training,scope='compare',reuse=True)   
        

        # Compare Final
        compmap = compare_block(compare_map_ab,compare_map_ba,training=is_training,scope='compare-final',reuse=False)  
        score = score_block(compmap,training=is_training)
        
        # summary
        if not is_evaluation:
            attentions_1 = tf.reshape(attentions_1, [attentions_1.get_shape()[0],attentions_1.get_shape()[1],attentions_1.get_shape()[2],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2] ])
            slice_a = tf.slice(attentions_1, [0,0,15,0,0],[-1,1,1,-1,-1])
            slice_a = tf.squeeze(slice_a)
            tf.summary.image('attentions_5_3_1_a',tf.expand_dims(slice_a,axis=3))
            slice_a = tf.slice(attentions_1, [0,0,18,0,0],[-1,1,1,-1,-1])
            slice_a = tf.squeeze(slice_a)
            tf.summary.image('attentions_6_4_1_a',tf.expand_dims(slice_a,axis=3))

            attentions_2 = tf.reshape(attentions_2, [attentions_2.get_shape()[0],attentions_2.get_shape()[1],attentions_2.get_shape()[2],feat_map_a.get_shape()[1],feat_map_a.get_shape()[2] ])

            slice_a = tf.slice(attentions_2, [0,0,15,0,0],[-1,1,1,-1,-1])
            slice_a = tf.squeeze(slice_a)
            tf.summary.image('attentions_5_3_1_b',tf.expand_dims(slice_a,axis=3))
            slice_a = tf.slice(attentions_2, [0,0,18,0,0],[-1,1,1,-1,-1])
            slice_a = tf.squeeze(slice_a)
            tf.summary.image('attentions_6_4_1_b',tf.expand_dims(slice_a,axis=3))

            tf.summary.image('feat_map_a',tf.subtract(tf.multiply(input_a, 0.5),-0.5))
            tf.summary.image('feat_map_b',tf.subtract(tf.multiply(input_b, 0.5),-0.5))
            return score
        else:
            attentions = []
            attentions.append(attentions_1)
            attentions.append(attentions_2)
            attentions.append(attentions_3)
            attentions.append(attentions_4)
            return score,attentions 