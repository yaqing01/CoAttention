from core.__init_paths import *

tf.app.flags.DEFINE_integer('num_layers', 4, 'the number of heads')
tf.app.flags.DEFINE_integer('num_views', 8, 'the number of heads')
FLAGS = tf.app.flags.FLAGS

class BaseModel(object):
    """Inherit from this class when implementing new models."""

    def create_model(self, unused_model_input, **unused_params):
        raise NotImplementedError()
    
class AttentionBlock:
    def __init__(self, reuse=False):
        self.reuse = reuse

    def __call__(self, Q, K, V, num_heads, training=False, scope = ""):
        """
        This module calculates the cosine distance between each of the support set embeddings and the target
        image embeddings.
        :param Q: The query, tensor of shape [batch_size, query_num, query_dim]
        :param K: The key, tensor of shape [batch_size, key_num, key_dim]
        :param V: The value, tensor of shape [batch_size, value_num, value_dim]
        :param training: Flag indicating training or evaluation (True/False)
        :return: the representation of shape [batch_size, query_num, value_dim]
        """
        
        d_model = V.get_shape().as_list()[-1]
        d_key = int(d_model / num_heads)
        d_value = int(d_model / num_heads)
        
        heads = []
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope(use_batch_norm=True,weight_decay=FLAGS.weight_decay)):
            with tf.variable_scope(scope, reuse=self.reuse):
                with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=training):
                    
                    for HeadIdx in range(num_heads):
                        query = slim.fully_connected(Q, d_key, scope='QueryRep-'+str(HeadIdx))
                        key = slim.fully_connected(K, d_key, scope='KeyRep-'+str(HeadIdx))
                        value = slim.fully_connected(V, d_value, scope='ValueRep-'+str(HeadIdx))
                        
                        head = Attention(query,key,value)
                        heads.append(head)

                    heads = tf.concat(heads,axis=2)
                    heads = slim.fully_connected(heads, d_model)
        return heads
    
class MatchNetwork_vec:
    def __init__(self, reuse=False):
        self.reuse = reuse

    def __call__(self, feat_a, feat_b, training=True):
            
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope(use_batch_norm=True,weight_decay=FLAGS.weight_decay)):
            with tf.variable_scope("", reuse=self.reuse):
                with slim.arg_scope([slim.batch_norm, slim.dropout],
                    is_training=training):
                    eps = 1e-10

                    net = tf.concat([feat_a,feat_b],axis=1)                    
                    net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='Match_Prelogits1')
                    net = slim.dropout(net, 0.50, scope='Dropout_match',is_training=training)
                    net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='Match_Prelogits2')
                    net = slim.dropout(net, 0.50, scope='Dropout_match',is_training=training)
                    net = slim.fully_connected(net, 2, activation_fn=None, scope='Match_logits')
                    self.reuse=True
        return net
class CoAttention(BaseModel):
    def create_model(self, input_a, input_b, reuse, is_training=True, **unused_params):

        fn_extraction = FeatureExtractor_inv1(reuse = reuse)
        fn_coattention = AttentionBlock(reuse = reuse)
        fn_match = MatchNetwork_vec(reuse = reuse)
        
        FMA = fn_extraction(input_a,training=is_training)
        FMB = fn_extraction(input_b,training=is_training)
        
        # location embedding
        FMA = add_timing_signal_nd(FMA)
        FMB = add_timing_signal_nd(FMB)
        
        # flatten
        FMA = tf.reshape(FMA,[FMA.get_shape().as_list()[0],-1,FMA.get_shape().as_list()[3]])
        FMB = tf.reshape(FMB,[FMB.get_shape().as_list()[0],-1,FMB.get_shape().as_list()[3]])
        
        for layerIdx in range(FLAGS.num_layers):
#             A2AAtt = fn_coattention(FMA,FMA,FMA,FLAGS.num_heads,training=is_training,scope='SelfAtt1-Layer'+str(layerIdx))
            A2BAtt = fn_coattention(FMA,FMB,FMB,FLAGS.num_heads,training=is_training,scope='CoAtt1'+str(layerIdx))
#             B2BAtt = fn_coattention(FMB,FMB,FMB,FLAGS.num_heads,training=is_training,scope='SelfAtt2'+str(layerIdx))
            B2AAtt = fn_coattention(FMB,FMA,FMA,FLAGS.num_heads,training=is_training,scope='CoAtt2'+str(layerIdx))
            FMA = A2BAtt
            FMB = B2AAtt
        
        V1 = tf.reduce_mean(FMA,axis=1)
        V2 = tf.reduce_mean(FMB,axis=1)
            
        score = fn_match(V1,V2,training=is_training)
        
        return score 

class CoAttention_onehot(BaseModel):
    def create_model(self, input_a, input_b, reuse, is_training=True, **unused_params):

        fn_extraction = FeatureExtractor_inv1(reuse = reuse)
        fn_coattention = AttentionBlock(reuse = reuse)
        fn_match = MatchNetwork_vec(reuse = reuse)
        
        batch_size = input_a.get_shape().as_list()[0]
        views = tf.range(FLAGS.num_views)
        views = tf.expand_dims(views,0)
        views = tf.tile(views, [batch_size,1])
        views = tf.one_hot(views, FLAGS.num_views)
        
        FMA = fn_extraction(input_a,training=is_training)
        FMB = fn_extraction(input_b,training=is_training)
        
        # location embedding
        FMA = add_timing_signal_nd(FMA)
        FMB = add_timing_signal_nd(FMB)
        
        # flatten
        FMA = tf.reshape(FMA,[FMA.get_shape().as_list()[0],-1,FMA.get_shape().as_list()[3]])
        FMB = tf.reshape(FMB,[FMB.get_shape().as_list()[0],-1,FMB.get_shape().as_list()[3]])
        
        # view embedding
        d_model = FMA.get_shape().as_list()[-1]
        d_key = int(d_model / FLAGS.num_heads)
        d_value = int(d_model / FLAGS.num_heads)
        
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope(use_batch_norm=True,weight_decay=FLAGS.weight_decay)):
            with tf.variable_scope('view_embed', reuse=reuse):
                with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=is_training):
                    VE = slim.fully_connected(views, d_key, activation_fn=tf.nn.relu)
        
        AATT = fn_coattention(VE,FMA,FMA,FLAGS.num_heads,training=is_training,scope='CoAttA')
        BATT = fn_coattention(VE,FMB,FMB,FLAGS.num_heads,training=is_training,scope='CoAttB')
        
        V1 = tf.reshape(AATT,[AATT.get_shape().as_list()[0],-1])
        V2 = tf.reshape(BATT,[BATT.get_shape().as_list()[0],-1])
        
        score = fn_match(V1,V2,training=is_training)
        
        return score 