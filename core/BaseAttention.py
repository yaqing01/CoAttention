from core.__init_paths import *
tf.app.flags.DEFINE_float('l1_loss', 2.5e-07, 'the number of heads')

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

class AttentionBlock:
    def __init__(self, reuse=False):
        self.reuse = reuse

    def __call__(self, net, atten=None, reuse=False, training=False, scope = ""):
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope(use_batch_norm=True,weight_decay=FLAGS.weight_decay)):
            with tf.variable_scope(scope, reuse=reuse):
                with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=training):
                    feature_len = net.get_shape().as_list()[-1]
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
                    feature_len = net_a.get_shape().as_list()[-1]
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
                    feature_len = net.get_shape().as_list()[-1]
                    
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
                    feature_len = net.get_shape().as_list()[-1]
                    
                    net = slim.flatten(net)
                    net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='Match_Prelogits1')
                    net = slim.dropout(net, 0.50, scope='Dropout_match',is_training=training)
                    net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='Match_Prelogits2')
                    net = slim.dropout(net, 0.50, scope='Dropout_match',is_training=training)
                    net = slim.fully_connected(net, 2, activation_fn=None, scope='Match_logits')
                    
            self.reuse=True
        return net
class AttentionBaseModel(BaseModel):
    def create_model(self, input_a, input_b, reuse, is_training=True,is_evaluation=False, **unused_params):
        
        fn_extraction = FeatureExtractor_inv1(reuse = reuse)
        stem_block = StemBlock(reuse = reuse)
        attention_block = AttentionBlock(reuse = reuse)
        query_block = QueryBlock(reuse = reuse)
        compare_block = CompareBlock(reuse = reuse)
        classification_block = ClassificationBlock(reuse = reuse)
        
        embedding_length = 256
        
        feat_map_a = fn_extraction(input_a,training=is_training, endpoint_name = 'Mixed_4f')
        feat_map_b = fn_extraction(input_b,training=is_training, endpoint_name = 'Mixed_4f')
        
        feat_map_a = stem_block(feat_map_a,embedding_length, scope='stem_embedding')
        feat_map_b = stem_block(feat_map_b,embedding_length, scope='stem_embedding')
        
        # ATTENTION
        preAttention = None
        for attenIdx in range(1):
            attention_a = attention_block(feat_map_a,preAttention,scope='attention-'+str(attenIdx),reuse=False)
            attention_b = attention_block(feat_map_b,preAttention,scope='attention-'+str(attenIdx),reuse=True)
        
        # QUERY
        feat_map_a = query_block(feat_map_a,attention_a,scope='query',reuse=False)  
        feat_map_b = query_block(feat_map_b,attention_b,scope='query',reuse=True)
        
        # L1 reg
        l1_a = tf.reduce_sum(attention_a)
        l1_b = tf.reduce_sum(attention_b)
        tf.losses.add_loss(l1_a*FLAGS.l1_loss)
        tf.losses.add_loss(l1_b*FLAGS.l1_loss)
        
        # COMPARE
        compare_map = compare_block(feat_map_a,feat_map_b,scope='compare',reuse=False)
        
        # CLASSIFICATION
        score = classification_block(compare_map)
        
        tf.summary.image('feat_map_a',tf.subtract(tf.multiply(input_a, 0.5),-0.5))
        tf.summary.image('attention_a',attention_a)
        tf.summary.image('feat_map_b',tf.subtract(tf.multiply(input_b, 0.5),-0.5))
        tf.summary.image('attention_b',attention_b)
        return score
 
class CoAttentionBaseModel(BaseModel):
    def create_model(self, input_a, input_b, reuse, is_training=True,is_evaluation=False, **unused_params):
        
        fn_extraction = FeatureExtractor_inv1(reuse = reuse)
        stem_block = StemBlock(reuse = reuse)
        attention_block = AttentionBlock(reuse = reuse)
        query_block = QueryBlock(reuse = reuse)
        compare_block = CompareBlock(reuse = reuse)
        classification_block = ClassificationBlock(reuse = reuse)
        
        embedding_length = 256
        
        feat_map_a = fn_extraction(input_a,training=is_training, endpoint_name = 'Mixed_4f')
        feat_map_b = fn_extraction(input_b,training=is_training, endpoint_name = 'Mixed_4f')
        
        feat_map_a = stem_block(feat_map_a,embedding_length, scope='stem_embedding')
        feat_map_b = stem_block(feat_map_b,embedding_length, scope='stem_embedding')
        
        # ATTENTION
        preAttention = None
        compare_map = compare_block(feat_map_a,feat_map_b,scope='precompare',reuse=False)
        attention_a = attention_block(compare_map,preAttention,scope='attention-a',reuse=False)
        attention_b = attention_block(compare_map,preAttention,scope='attention-b',reuse=False)
        
        # QUERY
        feat_map_a = query_block(feat_map_a,attention_a,scope='query',reuse=False)  
        feat_map_b = query_block(feat_map_b,attention_b,scope='query',reuse=True)
        
        # L1 reg
        l1_a = tf.reduce_sum(attention_a)
        l1_b = tf.reduce_sum(attention_b)
        tf.losses.add_loss(l1_a*FLAGS.l1_loss)
        tf.losses.add_loss(l1_b*FLAGS.l1_loss)
        
        # COMPARE
        compare_map = compare_block(feat_map_a,feat_map_b,scope='compare',reuse=False)
        
        # CLASSIFICATION
        score = classification_block(compare_map)
        
        tf.summary.image('feat_map_a',tf.subtract(tf.multiply(input_a, 0.5),-0.5))
        tf.summary.image('attention_a',attention_a)
        tf.summary.image('feat_map_b',tf.subtract(tf.multiply(input_b, 0.5),-0.5))
        tf.summary.image('attention_b',attention_b)
        return score

class ParallelAttentionBaseModel_v2(BaseModel):
    def create_model(self, input_a, input_b, reuse, is_training=True,is_evaluation=False, **unused_params):
        
        fn_extraction = FeatureExtractor_inv1(reuse = reuse)
        stem_block = StemBlock(reuse = reuse)
        attention_block = AttentionBlock(reuse = reuse)
        query_block = QueryBlock(reuse = reuse)
        compare_block = CompareBlock(reuse = reuse)
        classification_block = ClassificationBlock(reuse = reuse)
        
        embedding_length = 256
        
        feat_map_a = fn_extraction(input_a,training=is_training, endpoint_name = 'Mixed_4f')
        feat_map_b = fn_extraction(input_b,training=is_training, endpoint_name = 'Mixed_4f')
        
        feat_map_a = stem_block(feat_map_a,embedding_length, scope='stem_embedding')
        feat_map_b = stem_block(feat_map_b,embedding_length, scope='stem_embedding')
        
        # ATTENTION
        preAttention = None
        compare_maps = []
        attention_maps_a = []
        attention_maps_b = []
        if FLAGS.num_heads==2:
            slice_list = [0,7]
            rows_list = [7,7]
        if FLAGS.num_heads==4:
            slice_list = [0,3,7,10]
            rows_list = [4,4,4,4]
        
        for attenIdx in range(FLAGS.num_heads):
            attention_a = attention_block(feat_map_a,preAttention,scope='attention-'+str(attenIdx),reuse=False)
            attention_b = attention_block(feat_map_b,preAttention,scope='attention-'+str(attenIdx),reuse=True)
            # QUERY
            query_a = query_block(feat_map_a,attention_a,scope='query-'+str(attenIdx),reuse=False)   
            query_b = query_block(feat_map_b,attention_b,scope='query-'+str(attenIdx),reuse=True)
            # COMPARE
            compare_map = compare_block(query_a,query_b,scope='compare-'+str(attenIdx),reuse=False)
            tf.summary.image('attention_a'+str(attenIdx),attention_a)
            tf.summary.image('attention_b'+str(attenIdx),attention_b)
            compare_maps.append(compare_map)
            attention_maps_a.append(attention_a)
            attention_maps_b.append(attention_b)
 
#             slice_a = tf.slice(attention_a, [0,slice_list[attenIdx],0,0],[-1,rows_list[attenIdx],-1,-1])
#             slice_b = tf.slice(attention_b, [0,slice_list[attenIdx],0,0],[-1,rows_list[attenIdx],-1,-1])
        
#             l1_a = tf.reduce_sum(slice_a)
#             l1_b = tf.reduce_sum(slice_b)
#             tf.losses.add_loss(l1_a*FLAGS.l1_loss)
#             tf.losses.add_loss(l1_b*FLAGS.l1_loss)
#         attention_maps_a_stack = tf.stack(attention_maps_a,axis=1)
#         attention_maps_b_stack = tf.stack(attention_maps_b,axis=1)
        
#         attention_maps_a_stack_min = tf.reduce_min(attention_maps_a_stack,axis=1)
#         attention_maps_a_stack_min_max = tf.reduce_max(tf.reduce_max(attention_maps_a_stack_min,axis=1),axis=1)
        
#         attention_maps_b_stack_min = tf.reduce_min(attention_maps_b_stack,axis=1)
#         attention_maps_b_stack_min_max = tf.reduce_max(tf.reduce_max(attention_maps_b_stack_min,axis=1),axis=1)
        
#         l1_a = tf.reduce_sum(attention_maps_a_stack_min_max)
#         l1_b = tf.reduce_sum(attention_maps_b_stack_min_max)
#         tf.losses.add_loss(l1_a*FLAGS.l1_loss)
#         tf.losses.add_loss(l1_b*FLAGS.l1_loss)

            
        pre_compmap = compare_maps[0]
        for compIdx in range(len(compare_maps)-1):
            
            this_compmap = compare_maps[compIdx+1]
            pre_compmap = compare_block(pre_compmap,this_compmap,scope='compare2nd-'+str(compIdx),reuse=False)
            
        
        
        # CLASSIFICATION
        score = classification_block(pre_compmap)
        
        tf.summary.image('feat_map_a',tf.subtract(tf.multiply(input_a, 0.5),-0.5))
        tf.summary.image('feat_map_b',tf.subtract(tf.multiply(input_b, 0.5),-0.5))
        
        return score