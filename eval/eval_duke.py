import numpy as np
import tensorflow as tf

import sys
sys.path.append("../slim/")
sys.path.append("..")
from tensorflow.contrib import slim
from core import models
FLAGS = tf.app.flags.FLAGS
import os
from sklearn.metrics import average_precision_score

tf.app.flags.DEFINE_integer('start_id',0, '')
tf.app.flags.DEFINE_integer('end_id',100, '')
FLAGS = tf.app.flags.FLAGS

def find_class_by_name(name, modules):
    """Searches the provided modules for the named class and returns it."""
    modules = [getattr(module, name, None) for module in modules]
    return next(a for a in modules if a)




    
class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that converts CMYK JPEG data to RGB JPEG data.
    self._cmyk_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
    self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._encode_jpeg_data = tf.placeholder(dtype=tf.uint8)
    self._width = tf.placeholder(dtype=tf.int32)
    self._height = tf.placeholder(dtype=tf.int32)
    self._image_data = tf.placeholder(shape=[None,None,3],dtype=tf.uint8)
    self._image_data_gray = tf.placeholder(shape=[None,None,14],dtype=tf.float32)

    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
    self._decode_jpeg_gray = tf.image.decode_jpeg(self._decode_jpeg_data, channels=1)
    self._encode_jpeg = tf.image.encode_jpeg(self._encode_jpeg_data, format='rgb', quality=100)
    self._encode_jpeg_gray = tf.image.encode_jpeg(self._encode_jpeg_data, format='grayscale', quality=100)
    
    self._resize_image = tf.image.resize_images(self._image_data,[self._height, self._width])
    self._resize_image_crop_pad = tf.image.resize_image_with_crop_or_pad(self._image_data, FLAGS.target_height,FLAGS.target_width)

    self._resize_image_gray = tf.image.resize_images(self._image_data_gray,[self._height, self._width])
    self._resize_image_crop_pad_gray = tf.image.resize_image_with_crop_or_pad(self._image_data_gray, FLAGS.target_height,FLAGS.target_width)
    
    # for string ops
  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def cmyk_to_rgb(self, image_data):
    return self._sess.run(self._cmyk_to_rgb,
                          feed_dict={self._cmyk_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

  def decode_jpeg_gray(self, image_data):
    image = self._sess.run(self._decode_jpeg_gray,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 1
    return image

  def resize_image(self, image, new_height, new_width):
    if image.shape[2] == 3:
        resized_image = self._sess.run(self._resize_image, 
                                       feed_dict={self._image_data: image, self._height: new_height, self._width: new_width})
    else:
         resized_image = self._sess.run(self._resize_image_gray, 
                                       feed_dict={self._image_data_gray: image, self._height: new_height, self._width: new_width})       
    return resized_image

  def resize_image_crop_pad(self, image):
    if image.shape[2] == 3:
        resized_image = self._sess.run(self._resize_image_crop_pad, feed_dict={self._image_data: image})
    else:
        resized_image = self._sess.run(self._resize_image_crop_pad_gray, feed_dict={self._image_data_gray: image})
    return resized_image

  def encode_jpeg(self, image):
    if image.shape[2] == 3:
        image_data = self._sess.run(self._encode_jpeg,
                               feed_dict={self._encode_jpeg_data: image})
    else:
        image_data = self._sess.run(self._encode_jpeg_gray,
                               feed_dict={self._encode_jpeg_data: image})        

    return image_data
def readList(list_name): 
    import random
    import os
    file_object = open(list_name)
    try:
        all_the_text = file_object.read()
    finally:
        file_object.close()

    lines = all_the_text.split('\n')
    #print all_the_text
    DATA_DIR='../dataset/cuhk03/cuhk03_release/'
    probes=[]
    gallerys=[]
    for filename in lines:
        if filename!='' :
            campair_no = int(filename.split(',')[0])
            person_id = int(filename.split(',')[1])
            #print int(campair_no)
            while True:
                probe_no=random.randint(1,5)
                probe_filename = DATA_DIR +  'data/campair_%d/'%campair_no + '%02d_%04d_%02d.jpg'%(campair_no,person_id,probe_no)
                if os.path.isfile(probe_filename):
                    probes.append(probe_filename)
                    break
            while True:
                gallery_no=random.randint(6,10)  
                gallery_filename = DATA_DIR +  'data/campair_%d/'%campair_no + '%02d_%04d_%02d.jpg'%(campair_no,person_id,gallery_no)
                if os.path.isfile(gallery_filename):
                    gallerys.append(gallery_filename)
                    break
    if len(probes)!=len(gallerys):
        print('something wrong! list length does not match!/n')
        return 0
    else:
        return probes,gallerys
    
def process_image_np(filename,coder):
    image_data = tf.gfile.FastGFile(filename, 'r').read()
    image_data = coder.decode_jpeg(image_data)

    height,width,col = image_data.shape
    resized_image = coder.resize_image(image_data, FLAGS.target_height, FLAGS.target_width).astype(np.uint8)
    return resized_image

def generateScoreList(probes,gallerys):
    

    

    
#     assert False
    scoreList=[]
    from time import clock
    start=clock()  
    coder = ImageCoder()
    #process each probe
    for probeIdx in range(len(probes)):
        probeName=probes[probeIdx]
        I_probe = process_image_np(probeName,coder)
        gallery = gallerys
        I_gallery = []
        for gallerypath in gallery:
            I_gallery.append(process_image_np(gallerypath,coder))
        I_gallery = np.stack(I_gallery,axis=0)
        [probe_features] = sess_feature_extract.run([logits],feed_dict={images_a :np.tile(np.asarray([I_probe]),[FLAGS.batch_size,1,1,1])})
        [gallery_features] = sess_feature_extract.run([logits],feed_dict={images_a :I_gallery})
        
        [scores] = sess_match.run([logits_match],feed_dict={feature_map_a :probe_features, feature_map_b:gallery_features})

#         print(scores[:,0][0:5])
#         print(outputs[:,0][0:5])
#         assert False
#         assert False
        
#         print(images_a_data[0,0,0,:])
#         print(images_b_data[0,0,0,:])
#         similarScore=outputs[:,0]
#         #scoreList.append each probe score
#         scoreList.append(similarScore.tolist())
# #         print(probeIdx,np.argsort(similarScore)[::-1][0])
#         if (probeIdx+1)%10==0:
#             sys.stdout.write('\r%3d/%d, '%(probeIdx+1,len(probes))+probeName)
#             sys.stdout.flush()
    #we get scoreList, then cal predictLists
    predictLists=[]
    for score in scoreList:
        probeRankList=np.argsort(score)[::-1]
        predictLists.append(probeRankList)
    finish=clock()
    print('\r  Processing %dx%d pairs cost %f second time'%(len(probes),len(gallerys),(finish-start)))
    return scoreList,predictLists

def evaluateCMC(gtLabels,predictLists):
    N=len(gtLabels)
    R=len(predictLists[0])
    histogram=np.zeros(N)
    for testIdx in range(N):
        for rankIdx in range(R):
            histogram[rankIdx]+=1*(predictLists[testIdx][rankIdx]==gtLabels[testIdx])    #1*(true or false)=1 or 0
    cmc=np.cumsum(histogram)
    return cmc/N
def calCMC(set_no,rand_times=10):
    coder = ImageCoder()
    # load feature extraction model
    with tf.Graph().as_default() as graph_feature_extract:
        tf_global_step = slim.get_or_create_global_step()
        
        images_a = tf.placeholder(dtype=tf.float32,shape=(None, FLAGS.target_height, FLAGS.target_width, 3))
        images_a_norm = tf.divide(images_a, 255)
        images_a_norm = tf.subtract(images_a_norm, 0.5)
        images_a_norm = tf.multiply(images_a_norm, 2.0)

        model = find_class_by_name(FLAGS.model_feature_extract, [models])()

        logits = model.create_model(images_a_norm, reuse=False, is_training = False) 
        variables_to_restore = tf.global_variables()
        info = variables_to_restore[0]
        if not info.name.startswith('global_step'):
            global_step = slim.get_or_create_global_step()
            variables_to_restore.append(global_step)

        saver_feature_extract = tf.train.Saver(variables_to_restore)
    sess_feature_extract = tf.Session("", graph=graph_feature_extract)
    saver_feature_extract.restore(sess_feature_extract,FLAGS.weights)
    # load match model
    with tf.Graph().as_default() as graph_match:
        tf_global_step = slim.get_or_create_global_step()
        
        feature_map_a = tf.placeholder(dtype=tf.float32,shape=(FLAGS.batch_size, 14, 7, FLAGS.feature_dim))
        feature_map_b = tf.placeholder(dtype=tf.float32,shape=(FLAGS.batch_size, 14, 7, FLAGS.feature_dim))
        model = find_class_by_name(FLAGS.model_match, [models])()

        logits_match = model.create_model(feature_map_a,feature_map_b, reuse=False, is_training = False) 

        variables_to_restore_match = tf.global_variables()
        info = variables_to_restore_match[0]
        if not info.name.startswith('global_step'):
            global_step_match = slim.get_or_create_global_step()
            variables_to_restore_match.append(global_step_match)

        saver_match = tf.train.Saver(variables_to_restore_match)
    sess_match = tf.Session("", graph=graph_match)
    saver_match.restore(sess_match,FLAGS.weights)
    
    # datasets
    DATA_DIR= '../dataset/DukeMTMC-reID/bounding_box_test/'
    file_list_a=os.listdir(DATA_DIR)
    
    testID_list = []
    testCAM_list = []
    testImages = []
    for filename in file_list_a:
        if filename[-3:]=='jpg':
            testImages.append(DATA_DIR + filename)
            if filename[0]=='-':
                testID_list.append(-1)
                testCAM_list.append(int(filename[4]))
            else:
                testID_list.append(int(filename[0:4]))
                testCAM_list.append(int(filename[6]))
    assert len(testID_list)==len(testCAM_list)
    assert len(testCAM_list)==len(testImages)
    
    queryID_list = []
    queryCAM_list = []
    queryImages = []
    DATA_DIR_query= '../dataset/DukeMTMC-reID/query/'
    file_list_b=os.listdir(DATA_DIR_query)
    for filename in file_list_b:
        if filename[-3:]=='jpg':
            queryImages.append(DATA_DIR_query + filename)
            if filename[0]=='-':
                queryID_list.append(-1)
                queryCAM_list.append(int(filename[4]))
            else:
                queryID_list.append(int(filename[0:4]))
                queryCAM_list.append(int(filename[6]))
    assert len(queryCAM_list)==len(queryID_list)
    assert len(queryID_list)==len(queryImages)  
    
    # extract test features
    testFeature_list = []
    num_batches = int(np.ceil((len(testImages)*1.0)/FLAGS.batch_size))
    for batchIdx in range(num_batches):
        this_batch = testImages[batchIdx*FLAGS.batch_size:(batchIdx+1)*FLAGS.batch_size]
        I_gallery = []
        for path in this_batch:
            I_gallery.append(process_image_np(path,coder))
        I_gallery = np.stack(I_gallery,axis=0)
        [gallery_features] = sess_feature_extract.run([logits],feed_dict={images_a :I_gallery})
        for Idx in range(gallery_features.shape[0]):
            testFeature_list.append(gallery_features[Idx])
    # extract query features
    queryFeature_list = []
    num_batches = int(np.ceil((len(queryImages)*1.0)/FLAGS.batch_size))
    for batchIdx in range(num_batches):
        this_batch = queryImages[batchIdx*FLAGS.batch_size:(batchIdx+1)*FLAGS.batch_size]
        I_gallery = []
        for path in this_batch:
            I_gallery.append(process_image_np(path,coder))
        I_gallery = np.stack(I_gallery,axis=0)
        [gallery_features] = sess_feature_extract.run([logits],feed_dict={images_a :I_gallery})
        for Idx in range(gallery_features.shape[0]):
            queryFeature_list.append(gallery_features[Idx])
            
    # match over queries
    hit = 0
    total = 0
    aps = []
    start = FLAGS.start_id
    end = FLAGS.end_id
    test_num = end - start
    
    for testIdx in range(test_num):
        queryIdx = testIdx + start
        
        this_score_list = []
#         print(queryImages[queryIdx])
        query_feature = queryFeature_list[queryIdx]
        num_batches = int(np.ceil((len(testImages)*1.0)/FLAGS.batch_size))
        max_Idx = 0
        max_value = -10
        y_true = []
        y_score = []
        
        for batchIdx in range(num_batches):
            gallery_features_list = testFeature_list[batchIdx*FLAGS.batch_size:(batchIdx+1)*FLAGS.batch_size]
            gallery_features = np.stack(gallery_features_list,axis=0)
            shape = gallery_features.shape
            if gallery_features.shape[0] < FLAGS.batch_size:
                gallery_features_pad = np.zeros((FLAGS.batch_size,shape[1],shape[2],shape[3])).astype(np.float32)
                gallery_features_pad[0:len(gallery_features_list)] = gallery_features
            else:
                gallery_features_pad = gallery_features
                
            [scores] = sess_match.run([logits_match],feed_dict={feature_map_a :np.tile(np.asarray([query_feature]),[FLAGS.batch_size,1,1,1]), feature_map_b:gallery_features_pad})
            

            
            for Idx in range(len(gallery_features_list)):
                this_score_list.append(scores[Idx,0])
                valid = ((queryCAM_list[queryIdx]!=testCAM_list[batchIdx*FLAGS.batch_size+Idx]) | (queryID_list[queryIdx]!=testID_list[batchIdx*FLAGS.batch_size+Idx])) and testID_list[batchIdx*FLAGS.batch_size+Idx]!=-1
                if valid:
                    y_true.append(testID_list[batchIdx*FLAGS.batch_size+Idx]==queryID_list[queryIdx])
                    y_score.append(scores[Idx,0])
                    if scores[Idx,0] > max_value:
                        max_Idx = batchIdx*FLAGS.batch_size+Idx
                        max_value = scores[Idx,0]
#         print(max_value)
#         print(queryID_list[queryIdx])
#         print(testID_list[max_Idx])
        if (testID_list[max_Idx]==queryID_list[queryIdx]):
            hit+=1
        total+=1
        aps.append(average_precision_score(np.asarray(y_true), np.asarray(y_score)))
        print('\r%d/%d, top-1: %.6f, mAP: %.6f'%(total,test_num,hit*1.0/total,np.mean(aps)))
#         sys.stdout.write('\r%d/%d, %.4f'%(total,len(queryImages),hit*1.0/total))
#         sys.stdout.flush()
        
#         print(total,hit*1.0/total)
#         assert False
    
    print list_name+'\n'
    #rand 10 times for stable result
    cmc_list=[]
    for i in range(rand_times):
        print 'Round %d with rand list:'%i
        probes,gallerys=readList(list_name)
        scoreList,predictLists=generateScoreList(probes,gallerys)
        gtLabels=range(len(probes))
        cmc=evaluateCMC(gtLabels,predictLists)
        cmc_list.append(cmc)
    return np.average(cmc_list,axis=0)

def getCVPRcmc():
    #return the cmc values, 100 dim vetor
    import numpy as np
    cmcIndex=[0,4,8,12,16,21,25,29,33,37,41,45,49,53]
    cmcOfCVPRImproved=[0.5474,0.8753,0.9293,0.9712,0.9764,0.9811,0.9899,0.9901,0.9912,0.9922,0.9937,0.9945,0.9951,1]
    pOfCVPRImproved = np.poly1d(np.polyfit(cmcIndex,cmcOfCVPRImproved,10))
    x_line=range(50)
    cmc=pOfCVPRImproved(x_line)
    return cmc

def plotCMC(cmcDict,pathname):
    import matplotlib.pyplot as plt
    get_ipython().magic(u'matplotlib inline')   
    from matplotlib.legend_handler import HandlerLine2D
    import numpy as np

    #plot the cmc curve, record CVPR from the pdf paper.cmc[0,4,8,12,16,21,25,29,33,37,41,45,49]
    rank2show=25
    rankStep=1
    cmcIndex=np.arange(0,rank2show,rankStep)   #0,5,10,15,20,25

    colorList=['rv-','g^-','bs-','yp-','c*-','mv-','kd-','gs-','b^-']
    #start to plot
    plt.ioff()
    fig = plt.figure(figsize=(6,5),dpi=180)
    sortedCmcDict = sorted(cmcDict.items(), key=lambda (k, v): v[1])[::-1]
    for idx in range(len(sortedCmcDict)):
        cmc_dictList=sortedCmcDict[idx]
        cmc_name=cmc_dictList[0]
        cmc_list=cmc_dictList[1]
        #print cmc_name,": ",cmc_list
        #x for plot
        x_point=[item+1 for item in cmcIndex]
        x_line=range(rank2show)
        x_plot=[temp+1 for temp in x_line]
        #start plot
        plt.plot(x_plot, cmc_list[x_line],colorList[idx],label="%02.02f%% %s"%(100*cmc_list[0],cmc_name))
        plt.plot(x_point,cmc_list[cmcIndex],colorList[idx]+'.')
        #plt.legend(loc=4,handler_map={line: HandlerLine2D(numpoints=1)})
        #idx of color +1
        idx+=1
    #something to render

    plt.xlabel('Rank')
    plt.ylabel('Identification Rate')
    plt.xticks(np.arange(0,rank2show+1,5))
    plt.yticks(np.arange(0,1.01,0.1))
    plt.grid()
    plt.legend(loc=4)
    plt.savefig(pathname)
    plt.show()

    #end of show
    
def main():
    test_list=range(3,4) #use set 1-10 for test (total 20)
    cmc_list=[]
    for set_no in test_list:
        #init net
        MODEL_FILE = '../../experiments/reid_simplemodel/set%02d/'%(set_no)+'deploy.prototxt'
        PRETRAINED = '../../experiments/reid_simplemodel/set%02d/'%(set_no)+'Snapshots/set%02d_iter_120000.caffemodel'%(set_no)
        caffe.set_device(0)
        caffe.set_mode_gpu()
        net = caffe.Classifier(MODEL_FILE, PRETRAINED,caffe.TEST)
        #caculate CMC
        cmc=calCMC(net,set_no,rand_times=10)
        cmc_list.append(cmc)
    cmc_all=np.average(cmc_list,axis=0)
    print('\nCMC from rank 1 to rank %d:'%(len(cmc_all)))
    print(cmc_all)
    plotCMC(cmc)
    
if __name__ == '__main__':
    main()