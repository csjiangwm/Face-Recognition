# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 15:06:15 2018

@author: jwm
"""

import tensorflow as tf
from utils.config import process_config
from utils.utils import load_data,load_model
from utils import lfw
import numpy as np
import os
import math
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate

def main(config):
    
    with tf.Graph().as_default():
        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pairs = lfw.read_pairs(os.path.expanduser(config.lfw.lfw_pairs_path)) # Read the file containing the pairs used for testing
            paths, actual_issame = lfw.get_paths(config.input_dir, pairs, 'png') # Get the paths for the corresponding images
            load_model(config.lfw.valid_model_path) # Load the model
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0") 
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            image_size = config.lfw.image_size
            embedding_size = embeddings.get_shape()[1]
            
            print('Runnning forward pass on LFW images')
            batch_size = config.batch_size
            nrof_images = len(paths)
            nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches):
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = load_data(paths_batch, image_size, False, False)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
        
            tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(emb_array, actual_issame, nrof_folds=config.lfw.lfw_nrof_folds)

            print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
            print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

            auc = metrics.auc(fpr, tpr)
            print('Area Under Curve (AUC): %1.3f' % auc)
            eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
            print('Equal Error Rate (EER): %1.3f' % eer)

if __name__ == '__main__':
    try:
        config = process_config()
    except:
        print("missing or invalid arguments")
        exit(0)
    main(config)