# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 09:37:00 2018

@author: jwm
"""

import tensorflow as tf
import numpy as np
from utils.config import process_config
from utils.utils import load_model,load_data,flip,prewhiten
from sklearn.svm import SVC
from data_loader.data_generator import DataGenerator
import pickle
import os
import math
import cv2

def compute_rate(model,config,embeddings,images_placeholder,phase_train_placeholder,embedding_size,sess):
    err = 0.0
    total = 0.0
    HumanNames = os.listdir(config.input_dir)
    for (path,dirnames,filenames) in os.walk(config.input_dir):
        for filename in filenames:
            total += 1
            emb_array = np.zeros((1, embedding_size))
            img = cv2.imread(os.path.join(path,filename))
            img = img[:, :, 0:3]
            img = flip(img, False)
            img = cv2.resize(img, (config.lfw.image_size,config.lfw.image_size),interpolation=cv2.INTER_CUBIC)
            img = prewhiten(img)
            img = img.reshape(-1,config.lfw.image_size,config.lfw.image_size,3)
            feed_dict = {images_placeholder: img, phase_train_placeholder: False}
            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
            predictions = model.predict(emb_array)
            best_class_indices = predictions

            for H_i in HumanNames:
                if HumanNames[best_class_indices[0]] == H_i:
                    result_names = HumanNames[best_class_indices[0]]
                    print result_names,'---------',path.split('/')[-1]
                    if result_names != path.split('/')[-1]:
                        err += 1
    return 1 - err/total

def main(config):
    data = DataGenerator(config)
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            print "loading model ..."
            load_model(config.lfw.valid_model_path)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0") 
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            nrof_images = len(data.input)
            print('Calculating features for images')
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / config.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i * config.batch_size
                end_index = min((i + 1) * config.batch_size, nrof_images)
                paths_batch = data.input[start_index:end_index]
                images = load_data(paths_batch, config.lfw.image_size, True) # return 4-d tensor
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
                
            classifier_filename_exp = os.path.expanduser(config.classifier_path)
            print('Training classifier')
            model = SVC(kernel='linear', probability=True)
            model.fit(emb_array, data.y)
            
            print "the compute is -->:", compute_rate(model,config,embeddings,images_placeholder,phase_train_placeholder,embedding_size,sess)
            
            class_names = [cls.name.replace('_', ' ') for cls in data.dataset]
            with open(classifier_filename_exp, 'wb') as outfile:
                pickle.dump((model, class_names), outfile)
            print('Saved classifier model to file "%s"' % classifier_filename_exp)
    
if __name__ == '__main__':
    try:
        config = process_config()
    except:
        print("missing or invalid arguments")
        exit(0)
    main(config)