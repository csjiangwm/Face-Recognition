# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 17:14:09 2018

@author: jwm
"""

from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import os
from utils.utils import load_model,prewhiten
from models.MTCNN import MTCNN


def main(config):
    
    if isinstance(config.compared_imgs,str):
        imgs = config.compared_imgs.split(' ')
    else:
        imgs = config.compared_imgs
        
    images = load_and_align_data(config,imgs)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            load_model(config.lfw.valid_model_path)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            # Run forward pass to calculate embeddings
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb = sess.run(embeddings, feed_dict=feed_dict)

            nrof_images = len(imgs)

            print('Images:')
            for i in range(nrof_images):
                print('%1d: %s' % (i, imgs[i]))
            print('')

            # Print distance matrix
            print('Distance matrix')
            print('    ', end='')
            for i in range(nrof_images):
                print('    %1d     ' % i, end='')
            print('')
            for i in range(nrof_images):
                print('%1d  ' % i, end='')
                for j in range(nrof_images):
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))
                    print('  %1.4f  ' % dist, end='')
                print('')


def load_and_align_data(config,imgs):
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            detector = MTCNN(config,sess)
#    print (config.compared_imgs)
    nrof_samples = len(imgs)
    img_list = [None] * nrof_samples
    for i in xrange(nrof_samples):
        img = misc.imread(os.path.expanduser(imgs[i]))
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = detector.detect_face(img)
        det = np.squeeze(bounding_boxes[0, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - config.mtcnn.margin / 2, 0)
        bb[1] = np.maximum(det[1] - config.mtcnn.margin / 2, 0)
        bb[2] = np.minimum(det[2] + config.mtcnn.margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + config.mtcnn.margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(cropped, (config.mtcnn.image_size, config.mtcnn.image_size), interp='bilinear')
        prewhitened = prewhiten(aligned)
        img_list[i] = prewhitened
    images = np.stack(img_list)
    return images