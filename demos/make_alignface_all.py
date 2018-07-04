# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 15:29:14 2018

@author: jwm
"""
from __future__ import division

import tensorflow as tf
from utils.config import process_config,create_dirs
from utils.utils import to_rgb
from models.MTCNN import MTCNN
from data_loader.data_generator import DataGenerator
import numpy as np
from scipy import misc
import os

def main(config):
    
    print('Creating networks and loading parameters')
    
    Data = DataGenerator(config,False)
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            detector = MTCNN(config,sess)
            
    nrof_images_total = 0
    nrof_successfully_aligned = 0
    
    for cls in Data.dataset:
        output_class_dir = os.path.join(config.output_dir, cls.name)
        create_dirs([output_class_dir])
            
        for image_path in cls.image_paths:
            nrof_images_total += 1
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            output_filename = os.path.join(output_class_dir, filename + '.png')
            print(image_path)
            if not os.path.exists(output_filename):
                try:
                    img = misc.imread(image_path)
#                        print('read data dimension: ', img.ndim)
                except (IOError, ValueError, IndexError) as e:
                    errorMessage = '{}: {}'.format(image_path, e)
                    print(errorMessage)
                else:
                    if img.ndim < 2:
                        print('Unable to align "%s"' % image_path)
                        continue
                    
                    if img.ndim == 2:
                        img = to_rgb(img)
                        print('to_rgb data dimension: ', img.ndim)
                    img = img[:, :, 0:3]
#                    print('after data dimension: ', img.ndim)

                    bounding_boxes, _ = detector.detect_face(img)
                    
                    nrof_faces = bounding_boxes.shape[0]
#                    print('detected_face: %d' % nrof_faces)
                    
                    if nrof_faces > 0:
                        det = bounding_boxes[:, 0:4]
                        bb_temp = np.zeros((nrof_faces,4), dtype=np.int32)
                        for i in range(nrof_faces):
                            bb_temp[i][0] = det[i][0]
                            bb_temp[i][1] = det[i][1]
                            bb_temp[i][2] = det[i][2]
                            bb_temp[i][3] = det[i][3]

                            cropped_temp = img[bb_temp[i][1]:bb_temp[i][3], bb_temp[i][0]:bb_temp[i][2], :]
                        
                            try:
                                scaled_temp = misc.imresize(cropped_temp, (config.mtcnn.image_size, config.mtcnn.image_size), interp='bilinear')
                            except (IOError, ValueError, IndexError) as e:
                                continue
                            
                            nrof_successfully_aligned += 1
                            
                            if nrof_faces == 1:
                                misc.imsave(output_filename, scaled_temp)
                            else:
                                output_bodyname = output_filename.split('.')[0]
                                output_indexname = '_%d' % i
                                misc.imsave(output_bodyname + output_indexname + '.png', scaled_temp)
                    else:
                        print('Unable to align "%s"' % image_path)
    
    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
                
                
if __name__ == '__main__':
    try:
        config = process_config()
    except:
        print("missing or invalid arguments")
        exit(0)
    main(config)
                    