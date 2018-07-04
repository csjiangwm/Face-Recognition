# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 11:07:13 2018

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
import cv2

def main(config):
    print('Creating networks and loading parameters')
    
    Data = DataGenerator(config)
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            detector = MTCNN(config,sess)
            
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(config.output_dir, 'bounding_boxes_%05d.txt' % random_key)
    
    with open(bounding_boxes_filename, "w") as text_file:
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
                        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
#                        print('read data dimension: ', img.ndim)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim < 2:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
                            continue
                        
                        if img.ndim == 2:
                            img = to_rgb(img)
                            print('to_rgb data dimension: ', img.ndim)
                        img = img[:, :, 0:3]
#                        print('after data dimension: ', img.ndim)
    
                        bounding_boxes, _ = detector.detect_face(img)
                        
                        nrof_faces = bounding_boxes.shape[0]
#                        print('detected_face: %d' % nrof_faces)
                        
                        if nrof_faces > 0:
                            det = bounding_boxes[:, 0:4]
                            img_size = np.asarray(img.shape)[0:2]
                            if nrof_faces > 1:
                                bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                                img_center = img_size / 2
                                offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                                index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                                det = det[index, :]
                            det = np.squeeze(det)
                            bb_temp = np.zeros(4, dtype=np.int32)
    
                            bb_temp[0] = det[0]
                            bb_temp[1] = det[1]
                            bb_temp[2] = det[2] 
                            bb_temp[3] = det[3]
    
                            cropped_temp = img[bb_temp[1]:bb_temp[3], bb_temp[0]:bb_temp[2], :]
                            
                            try:
                                scaled_temp = misc.imresize(cropped_temp, (config.mtcnn.image_size, config.mtcnn.image_size), interp='bilinear')
                            except (IOError, ValueError, IndexError) as e:
                                continue
                            nrof_successfully_aligned += 1
                            
                            misc.imsave(output_filename, scaled_temp)
                            text_file.write('%s %d %d %d %d\n' % (output_filename, bb_temp[0], bb_temp[1], bb_temp[2], bb_temp[3]))
                        else:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
    
    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
                
                
if __name__ == '__main__':
    try:
        config = process_config()
    except:
        print("missing or invalid arguments")
        exit(0)
    main(config)