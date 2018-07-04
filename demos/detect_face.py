# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 09:55:39 2018

@author: jwm
"""

import tensorflow as tf
from utils.config import process_config
from utils.utils import to_rgb
from models.MTCNN import MTCNN
import numpy as np
import cv2

def main(config):
    
    print('Creating networks and loading parameters')
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            detector = MTCNN(config,sess)
            frame = cv2.imread(config.image)
#            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    # If necessary
            if frame.ndim == 2:
                frame = to_rgb(frame)
            frame = frame[:, :, 0:3]
            bounding_boxes, _ = detector.detect_face(frame)
            nrof_faces = bounding_boxes.shape[0]
            
            print('Detected_FaceNum: %d' % nrof_faces)
            
            if nrof_faces > 0:
                det = bounding_boxes[:, 0:4]
                
                bb = np.zeros((nrof_faces,4), dtype=np.int32)
                for i in range(nrof_faces):
                    
                    bb[i][0] = det[i][0]
                    bb[i][1] = det[i][1]
                    bb[i][2] = det[i][2]
                    bb[i][3] = det[i][3]
                    
                    # inner exception
                    if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                        print('face is inner of range!')
                        continue
                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2) 
            else:
                print('Unable to align')
            cv2.imshow('Image', frame)
            if cv2.waitKey(0):
                cv2.destroyAllWindows()
                
                
if __name__ == '__main__':
    try:
        config = process_config()
    except:
        print("missing or invalid arguments")
        exit(0)
    main(config)