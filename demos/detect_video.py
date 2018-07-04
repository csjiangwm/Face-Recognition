# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 14:12:38 2018

@author: jwm
"""

import tensorflow as tf
from utils.config import process_config
from utils.utils import to_rgb
from models.MTCNN import MTCNN
import numpy as np
import cv2
import time

def main(config):
    
    print('Creating networks and loading parameters')
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            detector = MTCNN(config,sess)
            video_capture = cv2.VideoCapture(0)
            prevTime = 0
            c = 0
            while True:
                ret, frame = video_capture.read()
                frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    # resize frame (optional), 
                curTime = time.time()    # calc fps
                timeF = config.mtcnn.frame_interval
                if (c % timeF == 0):
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
                sec = curTime - prevTime
                prevTime = curTime
                fps = 1 / (sec)
                string = 'FPS: %2.3f' % fps
                text_fps_x = len(frame[0]) - 150
                text_fps_y = 20
                cv2.putText(frame, string, (text_fps_x, text_fps_y),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=1, lineType=2)
#                c += 1
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            video_capture.release()
            cv2.destroyAllWindows()
                
                
if __name__ == '__main__':
    try:
        config = process_config()
    except:
        print("missing or invalid arguments")
        exit(0)
    main(config)
                    