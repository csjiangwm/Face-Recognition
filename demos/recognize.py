# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 16:28:56 2018

@author: jwm
"""

import tensorflow as tf
from utils.config import process_config
from utils.utils import to_rgb,load_model,flip,prewhiten
from models.MTCNN import MTCNN
import numpy as np
import cv2
from scipy import misc
import time
import os
import pickle


def main(config):
    
    print('Creating networks and loading parameters')
    
    graph_detect = tf.Graph()
    with graph_detect.as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            detector = MTCNN(config,sess)
            
            load_model(config.lfw.valid_model_path)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0") 
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            classifier_filename = config.classifier_path
            classifier_filename_exp = os.path.expanduser(classifier_filename)
            print('load classifier file-> %s' % classifier_filename_exp)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)
                
            HumanNames = os.listdir(config.input_dir)
            
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
                        cropped = []
                        scaled = []
                        scaled_reshape = []
                        bb = np.zeros((nrof_faces,4), dtype=np.int32)
                        for i in range(nrof_faces):
                            emb_array = np.zeros((1, embedding_size))
                            
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]
                            
                            # inner exception
                            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                print('face is inner of range!')
                                continue
                            cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                            cropped[i] = flip(cropped[i], False)
                            scaled.append(misc.imresize(cropped[i], (182, 182), interp='bilinear'))
                            scaled[i] = cv2.resize(scaled[i], (160,160),interpolation=cv2.INTER_CUBIC)
                            scaled[i] = prewhiten(scaled[i])
                            scaled_reshape.append(scaled[i].reshape(-1,160,160,3))
                            feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
#                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
#                            print "best_class_probabilities:", best_class_probabilities
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2) 
                            
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20     
                            print('result: ', best_class_indices[0])
                            for H_i in HumanNames:
                                if HumanNames[best_class_indices[0]] == H_i:
                                    result_names = HumanNames[best_class_indices[0]]
                                    cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (0, 0, 255), thickness=1, lineType=2)
                    else:
                        print('Unable to align')
                sec = curTime - prevTime
                prevTime = curTime
                fps = 1 / (sec)
                string = 'FPS: %2.3f' % fps
                text_fps_x = len(frame[0]) - 150
                text_fps_y = 20
                cv2.putText(frame, string, (text_fps_x, text_fps_y),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=1, lineType=2)
                c += 1
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