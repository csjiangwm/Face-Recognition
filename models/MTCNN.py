# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:07:07 2018

@author: jwm
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from base.network import Network
import tensorflow as tf
import numpy as np
import os
from utils.utils import imresample,generateBoundingBox,nms,rerec,pad,bbreg,normalize

class PNet(Network):
    '''PNet
       Input : None x 12 x 12 x 3    images
       Output: None x  1 x  1 x 2    face classification
               None x  1 x  1 x 4    bounding box regression
               
      Input :  None x  None x  None x 3    images
       Output: None x  num  x  num  x 2    face classification          num grids with each grid contains 2 elements represent face or not face probality
               None x  num  x  num  x 4    bounding box regression      num grids with each grid contains 4 elements represent bounding box regression 
    '''
    def setup(self):
        (self.feed('data') #pylint: disable=no-value-for-parameter, no-member
             .conv(3, 3, 10, 1, 1, padding='VALID', relu=False, name='conv1')   # output: None x 10 x 10 x 10
             .prelu(name='PReLU1')
             .max_pool(2, 2, 2, 2, name='pool1')                                # output: None x 5  x 5  x 10
             .conv(3, 3, 16, 1, 1, padding='VALID', relu=False, name='conv2')   # output: None x 3  x 3  x 16
             .prelu(name='PReLU2')
             .conv(3, 3, 32, 1, 1, padding='VALID', relu=False, name='conv3')   # output: None x 1  x 1  x 32
             .prelu(name='PReLU3')
             .conv(1, 1, 2, 1, 1, relu=False, name='conv4-1')                   # output: None x 1  x 1  x 2   face classification
             .softmax(3,name='prob1'))                                          # output: None x 1  x 1  x 2   face classification

        (self.feed('PReLU3') #pylint: disable=no-value-for-parameter
             .conv(1, 1, 4, 1, 1, relu=False, name='conv4-2'))                  # output: None x 1  x 1  x 4   bounding box regression
             
class RNet(Network):
    '''RNet
       Input : None x 24 x 24 x 3    images
       Output: None x  1 x  1 x 2    face classification
               None x  1 x  1 x 4    bounding box regression
               None x  1 x  1 x 10   facial landmark localization
    '''
    def setup(self):
        (self.feed('data') #pylint: disable=no-value-for-parameter, no-member
             .conv(3, 3, 28, 1, 1, padding='VALID', relu=False, name='conv1')   # output: None x 22 x 22 x 28
             .prelu(name='prelu1')
             .max_pool(3, 3, 2, 2, name='pool1')                                # output: None x 11 x 11 x 28
             .conv(3, 3, 48, 1, 1, padding='VALID', relu=False, name='conv2')   # output: None x 8  x 8  x 48
             .prelu(name='prelu2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')               # output: None x 4  x 4  x 48
             .conv(2, 2, 64, 1, 1, padding='VALID', relu=False, name='conv3')   # output: None x 3  x 3  x 64
             .prelu(name='prelu3')
             .fc(128, relu=False, name='conv4')                                 # output: None x 1  x 1  x 128
             .prelu(name='prelu4')
             .fc(2, relu=False, name='conv5-1')                                 # output: None x 2             face classification
             .softmax(1,name='prob1'))                                          # output: None x 2             face classification

        (self.feed('prelu4') #pylint: disable=no-value-for-parameter
             .fc(4, relu=False, name='conv5-2'))                                # output: None x 1  x 1  x 4   bounding box regression

class ONet(Network):
    '''PNet
       Input : None x 48 x 48 x 3    images
       Output: None x  1 x  1 x 2    face classification
               None x  1 x  1 x 4    bounding box regression
               None x  1 x  1 x 10   facial landmark localization
    '''
    def setup(self):
        (self.feed('data') #pylint: disable=no-value-for-parameter, no-member
             .conv(3, 3, 32, 1, 1, padding='VALID', relu=False, name='conv1')   # output: None x 46 x 46 x 32
             .prelu(name='prelu1')
             .max_pool(3, 3, 2, 2, name='pool1')                                # output: None x 23 x 23 x 32
             .conv(3, 3, 64, 1, 1, padding='VALID', relu=False, name='conv2')   # output: None x 21 x 21 x 64
             .prelu(name='prelu2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')               # output: None x 10 x 10 x 64
             .conv(3, 3, 64, 1, 1, padding='VALID', relu=False, name='conv3')   # output: None x 8  x 8  x 64
             .prelu(name='prelu3')
             .max_pool(2, 2, 2, 2, name='pool3')                                # output: None x 4  x 4  x 64
             .conv(2, 2, 128, 1, 1, padding='VALID', relu=False, name='conv4')  # output: None x 3  x 3  x 128
             .prelu(name='prelu4')
             .fc(256, relu=False, name='conv5')                                 # output: None x 1  x 1  x 256
             .prelu(name='prelu5')
             .fc(2, relu=False, name='conv6-1')                                 # output: None x 2             face classification
             .softmax(1, name='prob1'))                                         # output: None x 2             face classification

        (self.feed('prelu5') #pylint: disable=no-value-for-parameter
             .fc(4, relu=False, name='conv6-2'))                                # output: None x 1  x 1  x 4   bounding box regression

        (self.feed('prelu5') #pylint: disable=no-value-for-parameter
             .fc(10, relu=False, name='conv6-3'))                               # output: None x 1  x 1  x 10  facial landmark localization


class MTCNN():
    
    def __init__(self, config, sess):
        self.config = config
        self.sess = sess
        self.build_model()
        
    def build_model(self):    
        '''
           Create the MTCNN model
        '''
        with tf.variable_scope('pnet'):
            data = tf.placeholder(tf.float32, (None,None,None,3), 'input')
            pnet = PNet({'data':data})
            try:
                pnet.load(os.path.join(self.config.mtcnn_path, 'det1.npy'), self.sess)
            except IOError:
                raise IOError('Cannot find file: %s' % os.path.join(self.config.mtcnn_path, 'det1.npy'))
        with tf.variable_scope('rnet'):
            data = tf.placeholder(tf.float32, (None,24,24,3), 'input')
            rnet = RNet({'data':data})
            try:
                rnet.load(os.path.join(self.config.mtcnn_path, 'det2.npy'), self.sess)
            except IOError:
                raise IOError('Cannot find file: %s' % os.path.join(self.config.mtcnn_path, 'det2.npy'))
        with tf.variable_scope('onet'):
            data = tf.placeholder(tf.float32, (None,48,48,3), 'input')
            onet = ONet({'data':data})
            try:
                onet.load(os.path.join(self.config.mtcnn_path, 'det3.npy'), self.sess)
            except IOError:
                raise IOError('Cannot find file: %s' % os.path.join(self.config.mtcnn_path, 'det3.npy'))
            
        self.pnet_fun = lambda img : self.sess.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'), feed_dict={'pnet/input:0':img})
        self.rnet_fun = lambda img : self.sess.run(('rnet/conv5-2/conv5-2:0', 'rnet/prob1:0'), feed_dict={'rnet/input:0':img})
        self.onet_fun = lambda img : self.sess.run(('onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0'), feed_dict={'onet/input:0':img})
        
    def compute_scale(self,height,width):
        factor_count = 0
        minl = np.amin([height, width])
        m = 12.0/self.config.mtcnn.minsize
        minl = minl * m
        scales = []
        
        while minl >= 12:
            scales += [m * np.power(self.config.mtcnn.factor, factor_count)]
            minl = minl * self.config.mtcnn.factor
            factor_count += 1
        return scales
        
    def generate_img(self,img, h, w, total_boxes, numbox,size):
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)
        tempimg = np.zeros((size,size,3,numbox))
        for k in range(0,numbox):
            tmp = np.zeros((int(tmph[k]),int(tmpw[k]),3))
            tmp[dy[k]-1:edy[k],dx[k]-1:edx[k],:] = img[y[k]-1:ey[k],x[k]-1:ex[k],:]
            if tmp.shape[0]>0 and tmp.shape[1]>0 or tmp.shape[0]==0 and tmp.shape[1]==0:
                tempimg[:,:,:,k] = imresample(tmp, (size, size))
            else:
                return np.empty()
        return tempimg
    
    def process_pnet(self,img,total_boxes,h,w):
        scales = self.compute_scale(h,w)
        
        for j in range(len(scales)):
            scale = scales[j]
            hs = int(np.ceil(h * scale))
            ws = int(np.ceil(w * scale))
            
            im_data = normalize(imresample(img, (hs, ws)))
            
            img_x = np.expand_dims(im_data, 0)
            img_y = np.transpose(img_x, (0,2,1,3)) # rotate 90 degrees
            
            out = self.pnet_fun(img_y)
            out0 = np.transpose(out[0], (0,2,1,3)) # bounding box regression, rotate 90 degrees
            out1 = np.transpose(out[1], (0,2,1,3)) # prob
            
            boxes, _ = generateBoundingBox(out1[0,:,:,1].copy(), out0[0,:,:,:].copy(), scale, self.config.mtcnn.threshold1)
            
            # inter-scale nms
            pick = nms(boxes.copy(), 0.5, 'Union')
            if boxes.size>0 and pick.size>0:
                boxes = boxes[pick,:]
                total_boxes = np.append(total_boxes, boxes, axis=0)
        return total_boxes
        
    
    def nms_pnet(self,total_boxes,h,w):
        
        pick = nms(total_boxes.copy(), 0.7, 'Union')
        
        total_boxes = total_boxes[pick,:]
        
        regw = total_boxes[:,2]-total_boxes[:,0]
        regh = total_boxes[:,3]-total_boxes[:,1]
        
        qq1 = total_boxes[:,0]+total_boxes[:,5]*regw
        qq2 = total_boxes[:,1]+total_boxes[:,6]*regh
        qq3 = total_boxes[:,2]+total_boxes[:,7]*regw
        qq4 = total_boxes[:,3]+total_boxes[:,8]*regh
        
        total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:,4]]))
        total_boxes = rerec(total_boxes.copy())
        total_boxes[:,0:4] = np.fix(total_boxes[:,0:4]).astype(np.int32)
        
        return total_boxes
        
    def process_rnet(self, img, h, w, total_boxes, numbox):
        
        try:
            tempimg = self.generate_img(img, h, w, total_boxes, numbox, 24)
        except TypeError:
            raise TypeError("rnet generate_img failed")
            
        tempimg1 = np.transpose(normalize(tempimg), (3,1,0,2))
        
        out = self.rnet_fun(tempimg1)
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        
        score = out1[1,:]
        ipass = np.where(score>self.config.mtcnn.threshold2)
        total_boxes = np.hstack([total_boxes[ipass[0],0:4].copy(), np.expand_dims(score[ipass].copy(),1)])
        mv = out0[:,ipass[0]]
        
        if total_boxes.shape[0]>0:
            pick = nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick,:]
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv[:,pick]))
            total_boxes = rerec(total_boxes.copy())
            
        return total_boxes
        
    def process_onet(self, img, h, w, total_boxes, numbox, points):
        
        total_boxes = np.fix(total_boxes).astype(np.int32)
        
        try:
            tempimg = self.generate_img(img, h, w, total_boxes, numbox, 48)
        except TypeError:
            raise TypeError("rnet generate_img failed")
            
        tempimg1 = np.transpose(normalize(tempimg), (3,1,0,2))
        
        out = self.onet_fun(tempimg1)
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        out2 = np.transpose(out[2])
        
        score = out2[1,:]
        points = out1
        ipass = np.where(score>self.config.mtcnn.threshold3)
        points = points[:,ipass[0]]
        total_boxes = np.hstack([total_boxes[ipass[0],0:4].copy(), np.expand_dims(score[ipass].copy(),1)])
        mv = out0[:,ipass[0]]

        w = total_boxes[:,2] - total_boxes[:,0] + 1
        h = total_boxes[:,3] - total_boxes[:,1] + 1
        
        points[0:5,:] = np.tile(w,(5, 1)) * points[0:5,:] + np.tile(total_boxes[:,0],(5, 1)) - 1
        points[5:10,:] = np.tile(h,(5, 1)) * points[5:10,:] + np.tile(total_boxes[:,1],(5, 1)) - 1
        
        if total_boxes.shape[0] > 0:
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv))
            pick = nms(total_boxes.copy(), 0.7, 'Min')
            total_boxes = total_boxes[pick,:]
            points = points[:,pick]
        return total_boxes,points
            
        
    def detect_face(self,img):
        total_boxes = np.empty((0,9))
        points = []
        h = img.shape[0]
        w = img.shape[1]
        
        total_boxes = self.process_pnet(img,total_boxes, h, w)  
        
        numbox = total_boxes.shape[0]
        if numbox>0:
            total_boxes = self.nms_pnet(total_boxes, h, w)
        
        numbox = total_boxes.shape[0]
        if numbox>0:
            total_boxes = self.process_rnet(img, h, w, total_boxes, numbox)# second stage
                
        numbox = total_boxes.shape[0]
        if numbox>0:
            total_boxes, points = self.process_onet(img,h,w,total_boxes,numbox,points)# third stage
                
        return total_boxes, points
