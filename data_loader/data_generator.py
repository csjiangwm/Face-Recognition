# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:07:07 2018

@author: jwm
"""

import numpy as np
import os
import random
from utils.utils import load_data,getPaddingSize
import cv2

class ImageClass():
    '''Stores the paths to images for a given class'''
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)

class DataGenerator(object):
    '''
       Generate data
    '''
    def __init__(self, config, do_shuffle=True):
        self.config = config
        self.do_shuffle = do_shuffle
        self.dataset = []
        self.get_dataset() # compute self.dataset
        self.get_image_paths_and_labels()
        self.nrof_images = len(self.input)
        assert self.nrof_images > 0, 'The dataset should not be empty'
        print "Total number of images is ", self.nrof_images
        print "Total number of classes is ", self.config.nrof_classes

    def next_batch_random(self):
        '''
           Stochastic gradient descent
        '''
        idx = np.random.choice(len(self.input), self.config.batch_size)
        image_paths = [self.input[i] for i in idx]
        image_index = np.array([self.y[i] for i in idx])
        yield load_data(image_paths,self.config.resnet.image_size,self.config.resnet.random_crop,self.config.resnet.random_flip), \
              image_index
        
    def next_batch(self,start_idx):
        '''Memory saving procedure, for batch gradient descent'''
        try:
            idx = np.array(range(start_idx*self.config.batch_size,(start_idx+1)*self.config.batch_size))
        except IndexError:
            idx = np.array(range(start_idx*self.config.batch_size,len(self.input)))
        image_paths = [self.input[i] for i in idx]
        image_index = np.array([self.y[i] for i in idx])
        yield self.load_images(image_paths), self.one_hot(image_index)
        
    def get_image_paths(self,facedir):
        image_paths = []
        if os.path.isdir(facedir):
            images = os.listdir(facedir)
            image_paths = [os.path.join(facedir,img) for img in images]
        return image_paths
        
    def get_dataset(self):
        ''' Package path and previous level dir name of current image '''
        for path in self.config.input_dir.split(':'):
            path_exp = os.path.expanduser(path)
            classes = os.listdir(path_exp)
            classes.sort()
            assert self.config.nrof_classes == len(classes), 'number of classes should be respect to fact'
            for i in range(self.config.nrof_classes):
                class_name = classes[i]
                facedir = os.path.join(path_exp, class_name)
                image_paths = self.get_image_paths(facedir)
                self.dataset.append(ImageClass(class_name, image_paths))
    
    def get_image_paths_and_labels(self):
        '''Achieve image paths and labels'''
        image_paths_flat = []
        labels_flat = []
        for i in range(len(self.dataset)):
            image_paths_flat += self.dataset[i].image_paths
            labels_flat += [i] * len(self.dataset[i].image_paths)
        if self.do_shuffle:
            img_index = range(len(labels_flat))
            random.shuffle(img_index)
            image_paths_flat = [image_paths_flat[i] for i in img_index]
            labels_flat = [labels_flat[i] for i in img_index]
        self.input = image_paths_flat
        self.y = labels_flat
        
    def one_hot(self,image_index):
        '''Transform label vector to matrix'''
        labels_mat = np.zeros((self.config.batch_size,self.nrof_classes))
        for i in set(image_index):
            indexOfCurClass = np.where(image_index == i)[0]
            labels_mat[indexOfCurClass,i] = 1
        return labels_mat
        
    def load_images(self,image_paths):
        imgs = []
        for image_path in image_paths:
            img = cv2.imread(image_path)
            top,bottom,left,right = getPaddingSize(img)
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
            img = cv2.resize(img, (self.config.img_height, self.config.img_width))
            imgs.append(img)
        imgs = np.array(imgs)
        imgs = imgs.astype('float32')/255.0
        return imgs
    
