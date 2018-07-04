# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 13:24:54 2018

@author: jwm
"""

import tensorflow as tf
from data_loader.data_generator import DataGenerator
from utils.utils import random_rotate_image

class DataQueue(DataGenerator):
    '''
       Generate data
    '''
    def __init__(self, config, do_shuffle=True):
        super(DataQueue,self).__init__(config,do_shuffle)
        self.produce()
        self.load_images()
        self.coord = tf.train.Coordinator()
        
    def produce(self):
        index_queue = tf.train.range_input_producer(self.nrof_images,num_epochs=None,shuffle=self.do_shuffle,seed=None,capacity=32)
        self.index_dequeue_op = index_queue.dequeue_many(self.config.batch_size * self.config.epoch_size, 'index_dequeue') # dequeue
        
        self.image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')
        self.labels_placeholder = tf.placeholder(tf.int64, shape=(None, 1), name='labels')

        self.input_queue = tf.FIFOQueue(capacity=100000,dtypes=[tf.string, tf.int64],shapes=[(1,), (1,)],shared_name=None, name=None) # create a FIFO queue that can contain 'capacity=100000' elements
        self.enqueue_op = self.input_queue.enqueue_many([self.image_paths_placeholder, self.labels_placeholder], name='enqueue_op')
        
    def load_images(self):
        nrof_preprocess_threads = 4
        images_and_labels = []
        
        for _ in range(nrof_preprocess_threads):
            filenames, label = self.input_queue.dequeue() # dequeue
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_png(file_contents)
                if self.config.resnet.random_rotate:
                    image = tf.py_func(random_rotate_image, [image], tf.uint8)
                if self.config.resnet.random_crop:
                    image = tf.random_crop(image, [self.config.resnet.image_size, self.config.resnet.image_size, 3])
                else:
                    image = tf.image.resize_image_with_crop_or_pad(image, self.config.resnet.image_size, self.config.resnet.image_size)
                if self.config.resnet.random_flip:
                    image = tf.image.random_flip_left_right(image)

                # pylint: disable=no-member
                image.set_shape((self.config.resnet.image_size, self.config.resnet.image_size, 3))
                images.append(tf.image.per_image_standardization(image))
            images_and_labels.append([images, label])

        self.image_batch, self.label_batch = tf.train.batch_join(
            images_and_labels, batch_size=self.config.batch_size,
            shapes=[(self.config.resnet.image_size, self.config.resnet.image_size, 3), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * self.config.batch_size,
            allow_smaller_final_batch=True)
          