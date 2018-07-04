# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:07:07 2018

@author: jwm
"""

from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import time
from utils import lfw
import os
import math
from utils.utils import load_data


class ResnetTrainer(BaseTrain):
    
    def __init__(self, sess, model, data, config, logger):
        super(ResnetTrainer, self).__init__(sess, model, data, config,logger)
        
        self.model.compute_loss()
        self.model.optimize(tf.global_variables())
        
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.model.init_saver()
        
        if not self.model.load(sess):
            self.sess.run(self.init)
            
        if self.config.lfw_dir:
            pairs = lfw.read_pairs(os.path.expanduser(self.config.lfw.lfw_pairs_path))
            self.lfw_paths, self.actual_issame = lfw.get_paths(os.path.expanduser(self.config.img_dir), pairs, 'png')
            
#        tf.train.start_queue_runners(coord=self.data.coord, sess=self.sess)
            
    def train_epoch(self,cur_epoch):
        loop = tqdm(range(self.config.epoch_size))
        losses = []
        start_time = time.time()
        for _ in loop:
            loss, reg_loss = self.train_step(cur_epoch) # without tf queue
            losses.append(loss)
        loss = np.mean(losses)
        
        duration = time.time() - start_time
        print ('Epoch: [%d]\tTime %.3f\tLoss %2.3f\tRegLoss %2.3f' % (cur_epoch, duration, loss, np.sum(reg_loss)))
        if self.config.lfw_dir:
            self.evaluate()
        
        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess,False,cur_it)
        
        return loss

    def train_step(self,cur_epoch):
        '''
           My first version, without tf queue
        '''
        if self.config.resnet.learning_rate > 0.0:
            lr = self.config.resnet.learning_rate
        else:
            lr = self.get_learning_rate_from_file(cur_epoch)
        batch_x, batch_y = next(self.data.next_batch_random())
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.learning_rate:lr,self.model.phase_train:True}
        _, loss, reg_loss = self.sess.run([self.model.train_op, self.model.loss, self.model.regularization_losses],feed_dict=feed_dict)
        return loss, reg_loss
        
    def train_step_queue(self,cur_epoch):
        '''
           My first version, without tf queue
        '''
        if self.config.resnet.learning_rate > 0.0:
            lr = self.config.resnet.learning_rate
        else:
            lr = self.get_learning_rate_from_file(cur_epoch)
            
        index_epoch = self.sess.run(self.data.index_dequeue_op)
        label_epoch = np.array(self.data.y)[index_epoch]
        image_epoch = np.array(self.data.input)[index_epoch]
        labels_array = np.expand_dims(np.array(label_epoch), 1)
        image_paths_array = np.expand_dims(np.array(image_epoch), 1)
        self.sess.run(self.data.enqueue_op, {self.data.image_paths_placeholder: image_paths_array, self.data.labels_placeholder: labels_array})
        batch_x = self.sess.run(self.data.image_batch)
        batch_y = self.sess.run(self.data.label_batch)
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.learning_rate:lr,self.model.phase_train:True}
        _, loss, reg_loss = self.sess.run([self.model.train_op, self.model.loss, self.model.regularization_losses],feed_dict=feed_dict)
        return loss, reg_loss
        
    def get_learning_rate_from_file(self,cur_epoch):
        with open(self.config.resnet.learning_rate_schedule_file, 'r') as f:
            for line in f.readlines():
                line = line.split('#', 1)[0]
                if line:
                    par = line.strip().split(':')
                    e = int(par[0])
                    lr = float(par[1])
                    if e <= cur_epoch:
                        learning_rate = lr
                    else:
                        return learning_rate
                        
    def evaluate(self):
        print "start evaluating ..."
        embedding_size = self.model.embeddings.get_shape()[1]
        nrof_images = len(self.actual_issame) * 2
        assert nrof_images % self.config.batch_size == 0, 'The number of LFW images must be an integer multiple of the LFW batch size'
        nrof_batches = int(math.ceil(1.0*nrof_images / self.config.batch_size))
        emb_array = np.zeros((nrof_images, embedding_size))
        
        for i in range(nrof_batches):
            start_index = i*self.config.batch_size
            end_index = min((i+1)*self.config.batch_size, nrof_images)
            paths_batch = self.lfw_paths[start_index:end_index]
            images = load_data(paths_batch, self.config.lfw.image_size, False, False)
            feed_dict = { self.model.x:images, self.model.phase_train:False }
            emb_array[start_index:end_index,:] = self.sess.run(self.model.embeddings, feed_dict=feed_dict)
    
        tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(emb_array, self.actual_issame, nrof_folds=self.config.lfw.lfw_nrof_folds)

        print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
        print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))        