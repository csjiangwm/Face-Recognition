# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 13:15:17 2018

@author: jwm
"""
from __future__ import division

import tensorflow as tf
import os

class BaseModel(object):
    def __init__(self, config):
        '''
           Create the base model
           All the variables are created
        '''
        self.config = config
        self.init_global_step() # init the global step
        self.init_cur_epoch() # init the epoch counter

    
    def save(self, sess, write_meta_graph=True, step=None):
        '''
           Save function that saves the checkpoint in the path defined in the config file
        '''
        print("Saving model...")
        if write_meta_graph:
            self.saver.save(sess, os.path.join(self.config.checkpoint_dir,'train.model'), self.global_step_tensor)
        else:
            checkpoint_path = os.path.join(self.config.checkpoint_dir, 'model-%s.ckpt' % self.config.model_name)
            self.saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=write_meta_graph)
            metagraph_filename = os.path.join(self.config.checkpoint_dir, 'model-%s.meta' % self.config.model_name)
            if not os.path.exists(metagraph_filename):
                print('Saving metagraph')
                self.saver.export_meta_graph(metagraph_filename)
        print("Model saved")

    
    def load(self, sess):
        '''
           Load latest checkpoint from the experiment path defined in the config file
        '''
        # latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(self.config.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Loading model checkpoint {} ...\n".format(ckpt.model_checkpoint_path))
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model loaded")
            return True
        return False
    
    def init_cur_epoch(self):
        '''
           Just initialize a tensorflow variable to use it as epoch counter
        '''
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            # self.cur_epoch_tensor add 1, for the convenience of restarting training
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    
    def init_global_step(self):
        '''
           Just initialize a tensorflow variable to use it as global step counter
        '''
        # Add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def init_saver(self, meta_path=None):
        '''
           Initialize the tensorflow saver that will be used in saving the checkpoints.
        '''
        self.saver = tf.train.Saver()

    def build_model(self):
        raise NotImplementedError