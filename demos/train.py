# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:07:07 2018

@author: jwm
"""

import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.ResNet import ResNet
from trainers.resnet_trainer import ResnetTrainer
from utils.config import process_config
from utils.logger import Logger


def main(config): 
    
    # create tensorflow session
    session_config = tf.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = config.gpu_memory_fraction
    sess = tf.Session(config=session_config)
    # create an instance of the model
    model = ResNet(config)
    # create your data generator
    data = DataGenerator(config)
    
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = ResnetTrainer(sess, model, data, config, logger)
    # train model
    print "Start training..."
    trainer.train()
    sess.close()


if __name__ == '__main__':
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        config = process_config()
    except:
        print("missing or invalid arguments")
        exit(0)

    main(config)
