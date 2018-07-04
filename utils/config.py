# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:07:07 2018

@author: jwm
"""

import argparse
import json
from bunch import Bunch
import os
from datetime import datetime


def get_args():
    argparser = argparse.ArgumentParser(description='this is a parameter description')
    argparser.add_argument(
        'task',
        metavar='task',
        help='which task you want to deal',
        type=str)
    argparser.add_argument(
        '--config',
        metavar='Config',
        default='configs/config.json',
        help='The configuration file')
    argparser.add_argument(
        '--mtcnn_path',
        metavar='mtcnn_path',
        default='./experiments/',
        help='path of det1.npz, det2.npz and det3.npz')
    argparser.add_argument(
        '--input_dir',
        metavar='input_dir',
        default='/media/jwm/DATA/For_Linux/general_data/lfw',
        help='make_alignface input')
    argparser.add_argument(
        '--output_dir',
        metavar='output_dir',
        default='/media/jwm/DATA/For_Linux/general_data/lfw_align',
        help='make_alignface output')
#    argparser.add_argument(
#        '--img_dir',
#        metavar='img_dir',
#        default='/media/jwm/DATA/For_Linux/general_data/lfw_align',
#        help='data_generator input')
    argparser.add_argument(
        '--image',
        metavar='image',
        default='images/timg.jpg',
        help='a test')
    argparser.add_argument(
        '--compared',
        metavar='compared_imgs',
        default='./images/xuanye.jpg ./images/fengjinwei.jpg',
        type=str, 
        nargs='+', 
        help='Images to compare')
    args = argparser.parse_args()
    return args
    
def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)
    return config, config_dict
    
def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)
    
def process_config():
    args = get_args()
    config, _ = get_config_from_json(args.config)
    
    config.mtcnn, _ = get_config_from_json(config.mtcnn_config_path)
    config.resnet, _ = get_config_from_json(config.resnet_config_path)
    config.lfw, _ = get_config_from_json(config.lfw_config_path)
    config.summary_dir = os.path.join("./experiments", "summary/")
    config.checkpoint_dir = os.path.join("./experiments", "ckpt/")
    
    config.mtcnn_path = args.mtcnn_path
    config.output_dir = args.output_dir
    config.input_dir = args.input_dir
#    config.img_dir = args.img_dir
    
    config.task = args.task
    config.image = args.image
    config.compared_imgs = args.compared
    
    create_dirs([config.summary_dir, config.checkpoint_dir, config.output_dir])
    config.model_name = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    return config
    
