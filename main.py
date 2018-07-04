# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 14:14:55 2018

@author: jwm
"""
from __future__ import absolute_import
from demos.detect_face import main as detect_face
from demos.detect_video import main as detect_video
from demos.make_alignface_all import main as make_alignface
from demos.make_alignface_single import main as make_alignface_single
from demos.valid_lfw import main as valid_lfw
from demos.compare import main as compare
from demos.train import main as train
from demos.recognize import main as recognize
from demos.classifier import main as classifier
from utils.config import process_config

try:
    config = process_config()
except:
    print("missing or invalid arguments")
    exit(0)
    
if config.task == 'detect':
    detect_face(config)
elif config.task == 'detecting':
    detect_video(config)
elif config.task == 'all':
    make_alignface(config)
elif config.task == 'single':
    make_alignface_single(config)
elif config.task == 'valid':
    valid_lfw(config)
elif config.task == 'compare':
    compare(config)
elif config.task == 'train':
    train(config)
elif config.task == 'recognize':
    recognize(config)
elif config.task == 'classifier':
    classifier(config)
    
