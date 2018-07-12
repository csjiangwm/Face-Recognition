# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 13:46:00 2018

@author: jwm
"""

from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import os
from utils.utils import load_model,prewhiten
from models.MTCNN import MTCNN
import re
import requests
import urllib

url_init_first = r'http://image.baidu.com/search/flip?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1497491098685_R&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&ctd=1497491098685%5E00_1519X735&word='

def get_onepage_urls(onepageurl):
    """获取单个翻页的所有图片的urls+当前翻页的下一翻页的url"""
    if not onepageurl:
        print('已到最后一页, 结束')
        return [], ''
    try:
        html = requests.get(onepageurl).text
    except Exception as e:
        print(e)
        pic_urls = []
        fanye_url = ''
        return pic_urls, fanye_url
    pic_urls = re.findall('"objURL":"(.*?)",', html, re.S)
    fanye_urls = re.findall(re.compile(r'<a href="(.*)" class="n">下一页</a>'), html, flags=0)
    fanye_url = 'http://image.baidu.com' + fanye_urls[0] if fanye_urls else ''
    return pic_urls, fanye_url
    
def download(pic_url,index,path):
    """给出图片链接列表, 下载所有图片"""
    try:
        pic = requests.get(pic_url, timeout=15)
        img_type = pic_url.split('.')[-1]
        string = str(index + 1) + '.' + img_type
        image_path = os.path.join(path,string)
        with open(image_path, 'wb') as f:
            f.write(pic.content)
#            print('成功下载第%s张图片: %s' % (str(index + 1), str(pic_url)))
            return image_path
    except Exception as e:
        print('下载第%s张图片时失败: %s' % (str(index + 1), str(pic_url)))
        print(e)
        return False
        
def get_keyword(name,path):
    keyword = name.split(' ')[0]
    name = re.search(r'[a-zA-Z0-9]+',name).group()
    saved_dir = os.path.join(path,name)
    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)
    return keyword,name,saved_dir

def detect(img,detector,config):
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detector.detect_face(img)
    nrof_faces = bounding_boxes.shape[0]
    if nrof_faces > 0:
        det = np.squeeze(bounding_boxes[0, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - config.mtcnn.margin / 2, 0)
        bb[1] = np.maximum(det[1] - config.mtcnn.margin / 2, 0)
        bb[2] = np.minimum(det[2] + config.mtcnn.margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + config.mtcnn.margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(cropped, (config.mtcnn.image_size, config.mtcnn.image_size), interp='bilinear')
        img = prewhiten(aligned)
    return nrof_faces,img
        

def main(config):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            detector = MTCNN(config,sess)
            # Load the model
            load_model(config.lfw.valid_model_path)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            # Run forward pass to calculate embeddings
            
            with open(config.parse_path) as f:
                names = f.readlines()
                
            for name in names:
                keyword,name,saved_dir = get_keyword(name,config.output_dir)
                fanye_url = url_init_first + urllib.quote(keyword, safe='/')
                feats = []
                page = 0
                while len(os.listdir(saved_dir)) < 20 and fanye_url != '':
                    onepage_urls, fanye_url = get_onepage_urls(fanye_url)
                    for index,url in enumerate(onepage_urls):
                        img_path = download(url, page * len(onepage_urls) + index, saved_dir)
                        try:
                            img = misc.imread(img_path)
                            nrof_faces,face = detect(img,detector,config)
                            face = np.expand_dims(face,0)
                            if nrof_faces:
                                feed_dict = {images_placeholder: face, phase_train_placeholder: False}
                                feats.append(sess.run(embeddings, feed_dict=feed_dict))
                                if index != 0:
                                    dist = np.sqrt(np.sum(np.square(np.subtract(feats[0], feats[-1]))))
                                    print ('the %d-th %s image dist: %f' % (index,name,dist))
                                    if dist > config.max_dist:
                                        os.remove(img_path)
                        except Exception as e:
                            print(e)
                            if img_path:
                                os.remove(img_path)
                            continue
                    page += 1