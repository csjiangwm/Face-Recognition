# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 10:34:54 2018

@author: jwm
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from base.base_model import BaseModel

class ResNet(BaseModel):
    '''
       resnet
    '''
    def __init__(self,config):
        super(ResNet,self).__init__(config)
#        self.config = config
        self.build()
#        self.init_saver()
    
    def build(self,reuse=None):
        
        self.x = tf.placeholder(tf.float32,shape=[None,self.config.resnet.image_size,self.config.resnet.image_size,3],name='input')
        self.y = tf.placeholder(tf.int64,shape=(None,),name='label_batch')
#        self.x = tf.identity(self.x,'input')
#        self.y = tf.identity(self.y,'label_batch')
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        
        batch_norm_params = {'decay'  : 0.995,            # Decay for the moving averages.
                             'epsilon': 0.001,            # epsilon to prevent 0s in variance.
                             'updates_collections': None, # force in-place updates of mean and variance estimates
                             'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],# Moving averages ends up 
                            }                                                              # in the trainable variables collection
        if self.config.resnet.version == 1 or self.config.resnet.version == 2:
            weights_initializer = tf.truncated_normal_initializer(stddev=0.1)
        else:
            weights_initializer = slim.xavier_initializer_conv2d(uniform=True)
            
        with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_initializer=weights_initializer,
                                                                 weights_regularizer=slim.l2_regularizer(self.config.resnet.weight_decay),
                                                                 normalizer_fn=slim.batch_norm,
                                                                 normalizer_params=batch_norm_params):
                                                                     
            if self.config.resnet.version == 1:
                self.inception_resnet_v1(reuse)
            elif self.config.resnet.version == 2:
                self.inception_resnet_v2(reuse)
            else:
                self.squeezenet(reuse)
                
    def inception_resnet_v1(self,reuse=None,scope='InceptionResnetV1'):
        print "building the inception resnet v1 model"
        end_points = {}
        with tf.variable_scope(scope,'InceptionResnetV1', values=[self.x], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=self.phase_train):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],stride=1, padding='SAME'):
                    
                    conv = slim.conv2d(self.x, 32, 3, stride=2, padding='VALID',scope='Conv2d_1a_3x3')     # 149 x 149 x 32
                    end_points['Conv2d_1a_3x3'] = conv
                    conv = slim.conv2d(conv, 32, 3, padding='VALID',scope='Conv2d_2a_3x3')                 # 147 x 147 x 32
                    end_points['Conv2d_2a_3x3'] = conv
                    conv = slim.conv2d(conv, 64, 3, scope='Conv2d_2b_3x3')                                 # 147 x 147 x 64
                    end_points['Conv2d_2b_3x3'] = conv
                    conv = slim.max_pool2d(conv, 3, stride=2, padding='VALID',scope='MaxPool_3a_3x3')      # 73 x 73 x 64
                    end_points['MaxPool_3a_3x3'] = conv
                    conv = slim.conv2d(conv, 80, 1, padding='VALID',scope='Conv2d_3b_1x1')                 # 73 x 73 x 80
                    end_points['Conv2d_3b_1x1'] = conv
                    conv = slim.conv2d(conv, 192, 3, padding='VALID',scope='Conv2d_4a_3x3')                # 71 x 71 x 192
                    end_points['Conv2d_4a_3x3'] = conv
                    conv = slim.conv2d(conv, 256, 3, stride=2, padding='VALID',scope='Conv2d_4b_3x3')      # 35 x 35 x 256
                    end_points['Conv2d_4b_3x3'] = conv
                    
                    conv = slim.repeat(conv, 5, self.block35, scale=0.17)                                  # 35 x 35 x 256 --> 5 x Inception-resnet-A  
                    
                    with tf.variable_scope('Mixed_6a'):
                        conv = self.reduction_a(conv, 192, 192, 256, 384)                                  # 17 x 17 x 896 --> Reduction-A
                    end_points['Mixed_6a'] = conv
                    
                    conv = slim.repeat(conv, 10, self.block17, scale=0.10)                                 # 17 x 17 x 896 -->10 x Inception-Resnet-B
                    
                    with tf.variable_scope('Mixed_7a'):
                        conv = self.reduction_b(conv,384, 256, 256)                                        # 17 x 17 x 1792--> Reduction-A
                    end_points['Mixed_7a'] = conv
                    
                    conv = slim.repeat(conv, 5, self.block8, scale=0.20)                                   # 17 x 17 x 1792--> 5 x Inception-Resnet-C 
                    
                    conv = self.block8(conv, activation_fn=None)                                           # 17 x 17 x 1792
                    
                    with tf.variable_scope('Logits'):
                         end_points['PrePool'] = conv
                         conv = slim.avg_pool2d(conv, conv.get_shape()[1:3], padding='VALID',scope='AvgPool_1a_8x8') # 8 × 8 × 1792
                         conv = slim.flatten(conv)
                         conv = slim.dropout(conv, self.config.resnet.dropout_keep_prob, is_training=self.phase_train,scope='Dropout')
                         end_points['PreLogitsFlatten'] = conv
                    conv = slim.fully_connected(conv,self.config.resnet.bottleneck_layer_size, activation_fn=None, scope='Bottleneck', reuse=False) 
        self.endpoints = end_points
        self.bottleneck_layer = conv
    
    def inception_resnet_v2(self,reuse=None,scope='InceptionResnetV2'):
        print "building the inception resnet v2 model"
        end_points = {}
        with tf.variable_scope(scope, 'InceptionResnetV2', [self.x], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=self.phase_train):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                    net = slim.conv2d(self.x, 32, 3, stride=2, padding='VALID',scope='Conv2d_1a_3x3')     # 149 x 149 x 32
                    end_points['Conv2d_1a_3x3'] = net
                    net = slim.conv2d(net, 32, 3, padding='VALID',scope='Conv2d_2a_3x3')                  # 147 x 147 x 32
                    end_points['Conv2d_2a_3x3'] = net
                    net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')                                  # 147 x 147 x 64
                    end_points['Conv2d_2b_3x3'] = net
                    net = slim.max_pool2d(net, 3, stride=2, padding='VALID',scope='MaxPool_3a_3x3')       # 73 x 73 x 64
                    end_points['MaxPool_3a_3x3'] = net
                    net = slim.conv2d(net, 80, 1, padding='VALID',scope='Conv2d_3b_1x1')                  # 73 x 73 x 80
                    end_points['Conv2d_3b_1x1'] = net
                    net = slim.conv2d(net, 192, 3, padding='VALID', scope='Conv2d_4a_3x3')                # 71 x 71 x 192
                    end_points['Conv2d_4a_3x3'] = net
                    net = slim.max_pool2d(net, 3, stride=2, padding='VALID',scope='MaxPool_5a_3x3')       # 35 x 35 x 192
                    end_points['MaxPool_5a_3x3'] = net
            
                    with tf.variable_scope('Mixed_5b'):
                        net = self.reduction_v2(net)                                                      # 35 x 35 x 320
            
                    end_points['Mixed_5b'] = net
                    net = slim.repeat(net, 10, self.block35, scale=0.17)
            
                    with tf.variable_scope('Mixed_6a'):
                        net = self.reduction_a(net, 256, 256, 384, 384)                                   # 17 x 17 x 1024
            
                    end_points['Mixed_6a'] = net
                    net = slim.repeat(net, 20, self.block17, scale=0.10)
            
                    with tf.variable_scope('Mixed_7a'):
                        net = self.reduction_b(net, 384, 288, 320)
            
                    end_points['Mixed_7a'] = net
            
                    net = slim.repeat(net, 9, self.block8, scale=0.20)
                    net = self.block8(net, activation_fn=None)
            
                    net = slim.conv2d(net, 1536, 1, scope='Conv2d_7b_1x1')
                    end_points['Conv2d_7b_1x1'] = net
            
                    with tf.variable_scope('Logits'):
                        end_points['PrePool'] = net
                        #pylint: disable=no-member
                        net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',scope='AvgPool_1a_8x8')
                        net = slim.flatten(net)
                        net = slim.dropout(net, self.config.resnet.dropout_keep_prob, is_training=self.phase_train,scope='Dropout')
                        end_points['PreLogitsFlatten'] = net
                    net = slim.fully_connected(net, self.config.resnet.bottleneck_layer_size, activation_fn=None, scope='Bottleneck', reuse=False)
        self.endpoints = end_points
        self.bottleneck_layer = net
    
    def squeezenet(self,reuse=None,scope='squeezenet'):
        print "building the squeezenet model"
        with tf.variable_scope('squeezenet', [self.x], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=self.phase_train):
                net = slim.conv2d(self.x, 96, [7, 7], stride=2, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool1')
                net = self.fire_module(net, 16, 64, scope='fire2')
                net = self.fire_module(net, 16, 64, scope='fire3')
                net = self.fire_module(net, 32, 128, scope='fire4')
                net = slim.max_pool2d(net, [2, 2], stride=2, scope='maxpool4')
                net = self.fire_module(net, 32, 128, scope='fire5')
                net = self.fire_module(net, 48, 192, scope='fire6')
                net = self.fire_module(net, 48, 192, scope='fire7')
                net = self.fire_module(net, 64, 256, scope='fire8')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool8')
                net = self.fire_module(net, 64, 256, scope='fire9')
                net = slim.dropout(net, self.config.resnet.dropout_keep_prob)
                net = slim.conv2d(net, 1000, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv10')
                net = slim.avg_pool2d(net, net.get_shape()[1:3], scope='avgpool10')
                net = tf.squeeze(net, [1, 2], name='logits')
                net = slim.fully_connected(net, self.config.resnet.bottleneck_layer_size, activation_fn=None, scope='Bottleneck', reuse=False)
        self.bottleneck_layer = net
                         
    
    def compute_loss(self):
        '''
           center loss
        '''
        logits = slim.fully_connected(self.bottleneck_layer, self.config.nrof_classes, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(self.config.resnet.weight_decay),
                                      scope='Logits', reuse=False)
        self.embeddings = tf.nn.l2_normalize(self.bottleneck_layer, 1, 1e-10, name='embeddings')
        
        if self.config.resnet.center_loss_factor > 0.0:
            prelogits_center_loss, _ = self.center_loss()
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * self.config.resnet.center_loss_factor)
        
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        self.regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = tf.add_n([cross_entropy_mean] + self.regularization_losses, name='total_loss')
        
    def optimize(self, update_gradient_vars):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step_tensor,
                                                   self.config.resnet.learning_rate_decay_epochs * self.config.epoch_size,
                                                   self.config.resnet.learning_rate_decay_factor, staircase=True)
        # Generate moving averages of all losses.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [self.loss])
    
        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            if self.config.resnet.optimizer=='ADAGRAD':
                opt = tf.train.AdagradOptimizer(learning_rate)
            elif self.config.resnet.optimizer=='ADADELTA':
                opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
            elif self.config.resnet.optimizer=='ADAM':
                opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
            elif self.config.resnet.optimizer=='RMSPROP':
                opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
            elif self.config.resnet.optimizer=='MOM':
                opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
            else:
                raise ValueError('Invalid optimization algorithm')
        
            grads = opt.compute_gradients(self.loss, update_gradient_vars)
            
        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step_tensor)
      
        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(self.config.resnet.moving_average_decay, self.global_step_tensor)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
      
        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')
        self.train_op = train_op
        
    def block35(self, inputs, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
        """Inception Resnet A
           Builds the 35x35 resnet block.
        """
        with tf.variable_scope(scope, 'Block35', [inputs], reuse=reuse):
            with tf.variable_scope('Branch_0'):
                tower_conv = slim.conv2d(inputs, 32, 1, scope='Conv2d_1x1')                                     # 35 x 35 x 32
            with tf.variable_scope('Branch_1'):
                tower_conv1_0 = slim.conv2d(inputs, 32, 1, scope='Conv2d_0a_1x1')                               # 35 x 35 x 32
                tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')                        # 35 x 35 x 32
            with tf.variable_scope('Branch_2'):
                tower_conv2_0 = slim.conv2d(inputs, 32, 1, scope='Conv2d_0a_1x1')                               # 35 x 35 x 32
                tower_conv2_1 = slim.conv2d(tower_conv2_0, 32, 3, scope='Conv2d_0b_3x3')                        # 35 x 35 x 32
                tower_conv2_2 = slim.conv2d(tower_conv2_1, 32, 3, scope='Conv2d_0c_3x3')                        # 35 x 35 x 32
            mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], 3)                                    # 35 x 35 x 96
            up = slim.conv2d(mixed, inputs.get_shape()[3], 1, normalizer_fn=None,activation_fn=None, scope='Conv2d_1x1') # 35 x 35 x 256
            outputs = inputs + scale * up
            if activation_fn:
                outputs = activation_fn(outputs)                                                                # 35 x 35 x 256
        return outputs
        
    def block17(self, inputs, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
        """Inception-Renset-B
           Builds the 17x17 resnet block.
        """
        with tf.variable_scope(scope, 'Block17', [inputs], reuse=reuse):
            with tf.variable_scope('Branch_0'):
                tower_conv = slim.conv2d(inputs, 128, 1, scope='Conv2d_1x1')                                    # 17 x 17 x 128
            with tf.variable_scope('Branch_1'):
                tower_conv1_0 = slim.conv2d(inputs, 128, 1, scope='Conv2d_0a_1x1')                              # 17 x 17 x 128
                tower_conv1_1 = slim.conv2d(tower_conv1_0, 128, [1, 7],scope='Conv2d_0b_1x7')                   # 17 x 17 x 128
                tower_conv1_2 = slim.conv2d(tower_conv1_1, 128, [7, 1],scope='Conv2d_0c_7x1')                   # 17 x 17 x 128
            mixed = tf.concat([tower_conv, tower_conv1_2], 3)                                                   # 17 x 17 x 384
            up = slim.conv2d(mixed, inputs.get_shape()[3], 1, normalizer_fn=None,activation_fn=None, scope='Conv2d_1x1') # 17 x 17 x 896
            outputs = inputs + scale * up
            if activation_fn:
                outputs = activation_fn(outputs)                                                                # 17 x 17 x 896
        return outputs
        
    def block8(self, inputs, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
        """Inception-Resnet-C
           Builds the 8x8 resnet block.
        """
        with tf.variable_scope(scope, 'Block8', [inputs], reuse=reuse):
            with tf.variable_scope('Branch_0'):
                tower_conv = slim.conv2d(inputs, 192, 1, scope='Conv2d_1x1')                                    # 8 x 8 x 192
            with tf.variable_scope('Branch_1'):
                tower_conv1_0 = slim.conv2d(inputs, 192, 1, scope='Conv2d_0a_1x1')                              # 8 x 8 x 192
                tower_conv1_1 = slim.conv2d(tower_conv1_0, 192, [1, 3],scope='Conv2d_0b_1x3')                   # 8 x 8 x 192
                tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [3, 1],scope='Conv2d_0c_3x1')                   # 8 x 8 x 192
            mixed = tf.concat([tower_conv, tower_conv1_2], 3)                                                   # 8 x 8 x 384
            up = slim.conv2d(mixed, inputs.get_shape()[3], 1, normalizer_fn=None,activation_fn=None, scope='Conv2d_1x1') # 8 x 8 x 1792
            outputs = inputs + scale * up
            if activation_fn:
                outputs = activation_fn(outputs)                                                                # 8 x 8 x 1792
        return outputs
        
    def reduction_a(self,inputs, k, l, m, n):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(inputs, n, 3, stride=2, padding='VALID',scope='Conv2d_1a_3x3')             # 17 x 17 x 384
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(inputs, k, 1, scope='Conv2d_0a_1x1')                                    # 35 x 35 x 192
            tower_conv1_1 = slim.conv2d(tower_conv1_0, l, 3, scope='Conv2d_0b_3x3')                              # 35 x 35 x 192
            tower_conv1_2 = slim.conv2d(tower_conv1_1, m, 3, stride=2, padding='VALID',scope='Conv2d_1a_3x3')    # 17 x 17 x 256
        with tf.variable_scope('Branch_2'):
            tower_pool = slim.max_pool2d(inputs, 3, stride=2, padding='VALID',scope='MaxPool_1a_3x3')           # 17 x 17 x 256
        outputs = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)                                         # 17 x 17 x 896
        return outputs
        
    def reduction_b(self,inputs, l, m , n):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(inputs, 256, 1, scope='Conv2d_0a_1x1')                                     # 17 x 17 x 256
            tower_conv_1 = slim.conv2d(tower_conv, l, 3, stride=2,padding='VALID', scope='Conv2d_1a_3x3')     #  8 x  8 x 384
        with tf.variable_scope('Branch_1'):
            tower_conv1 = slim.conv2d(inputs, 256, 1, scope='Conv2d_0a_1x1')                                     # 17 x 17 x 256
            tower_conv1_1 = slim.conv2d(tower_conv1, m, 3, stride=2,padding='VALID', scope='Conv2d_1a_3x3')    #  8 x  8 x 384
        with tf.variable_scope('Branch_2'):
            tower_conv2 = slim.conv2d(inputs, 256, 1, scope='Conv2d_0a_1x1')                                     # 17 x 17 x 256
            tower_conv2_1 = slim.conv2d(tower_conv2, m, 3,scope='Conv2d_0b_3x3')                               # 17 x 17 x 256
            tower_conv2_2 = slim.conv2d(tower_conv2_1, n, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3') # 8  x  8 x 256
        with tf.variable_scope('Branch_3'):
            tower_pool = slim.max_pool2d(inputs, 3, stride=2, padding='VALID',scope='MaxPool_1a_3x3')            # 8  x  8 x 896
        outputs = tf.concat([tower_conv_1, tower_conv1_1,tower_conv2_2, tower_pool], 3)                          # 8  x  8 x 1792
        return outputs
        
    def reduction_v2(self, inputs):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(inputs, 96, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(inputs, 48, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5,scope='Conv2d_0b_5x5')
        with tf.variable_scope('Branch_2'):
            tower_conv2_0 = slim.conv2d(inputs, 64, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3,scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3,scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
            tower_pool = slim.avg_pool2d(inputs, 3, stride=1, padding='SAME',scope='AvgPool_0a_3x3')
            tower_pool_1 = slim.conv2d(tower_pool, 64, 1,scope='Conv2d_0b_1x1')
        outputs = tf.concat([tower_conv, tower_conv1_1,tower_conv2_2, tower_pool_1], 3)
        return outputs

    def fire_module(self,inputs,squeeze_depth,expand_depth,reuse=None,scope=None,outputs_collections=None):
        with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                outputs_collections=None):
                net = self.squeeze(inputs, squeeze_depth)
                outputs = self.expand(net, expand_depth)
                return outputs
                
    def squeeze(self, inputs, num_outputs):
        return slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='squeeze')
        
    def expand(self, inputs, num_outputs):
        with tf.variable_scope('expand'):
            e1x1 = slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='1x1')
            e3x3 = slim.conv2d(inputs, num_outputs, [3, 3], scope='3x3')
        return tf.concat([e1x1, e3x3], 3)
        
    def center_loss(self):
        """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
           (http://ydwen.github.io/papers/WenECCV16.pdf)
        """
        nrof_features = self.bottleneck_layer.get_shape()[1]
        # give nrof_classes center
        centers = tf.get_variable('centers', [self.config.nrof_classes, nrof_features], dtype=tf.float32,
            initializer=tf.constant_initializer(0), trainable=False)
        label = tf.reshape(self.y, [-1])
        # put center in the right place (according to label)
        centers_batch = tf.gather(centers, label)
        diff = (1 - self.config.resnet.center_loss_alfa) * (centers_batch - self.bottleneck_layer)
        centers = tf.scatter_sub(centers, label, diff)
        loss = tf.reduce_mean(tf.square(self.bottleneck_layer - centers_batch))
        return loss, centers    
