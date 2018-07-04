# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 13:15:17 2018

@author: jwm
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
from six import string_types,iteritems
import numpy as np

def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__)) # Automatically set a name if not provided.
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        
        layer_output = op(self, layer_input, *args, **kwargs) # Perform the operation and get the output.
        self.layers[name] = layer_output                      # Add to layer LUT.
        self.feed(layer_output)                               # This output is now the input for the next layer.
        return self                                           # Return self for chained calls.

    return layer_decorated

class Network(object):

    def __init__(self, inputs, trainable=True):
        
        self.inputs = inputs            # The input nodes for this network
        self.terminals = []             # The current list of terminal nodes
        self.layers = dict(inputs)      # Mapping from layer names to layers
        self.trainable = trainable      # If true, the resulting variables are set as trainable
        self.setup()

    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')
        
    def load(self, weight_path, sess, ignore_missing=False):
        '''
           Load network weights.
           sess: The current TensorFlow session
           ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(weight_path, encoding='latin1').item() #pylint: disable=no-member

        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in iteritems(data_dict[op_name]):
                    try:
                        var = tf.get_variable(param_name)
                        sess.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        '''
            Set the input(s) for the next operation by replacing the terminal nodes.
            The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, string_types):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''
            Returns the current network output.
        '''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''
            Returns an index-suffixed unique name for the given prefix.
            This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        '''
            Creates a new TensorFlow variable.
        '''
        return tf.get_variable(name, shape, trainable=self.trainable)

    def validate_padding(self, padding):
        '''
            Verifies that the padding is one of the supported ones.
        '''
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self,inp,k_h,k_w,c_o,s_h,s_w,name,relu=True,padding='SAME',group=1,biased=True):
        """ parameters
            inp: input images
            k_h: kernal height
            k_w: kernal width
            c_o: output channel
            s_h: stride height
            s_w: stride width
        """
        self.validate_padding(padding)       # Verify that the padding is acceptable
        c_i = int(inp.get_shape()[-1])       # Get the number of channels in the input
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)  # Convolution for a given input and kernel
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i // group, c_o])
            output = convolve(inp, kernel)        # This is the common-case. Convolve the input without any further complications.
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases) # default name: "BiasAdd"
            if relu:
                output = tf.nn.relu(output, name=scope.name)  # ReLU non-linearity
            return output

    @layer
    def prelu(self, inp, name):
        with tf.variable_scope(name):
            i = int(inp.get_shape()[-1])
            alpha = self.make_var('alpha', shape=(i,))
            output = tf.nn.relu(inp) + tf.multiply(alpha, -tf.nn.relu(-inp))
        return output

    @layer
    def max_pool(self, inp, k_h, k_w, s_h, s_w, name, padding='SAME'):
        """ parameters
            inp: input images
            k_h: kernal height
            k_w: kernal width
            s_h: stride height
            s_w: stride width
        """
        self.validate_padding(padding)
        return tf.nn.max_pool(inp,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def fc(self, inp, num_out, name, relu=True):
        """ parameters
            inp    : input images
            num_out: number of classes
        """
        with tf.variable_scope(name):
            input_shape = inp.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= int(d)
                feed_in = tf.reshape(inp, [-1, dim])
            else:
                feed_in, dim = (inp, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=name)
            return fc

    @layer
    def softmax(self, target, axis, name=None):
        """
            Multi dimensional softmax,
            refer to https://github.com/tensorflow/tensorflow/issues/210
            compute softmax along the dimension of target
            the native softmax only supports batch_size x dimension
        """
        max_axis = tf.reduce_max(target, axis, keep_dims=True)
        target_exp = tf.exp(target-max_axis)
        normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
        softmax = tf.div(target_exp, normalize, name)
        return softmax