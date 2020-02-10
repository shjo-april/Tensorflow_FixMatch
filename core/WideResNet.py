# Copyright (C) 2020 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import numpy as np
import tensorflow as tf

def WideResNet(image_var, is_training, scales = int(np.ceil(np.log2(32))) - 2, filters = 32, repeat = 4, getter = None):
    # bn_args = dict(training = is_training, momentum = 0.999)
    bn_args = dict(training = is_training, momentum = 0.9, epsilon = 1e-5, scale = True)
    
    def conv_args(k, f):
        return dict(padding = 'same', kernel_initializer = tf.random_normal_initializer(stddev = tf.rsqrt(0.5 * k * k * f)))

    def residual(x0, filters, stride = 1, activate_before_residual = False):
        x = tf.layers.batch_normalization(x0, **bn_args)
        x = tf.nn.leaky_relu(x, alpha = 0.1)
        if activate_before_residual:
            x0 = x
        
        x = tf.layers.conv2d(x, filters, 3, strides = stride, **conv_args(3, filters))

        x = tf.layers.batch_normalization(x, **bn_args)
        x = tf.nn.leaky_relu(x, alpha = 0.1)
        x = tf.layers.conv2d(x, filters, 3, **conv_args(3, filters))

        if x0.get_shape()[-1] != filters:
            x0 = tf.layers.conv2d(x0, filters, 1, strides = stride, **conv_args(1, filters))

        return x + x0

    with tf.variable_scope('WideResNet', reuse = tf.AUTO_REUSE, custom_getter = getter):
        x = image_var[..., ::-1] / 255.
        x = (x - [0.49139968, 0.48215841, 0.44653091]) / [0.24703223, 0.24348513, 0.26158784]
        
        x = tf.layers.conv2d(x, 16, 3, **conv_args(3, 16))

        for scale in range(scales):
            x = residual(x, filters << scale, stride = 2 if scale else 1, activate_before_residual = scale == 0)
            for i in range(repeat - 1):
                x = residual(x, filters << scale)

        x = tf.layers.batch_normalization(x, **bn_args)
        x = tf.nn.leaky_relu(x, alpha = 0.1)
        x = tf.reduce_mean(x, axis = [1, 2]) # GAP

        logits = tf.layers.dense(x, 10, kernel_initializer = tf.glorot_normal_initializer())
        predictions = tf.nn.softmax(logits, axis = -1)

    return {
        'logits' : logits,
        'predictions' : predictions,
    }

