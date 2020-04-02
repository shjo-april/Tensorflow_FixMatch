# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import numpy as np
import tensorflow as tf

class WideResNet:
    def __init__(self, option):
        self.model_name = 'WideResNet'

        self.is_training = option['is_training']

        self.mean = option['mean']
        self.std = option['std']

        self.getter = option['getter']
        
        self.repeat = option['repeat']
        self.scales = option['scales']
        self.filters = option['filters']
        self.dropout = option['dropout']

        self.classes = option['classes']
        
        self.leaky_relu = lambda x: tf.nn.leaky_relu(x, alpha = 0.1)

        self.bn_args = dict(training = self.is_training, momentum = 0.999)
        self.conv_args = lambda k, f: dict(padding = 'same', kernel_initializer = tf.random_normal_initializer(stddev = tf.rsqrt(0.5 * k * k * f)))

    def residual(self, x0, filters, stride = 1, activate_before_residual=False):
        x = self.leaky_relu(tf.layers.batch_normalization(x0, **self.bn_args))
        if activate_before_residual:
            x0 = x
        
        x = tf.layers.conv2d(x, filters, 3, strides = stride, **self.conv_args(3, filters))
        x = self.leaky_relu(tf.layers.batch_normalization(x, **self.bn_args))
        x = tf.layers.conv2d(x, filters, 3, **self.conv_args(3, filters))

        if x0.get_shape()[3] != filters:
            x0 = tf.layers.conv2d(x0, filters, 1, strides = stride, **self.conv_args(1, filters))

        return x0 + x

    def forward(self, x):
        with tf.variable_scope('Classifier', reuse = tf.AUTO_REUSE, custom_getter = self.getter):
            x = x[..., ::-1] / 255.

            y = tf.layers.conv2d((x - self.mean) / self.std, 16, 3, **self.conv_args(3, 16))
            for scale in range(self.scales):
                y = self.residual(y, self.filters << scale, stride = 2 if scale else 1, activate_before_residual = scale == 0)
                for i in range(self.repeat - 1):
                    y = self.residual(y, self.filters << scale)

            y = self.leaky_relu(tf.layers.batch_normalization(y, **self.bn_args))
            y = embeds = tf.reduce_mean(y, [1, 2])

            if self.dropout and self.training:
                y = tf.nn.dropout(y, 1 - self.dropout)

            logits = tf.layers.dense(y, self.classes, kernel_initializer = tf.glorot_normal_initializer())
            predictions = tf.nn.softmax(logits, axis = -1)

        return {
            'logits' : logits,
            'embeds' : embeds,
            'predictions' : predictions,
        }

