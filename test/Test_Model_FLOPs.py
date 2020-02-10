# Copyright (C) 2020 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import sys
sys.path.insert(1, './')

import numpy as np

from core.WideResNet import *
from utils.Tensorflow_Utils import *

image_var = tf.placeholder(tf.float32, [None, 32, 32, 3])
is_training = tf.placeholder(tf.bool)

logits, predictions = WideResNet(image_var, is_training, filters = 32, repeat = 4)

vars = tf.trainable_variables()
sess = tf.Session()

model_summary(vars, sess.graph, './test/WideResNet.txt')
