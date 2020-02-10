# Copyright (C) 2020 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt

batch_size = 64
train_kimages = 1024

learning_rate = 0.03

train_iteration = train_kimages // 64 * 1024

global_step = tf.placeholder(tf.float32)

learning_rate_ratio = tf.clip_by_value(tf.to_float(global_step) / train_iteration, 0, 1)
learning_rate_op = learning_rate * tf.cos(learning_rate_ratio * (7 * np.pi) / (2 * 8)) 

sess = tf.Session()

step_list = []
learning_rate_list = []

for step in range(train_iteration):
    lr = sess.run(learning_rate_op, feed_dict = {global_step : step})

    step_list.append(step)
    learning_rate_list.append(lr)

print(np.min(learning_rate_list), np.max(learning_rate_list))

plt.plot(step_list, learning_rate_list)
plt.savefig('./test/cosine.png')
# plt.show()

