# Copyright (C) 2020 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import sys
import os
import cv2
import time
import random
import argparse

import numpy as np
import tensorflow as tf

from queue import Queue

from core.WideResNet import *

from utils.Utils import *
from utils.Teacher_FixMatch import *
from utils.Tensorflow_Utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='FixMatch', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # gpu properties
    parser.add_argument('--use_gpu', dest='use_gpu', help='use gpu', default='0', type=str)
    
    # preprocessing
    parser.add_argument('--num_labels', dest='num_labels', help='num_labels', default=250, type=int)
    parser.add_argument('--num_threads', dest='num_threads', help='num_threads', default=4, type=int)

    # fixmatch properties
    parser.add_argument('--model_name', dest='model_name', help='model_name', default='WideResNet', type=str)
    
    parser.add_argument('--batch_size', dest='batch_size', help='batch_size', default=64, type=int)
    parser.add_argument('--unlabeled_ratio', dest='unlabeled_ratio', help='unlabeled_ratio', default=7, type=int)

    parser.add_argument('--learning_rate', dest='learning_rate', help='learning_rate', default=0.03, type=float)
    
    parser.add_argument('--ema_decay', dest='ema_decay', help='ema_decay', default=0.999, type=float)
    parser.add_argument('--weight_decay', dest='weight_decay', help='weight_decay', default=0.0005, type=float)
    parser.add_argument('--confidence_threshold', dest='confidence_threshold', help='confidence_threshold', default=0.95, type=float),

    parser.add_argument('--strongly_augment', dest='strongly_augment', help='strongly_augment', default='randaugment', type=str),

    parser.add_argument('--max_epochs', dest='max_epochs', help='max_epochs', default=1<<10, type=int)
    parser.add_argument('--train_kimgs', dest='train_kimgs', help='train_kimgs', default=1<<16, type=int)

    parser.add_argument('--log_iteration', dest='log_iteration', help='log_iteration', default=100, type=int)
    parser.add_argument('--val_iteration', dest='val_iteration', help='val_iteration', default=10000, type=int)
    
    return vars(parser.parse_args())

args = parse_args()

folder_name = '{}'.format(args['model_name'])
folder_name += '_cifar10@{}'.format(args['num_labels'])
folder_name += '_unlabeled_ratio@{}'.format(args['unlabeled_ratio'])

if args['ema_decay'] != -1: folder_name += '_#ema'

model_dir = './experiments/model/{}/'.format(folder_name)
tensorboard_dir = './experiments/tensorboard/{}'.format(folder_name)

ckpt_format = model_dir + '{}.ckpt'
log_txt_path = model_dir + 'log.txt'

os.environ["CUDA_VISIBLE_DEVICES"] = args['use_gpu']

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

open(log_txt_path, 'w').close()

#######################################################################################
# 1. Dataset
#######################################################################################
labeled_dataset, unlabeled_image_data, test_dataset = get_dataset('./dataset/', n_labels = args['num_labels'])

train_iteration = args['train_kimgs'] // args['batch_size'] * args['max_epochs']
test_iteration = len(test_dataset) // args['batch_size']

#######################################################################################
# 1.1. Info (Dataset)
#######################################################################################
log_print('# labeled dataset : {}'.format(len(labeled_dataset)), log_txt_path)
log_print('# unlabeled dataset : {}'.format(len(unlabeled_image_data)), log_txt_path)

log_print('# train iteration : {}'.format(train_iteration), log_txt_path)
log_print('# test iteration : {}'.format(test_iteration), log_txt_path)

#######################################################################################
# 2. Model
#######################################################################################
# 2.1. define placeholders.
x_image_var = tf.placeholder(tf.float32, [args['batch_size']] + [32, 32, 3])
x_label_var = tf.placeholder(tf.float32, [args['batch_size'], 10])

u_image_var = tf.placeholder(tf.float32, [args['batch_size'] * args['unlabeled_ratio'], 2] + [32, 32, 3])

is_training = tf.placeholder(tf.bool)
global_step = tf.placeholder(tf.int32)

# 2.2. build model.
logits_list = []
for image_var in tf.split(tf.concat([x_image_var, u_image_var[:, 0], u_image_var[:, 1]], axis = 0), 8):
    logits_list.append(WideResNet(image_var, is_training)['logits'])

logits_op = tf.concat(logits_list, axis = 0)

labeled_logits_op = logits_op[:args['batch_size']]
weak_logits_op, strong_logits_op = tf.split(logits_op[args['batch_size']:], 2)

# 2.3 calculate losses
labeled_loss_op = tf.nn.softmax_cross_entropy_with_logits(logits = labeled_logits_op, labels = x_label_var)
labeled_loss_op = tf.reduce_mean(labeled_loss_op)

pseudo_labels = tf.stop_gradient(tf.nn.softmax(weak_logits_op, axis = -1))
pseudo_masks = tf.to_float(tf.greater_equal(tf.reduce_max(pseudo_labels, axis = -1), args['confidence_threshold']))

unlabeled_loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = strong_logits_op, labels = tf.argmax(pseudo_labels, axis = -1))
unlabeled_loss_op = tf.reduce_mean(pseudo_masks * unlabeled_loss_op)

# 2.4. l2 regularization loss
train_vars = tf.trainable_variables()
l2_reg_loss_op = tf.add_n([tf.nn.l2_loss(var) for var in train_vars]) * args['weight_decay']

# 2.5. total loss
loss_op = labeled_loss_op + unlabeled_loss_op + l2_reg_loss_op

# 2.6. ema
if args['ema_decay'] != -1:
    ema = tf.train.ExponentialMovingAverage(decay = args['ema_decay'])
    ema_op = ema.apply(train_vars)

    predictions_op = WideResNet(x_image_var, False, getter = get_getter(ema))['predictions']
else:
    predictions_op = labeled_logits_op

correct_op = tf.equal(tf.argmax(predictions_op, axis = -1), tf.argmax(x_label_var, axis = -1))
accuracy_op = tf.reduce_mean(tf.cast(correct_op, tf.float32)) * 100

#######################################################################################
# 3. optimizer
#######################################################################################
learning_rate_ratio = tf.clip_by_value(tf.to_float(global_step) / train_iteration, 0, 1)
learning_rate = args['learning_rate'] * tf.cos(learning_rate_ratio * (7 * np.pi) / (2 * 8)) 

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_op = tf.train.MomentumOptimizer(learning_rate, momentum = 0.9, use_nesterov = True).minimize(loss_op, colocate_gradients_with_ops = True)
    train_op = tf.group(train_op, ema_op)

#######################################################################################
# 4. tensorboard
#######################################################################################
train_summary_dic = {
    'Loss/Total_Loss' : loss_op,
    'Loss/Labeled_Loss' : labeled_loss_op,
    'Loss/Unlabeled_Loss' : unlabeled_loss_op,
    'Loss/L2_Regularization_Loss' : l2_reg_loss_op,     

    'Accuracy/Train' : accuracy_op,
    
    'Monitors/Learning_rate' : learning_rate,
    'Monitors/Mask' : tf.reduce_mean(pseudo_masks),
}
train_summary_op = tf.summary.merge([tf.summary.scalar(name, train_summary_dic[name]) for name in train_summary_dic.keys()])

test_accuracy_var = tf.placeholder(tf.float32)
test_accuracy_op = tf.summary.scalar('Accuracy/Test', test_accuracy_var)

train_writer = tf.summary.FileWriter(tensorboard_dir)

#######################################################################################
# 5. Session, Saver
#######################################################################################
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(max_to_keep = 2)

#######################################################################################
# 6. create Thread
#######################################################################################
total_thread_data = 200
image_per_thread = total_thread_data // args['num_threads']

main_queue = Queue(image_per_thread)

thread_option = {
    'main_queue' : main_queue,

    'labeled_dataset' : labeled_dataset,
    'unlabeled_dataset' : unlabeled_image_data,

    'batch_size' : args['batch_size'],
    'unlabeled_ratio' : args['unlabeled_ratio'],

    'strongly_augment' : args['strongly_augment'],
}

train_threads = []

for i in range(args['num_threads']):
    log_print('# create thread : {}'.format(i), log_txt_path)

    train_thread = Teacher(thread_option)
    train_thread.start()
    
    train_threads.append(train_thread)

#######################################################################################
# 7. Train
#######################################################################################
best_test_accuracy = 0.0
train_ops = [train_op, loss_op, labeled_loss_op, unlabeled_loss_op, l2_reg_loss_op, accuracy_op, train_summary_op]

loss_list = []
labeled_loss_list = []
unlabeled_loss_list = []
l2_reg_loss_list = []
accuracy_list = []
train_time = time.time()

for step in range(1, train_iteration + 1):
    batch_x_image_data, batch_x_label_data, batch_u_image_data = main_queue.get()

    _feed_dict = {
        x_image_var : batch_x_image_data, 
        x_label_var : batch_x_label_data, 
        u_image_var : batch_u_image_data, 
        is_training : True,
        global_step : step,
    }
    _, loss, labeled_loss, unlabeled_loss, l2_reg_loss, accuracy, summary = sess.run(train_ops, feed_dict = _feed_dict)
    train_writer.add_summary(summary, step)

    loss_list.append(loss)
    labeled_loss_list.append(labeled_loss)
    unlabeled_loss_list.append(unlabeled_loss)
    l2_reg_loss_list.append(l2_reg_loss)
    accuracy_list.append(accuracy)
    
    if step % args['log_iteration'] == 0:
        loss = np.mean(loss_list)
        labeled_loss = np.mean(labeled_loss_list)
        unlabeled_loss = np.mean(unlabeled_loss_list)
        l2_reg_loss = np.mean(l2_reg_loss_list)
        accuracy = np.mean(accuracy_list)
        train_time = int(time.time() - train_time)
        
        log_print('[i] step = {}, loss = {:.4f}, labeled_loss = {:.4f}, unlabeled_loss = {:.4f}, l2_reg_loss = {:.4f}, accuracy = {:.2f}, train_time = {}sec'.format(step, loss, labeled_loss, unlabeled_loss, l2_reg_loss, accuracy, train_time), log_txt_path)
        
        loss_list = []
        labeled_loss_list = []
        unlabeled_loss_list = []
        l2_reg_loss_list = []
        accuracy_list = []
        train_time = time.time()
            
    if step % args['val_iteration'] == 0:
        test_time = time.time()
        test_accuracy_list = []
        
        for i in range(test_iteration):
            batch_data_list = test_dataset[i * args['batch_size'] : (i + 1) * args['batch_size']]

            batch_image_data = np.zeros((args['batch_size'], 32, 32, 3), dtype = np.float32)
            batch_label_data = np.zeros((args['batch_size'], 10), dtype = np.float32)
            
            for i, (image, label) in enumerate(batch_data_list):
                batch_image_data[i] = image.astype(np.float32)
                batch_label_data[i] = label.astype(np.float32)

            _feed_dict = {
                x_image_var : batch_image_data,
                x_label_var : batch_label_data,
                is_training : False
            }

            accuracy = sess.run(accuracy_op, feed_dict = _feed_dict)
            test_accuracy_list.append(accuracy)

        test_time = int(time.time() - test_time)
        test_accuracy = np.mean(test_accuracy_list)

        summary = sess.run(test_accuracy_op, feed_dict = {test_accuracy_var : test_accuracy})
        train_writer.add_summary(summary, step)

        if best_test_accuracy <= test_accuracy:
            best_test_accuracy = test_accuracy
            saver.save(sess, ckpt_format.format(step))            

        log_print('[i] step = {}, test_accuracy = {:.2f}, best_test_accuracy = {:.2f}, test_time = {}sec'.format(step, test_accuracy, best_test_accuracy, test_time), log_txt_path)

saver.save(sess, ckpt_format.format('end'))

for th in train_threads:
    th.train = False
    th.join()

