# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import cv2
import sys
import copy
import time
import glob
import json
import random
import pickle
import argparse

import numpy as np
import tensorflow as tf

import multiprocessing as mp

from core.config import *
from core.classifier import *
from core.augment.augmentors import *

from data.loader import *
from data.decoder import *
from data.batch_loader import *

from data.prefetch import *
from data.generator import *

from utility.utils import *
from utility.dataset_utils import *
from utility.tensorflow_utils import *

from utility.timer import *
from utility.logger import *

if __name__ == '__main__':
    #######################################################################################
    # 1. Config
    #######################################################################################
    flags = get_config()
    set_seed(flags.seed)
    
    num_gpu = len(flags.use_gpu.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.use_gpu
    
    iteration = flags.train_kimgs // flags.batch_size
    flags.max_iteration = flags.max_epochs * iteration
    
    model_name = 'FixMatch_cifar@{}'.format(flags.number_of_labels)
    model_dir = './experiments/model/{}/'.format(model_name)
    tensorboard_dir = './experiments/tensorboard/{}'.format(model_name)

    ckpt_format = model_dir + '{}.ckpt'

    log_txt_path = model_dir + 'log.txt'
    log_csv_path = model_dir + 'log.csv'
    valid_log_path = model_dir + 'valid_log.csv'
    test_log_path = model_dir + 'test_log.csv'

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    if os.path.isfile(log_txt_path):
        open(log_txt_path, 'w').close()

    # log_func = lambda x: print(x)
    log_func = lambda x: log_print(x, log_txt_path)
    csv_log_func = lambda x: csv_print(x, log_csv_path)
    valid_log_func = lambda x: csv_print(x, valid_log_path)
    test_log_func = lambda x: csv_print(x, test_log_path)

    #######################################################################################
    # 1. Dataset
    #######################################################################################
    labeled_dataset, unlabeled_dataset, valid_dataset, test_dataset = get_dataset_cifar10(flags.number_of_labels)

    #######################################################################################
    # 1.1. Info (Dataset)
    #######################################################################################
    log_func('\n')
    log_func('# labeled_dataset : {}'.format(len(labeled_dataset)))
    log_func('# unlabeled_dataset : {}'.format(len(unlabeled_dataset)))
    log_func('# valid_dataset : {}'.format(len(valid_dataset)))
    log_func('# test_dataset : {}'.format(len(test_dataset)))

    #######################################################################################
    # 2. Generator & Queue
    #######################################################################################

    #######################################################################################
    # 2.1. Select DataAugmentations.
    #######################################################################################
    # for weakly augmented function
    if flags.weak_augmentation == 'flip_and_crop':
        weakly_augment_func = Flip_and_Crop((32, 32))

    # for strongly augmented function
    if flags.strong_augmentation == 'randaugment':
        strongly_augment_func = DataAugmentation([
            Flip_and_Crop((32, 32)),
            RandAugmentation(),
        ])
    elif flags.strong_augmentation == 'randaugment':
        strongly_augment_func = Flip_and_Crop((32, 32))

    log_func('1. weakly={}, strongly={}'.format(weakly_augment_func, strongly_augment_func))
    log_func('2. weakly={}, strongly={}'.format(flags.weak_augmentation, flags.strong_augmentation))

    #######################################################################################
    # 2.2. Set option.
    #######################################################################################
    # for labeled option
    labeled_loader_option = {
        'debug' : False,
        'bShuffle' : True,

        'dataset' : labeled_dataset,
        'number_of_loading_dataset' : flags.number_of_loading_dataset,
    }

    labeled_decoder_option = {
        'debug' : False,
        'use_label' : True,
        'augment_func' : weakly_augment_func,
    }

    labeled_batch_loader_option = {
        'debug' : False,

        'batch_size' : flags.batch_size,
        'batch_length' : 2,
    }

    # for unlabeled option
    unlabeled_loader_option = {
        'debug' : False,
        'bShuffle' : True,
        
        'dataset' : unlabeled_dataset,
        'number_of_loading_dataset' : flags.number_of_loading_dataset,
    }

    unlabeled_decoder_option = {
        'debug' : False,
        'use_label' : False,
        'augment_func' : [weakly_augment_func, strongly_augment_func],
    }
    
    unlabeled_batch_loader_option = {
        'debug' : False,

        'batch_size' : flags.batch_size * flags.unlabeled_ratio,
        'batch_length' : 1,
    }

    #######################################################################################
    # 2.3. Create loader, decoder, batch_loader, and generator
    #######################################################################################
    labeled_loader = Prefetch_using_queue(lambda q: Loader(q, labeled_loader_option), use_cores = flags.number_of_loader, max_size = flags.max_size_of_loader)
    labeled_decoder = Prefetch_using_queue(lambda q: Decoder(q, labeled_loader.main_queue, labeled_decoder_option), use_cores = flags.number_of_labeled_decoder, max_size = flags.max_size_of_labeled_decoder)
    labeled_batch_loader = Prefetch_using_queue(lambda q: Batch_Loader(q, labeled_decoder.main_queue, labeled_batch_loader_option), use_cores = flags.number_of_batch_loader, max_size = flags.max_size_of_batch_loader)

    unlabeled_loader = Prefetch_using_queue(lambda q: Loader(q, unlabeled_loader_option), use_cores = flags.number_of_loader, max_size = flags.max_size_of_loader)
    unlabeled_decoder = Prefetch_using_queue(lambda q: Decoder(q, unlabeled_loader.main_queue, unlabeled_decoder_option), use_cores = flags.number_of_unlabeled_decoder, max_size = flags.max_size_of_unlabeled_decoder)
    unlabeled_batch_loader = Prefetch_using_queue(lambda q: Batch_Loader(q, unlabeled_decoder.main_queue, unlabeled_batch_loader_option), use_cores = flags.number_of_batch_loader, max_size = flags.max_size_of_batch_loader)

    # create placeholders.
    x_image_var = tf.placeholder(tf.float32, [flags.batch_size] + [32, 32, 3])
    x_label_var = tf.placeholder(tf.float32, [flags.batch_size, 10])
    u_image_var = tf.placeholder(tf.float32, [flags.batch_size * flags.unlabeled_ratio, 2] + [32, 32, 3])

    train_generator = Generator({
            'labeled_batch_loader' : labeled_batch_loader, 
            'unlabeled_batch_loader' : unlabeled_batch_loader, 
            
            'placeholders' : [x_image_var, x_label_var, u_image_var], 
            'queue_size' : 5, 
    })

    #######################################################################################
    # 3. Model
    #######################################################################################
    # 3.1. set parameters.
    classifier_option = {
        'is_training' : True,

        'mean' : [0.49139968, 0.48215841, 0.44653091],
        'std' : [0.24703223, 0.24348513, 0.26158784],
        
        'getter' : None,

        'repeat' : 4,
        'scales' : int(np.ceil(np.log2(32))) - 2,
        'filters' : 32,
        'dropout' : 0,

        'classes' : 10,
    }
    classifier_func = lambda x: Classifier(classifier_option).forward(x)['logits']

    # 3.2. build model.
    x_image_op, x_label_op, u_image_op = train_generator.dequeue()
    
    # concatenate all images which are labeled images, unlabeled images.
    total_image_op = tf.concat([x_image_op, u_image_op[:, 0], u_image_op[:, 1]], axis = 0)

    # interleave
    skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    total_image_op = interleave(total_image_op, 2 * flags.unlabeled_ratio + 1)

    # split 1/8
    image_ops = tf.split(total_image_op, 1 + flags.unlabeled_ratio)

    '''
    [i] build model (gpu_id = 0, device_index = 0, reuse = False)
    [i] build model (gpu_id = 1, device_index = 0, reuse = True)
    [i] build model (gpu_id = 2, device_index = 0, reuse = True)
    [i] build model (gpu_id = 3, device_index = 0, reuse = True)
    [i] build model (gpu_id = 4, device_index = 0, reuse = True)
    [i] build model (gpu_id = 5, device_index = 0, reuse = True)
    [i] build model (gpu_id = 6, device_index = 0, reuse = True)
    [i] build model (gpu_id = 7, device_index = 0, reuse = True)
    [i] logits_op = Tensor("concat_1:0", shape=(960, 10), dtype=float32)
    '''
    logits_ops = []
    for gpu_id, image_op in enumerate(image_ops):
        with tf.device(tf.DeviceSpec(device_type = "GPU", device_index = gpu_id % num_gpu)):
            with tf.variable_scope(tf.get_variable_scope(), reuse = gpu_id > 0):
                logits_ops.append(classifier_func(image_op))
                log_func('[i] build model (gpu_id = {}, device_index = {}, reuse = {})'.format(gpu_id, gpu_id % num_gpu, gpu_id > 0))

    logits_op = tf.concat(logits_ops, axis = 0)

    # de-interleave
    post_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if v not in skip_ops]
    logits_op = de_interleave(logits_op, 2 * flags.unlabeled_ratio + 1)

    log_func('[i] logits_op = {}'.format(logits_op))

    # 3.3. calculate labeled loss and unlabeled loss.
    labeled_logits_op = logits_op[:flags.batch_size]
    weak_logits_op, strong_logits_op = tf.split(logits_op[flags.batch_size:], 2)

    labeled_loss_op = tf.nn.softmax_cross_entropy_with_logits(logits = labeled_logits_op, labels = x_label_op)
    labeled_loss_op = tf.reduce_mean(labeled_loss_op)

    pseudo_labels = tf.stop_gradient(tf.nn.softmax(weak_logits_op, axis = -1))
    pseudo_masks = tf.to_float(tf.greater_equal(tf.reduce_max(pseudo_labels, axis = -1), flags.confidence_threshold))

    unlabeled_loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = strong_logits_op, labels = tf.argmax(pseudo_labels, axis = -1))
    unlabeled_loss_op = tf.reduce_mean(pseudo_masks * unlabeled_loss_op) * flags.lambda_u

    train_vars = get_model_vars()
    l2_reg_loss_op = tf.add_n([tf.nn.l2_loss(var) for var in train_vars if 'kernel' in var.name]) * flags.weight_decay

    loss_op = labeled_loss_op + unlabeled_loss_op + l2_reg_loss_op

    # 3.4. build ema model.
    ema = tf.train.ExponentialMovingAverage(decay = flags.ema_decay)
    ema_op = ema.apply(train_vars)

    # 3.5. evaluate train accuracy.
    train_correct_op = tf.equal(tf.argmax(labeled_logits_op, axis = -1), tf.argmax(x_label_op, axis = -1))
    train_accuracy_op = tf.reduce_mean(tf.cast(train_correct_op, tf.float32)) * 100

    #######################################################################################
    # 3. Optimizer
    #######################################################################################
    global_step = tf.placeholder(tf.float32)
    
    learning_rate_ratio = tf.clip_by_value(tf.to_float(global_step) / flags.max_iteration, 0, 1)
    learning_rate = flags.init_learning_rate * tf.cos(learning_rate_ratio * (7 * np.pi) / (2 * 8)) 

    with tf.control_dependencies(post_ops):
        train_op = tf.train.MomentumOptimizer(learning_rate, momentum = 0.9, use_nesterov = True).minimize(loss_op, colocate_gradients_with_ops = True)
        train_op = tf.group(train_op, ema_op)
    
    #######################################################################################
    # 4. Test 
    #######################################################################################
    test_image_var = tf.placeholder(tf.float32, [None, 32, 32, 3])
    test_label_var = tf.placeholder(tf.float32, [None, 10])

    test_option = {
        'is_training' : False,

        'mean' : [0.49139968, 0.48215841, 0.44653091],
        'std' : [0.24703223, 0.24348513, 0.26158784],
        
        'getter' : get_getter(ema),

        'repeat' : 4,
        'scales' : int(np.ceil(np.log2(32))) - 2,
        'filters' : 32,
        'dropout' : 0,

        'classes' : 10,
    }
    classifier_func = lambda x: Classifier(test_option).forward(x)['predictions']

    with tf.device(tf.DeviceSpec(device_type = "GPU", device_index = 0)):
        with tf.variable_scope(tf.get_variable_scope(), reuse = True):
            predictions_op = classifier_func(test_image_var)

    test_correct_op = tf.equal(tf.argmax(predictions_op, axis = -1), tf.argmax(test_label_var, axis = -1))
    test_accuracy_op = tf.reduce_mean(tf.cast(test_correct_op, tf.float32)) * 100

    #######################################################################################
    # 5. Tensorboard
    #######################################################################################
    train_summary_dic = {
        'losses/total_loss' : loss_op,
        'losses/labeled_loss' : labeled_loss_op,
        'losses/unlabeled_loss' : unlabeled_loss_op,
        'losses/l2_regularization' : l2_reg_loss_op,     

        'monitors/train_accuracy' : train_accuracy_op,
        
        'monitors/learning_rate' : learning_rate,
        'monitors/pseudo_mask' : tf.reduce_mean(pseudo_masks),
    }
    train_summary_op = tf.summary.merge([tf.summary.scalar(name, train_summary_dic[name]) for name in train_summary_dic.keys()])

    train_writer = tf.summary.FileWriter(tensorboard_dir)

    #######################################################################################
    # 6. create Session and Saver
    #######################################################################################
    sess = tf.Session()
    coord = tf.train.Coordinator()

    saver = tf.train.Saver(max_to_keep = 2)

    #######################################################################################
    # 7. initialize
    #######################################################################################
    sess.run(tf.global_variables_initializer())

    labeled_loader.start()
    labeled_decoder.start()
    labeled_batch_loader.start()
    log_func('[i] labeled objects = {}'.format([labeled_loader, labeled_decoder, labeled_batch_loader]))

    unlabeled_loader.start()
    unlabeled_decoder.start()
    unlabeled_batch_loader.start()
    log_func('[i] unlabeled objects = {}'.format([unlabeled_loader, unlabeled_decoder, unlabeled_batch_loader]))

    train_generator.set_session(sess)
    train_generator.set_coordinator(coord)
    train_generator.start()

    #######################################################################################
    # 8. Train
    #######################################################################################
    best_valid_accuracy = 0.0
    best_test_accuracy = 0.0

    train_timer = Timer()
    valid_timer = Timer()
    test_timer = Timer()

    train_logger = Logger(
        [
            'total_loss',
            'labeled_loss',
            'unlabeled_loss',
            'l2_regularization',
            'train_accuracy',

        ],
        [
            'total_loss={:04.6f}',
            'labeled_loss={:04.6f}',
            'unlabeled_loss={:04.6f}',
            'l2_regularization={:04.6f}',
            'train_accuracy={:02.2f}%',
        ]
    )

    valid_logger = Logger(['Accuracy'],['valid_accuracy={:02.2f}%'])
    test_logger = Logger(['Accuracy'],['test_accuracy={:02.2f}%'])

    csv_log_func(train_logger.names)
    valid_log_func(valid_logger.names)
    test_log_func(test_logger.names)

    train_timer.tik()
    train_ops = [
        train_op, 
        loss_op, 
        labeled_loss_op,
        unlabeled_loss_op,
        l2_reg_loss_op, 
        train_accuracy_op,
        train_summary_op
    ]

    for step in range(1, flags.max_iteration + 1):
        data  = sess.run(train_ops, feed_dict = {
            global_step : step,
        })
        
        train_logger.update(data[1:-1])
        train_writer.add_summary(data[-1], step)
        
        if step % flags.log_iteration == 0:
            loader_size = unlabeled_loader.get_size()
            decoder_size = unlabeled_decoder.get_size()
            batch_loader_size = unlabeled_batch_loader.get_size()
            
            log_string = train_logger.log()
            log_string = '[i] step={} '.format(step) + log_string
            log_string += 'loader_size={} '.format(loader_size)
            log_string += 'decoder_size={} '.format(decoder_size)
            log_string += 'batch_loader_size={} '.format(batch_loader_size)
            log_string += 'train_sec={}sec '.format(train_timer.tok())
            
            log_func(log_string)
            csv_log_func(train_logger.get_data())
            
            train_logger.clear()
            train_timer.tik()
        
        #######################################################################################
        # 10. Validation
        #######################################################################################
        if step % flags.valid_iteration == 0:
            # validation
            valid_timer.tik()
            valid_logger.clear()

            for i in range(len(valid_dataset) // flags.batch_size):
                batch_data_list = valid_dataset[i * flags.batch_size : (i + 1) * flags.batch_size]

                batch_image_data = np.zeros((flags.batch_size, 32, 32, 3), dtype = np.float32)
                batch_label_data = np.zeros((flags.batch_size, 10), dtype = np.float32)
                
                for i, (image, label) in enumerate(batch_data_list):
                    batch_image_data[i] = image.astype(np.float32)
                    batch_label_data[i] = label.astype(np.float32)

                _feed_dict = {
                    test_image_var : batch_image_data,
                    test_label_var : batch_label_data,
                }
                accuracy = sess.run(test_accuracy_op, feed_dict = _feed_dict)
                valid_logger.update([accuracy])
            
            [valid_accuracy] = valid_logger.get_data()
            if best_valid_accuracy <= valid_accuracy:
                best_valid_accuracy = valid_accuracy
                saver.save(sess, ckpt_format.format(step))

            valid_log_func(valid_logger.get_data())

            log_string = valid_logger.log()
            log_string = '[i] step={} '.format(step) + log_string
            log_string += 'best_valid_accuracy={:02.2f}% valid_sec={}sec'.format(best_valid_accuracy, valid_timer.tok())

            log_func(log_string)

            # test
            test_timer.tik()
            test_logger.clear()

            for i in range(len(test_dataset) // flags.batch_size):
                batch_data_list = test_dataset[i * flags.batch_size : (i + 1) * flags.batch_size]

                batch_image_data = np.zeros((flags.batch_size, 32, 32, 3), dtype = np.float32)
                batch_label_data = np.zeros((flags.batch_size, 10), dtype = np.float32)
                
                for i, (image, label) in enumerate(batch_data_list):
                    batch_image_data[i] = image.astype(np.float32)
                    batch_label_data[i] = label.astype(np.float32)

                _feed_dict = {
                    test_image_var : batch_image_data,
                    test_label_var : batch_label_data,
                }
                accuracy = sess.run(test_accuracy_op, feed_dict = _feed_dict)
                test_logger.update([accuracy])
            
            [test_accuracy] = test_logger.get_data()
            if best_test_accuracy <= test_accuracy:
                best_test_accuracy = test_accuracy
                saver.save(sess, ckpt_format.format(step))
            
            test_log_func(test_logger.get_data())

            log_string = test_logger.log()
            log_string = '[i] step={} '.format(step) + log_string
            log_string += 'best_test_accuracy={:02.2f}% test_sec={}sec'.format(best_test_accuracy, test_timer.tok())

            log_func(log_string)

    saver.save(sess, ckpt_format.format('end'))

