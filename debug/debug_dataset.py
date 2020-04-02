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
    os.environ['DISPLAY'] = ':0'

    flags = get_config()

    number_of_labels = 250
    labeled_dataset, unlabeled_dataset, valid_dataset, test_dataset = get_dataset_cifar10(number_of_labels)

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

    print(weakly_augment_func, strongly_augment_func)
    print(flags.weak_augmentation, flags.strong_augmentation)

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

    labeled_loader = Prefetch_using_queue(lambda q: Loader(q, labeled_loader_option), use_cores = flags.number_of_loader, max_size = flags.max_size_of_loader)
    labeled_decoder = Prefetch_using_queue(lambda q: Decoder(q, labeled_loader.main_queue, labeled_decoder_option), use_cores = flags.number_of_labeled_decoder, max_size = flags.max_size_of_labeled_decoder)
    labeled_batch_loader = Prefetch_using_queue(lambda q: Batch_Loader(q, labeled_decoder.main_queue, labeled_batch_loader_option), use_cores = flags.number_of_batch_loader, max_size = flags.max_size_of_batch_loader)

    unlabeled_loader = Prefetch_using_queue(lambda q: Loader(q, unlabeled_loader_option), use_cores = flags.number_of_loader, max_size = flags.max_size_of_loader)
    unlabeled_decoder = Prefetch_using_queue(lambda q: Decoder(q, unlabeled_loader.main_queue, unlabeled_decoder_option), use_cores = flags.number_of_unlabeled_decoder, max_size = flags.max_size_of_unlabeled_decoder)
    unlabeled_batch_loader = Prefetch_using_queue(lambda q: Batch_Loader(q, unlabeled_decoder.main_queue, unlabeled_batch_loader_option), use_cores = flags.number_of_batch_loader, max_size = flags.max_size_of_batch_loader)

    # Tensorflow
    x_image_var = tf.placeholder(tf.float32, [flags.batch_size] + [32, 32, 3])
    x_label_var = tf.placeholder(tf.float32, [flags.batch_size, 10])
    u_image_var = tf.placeholder(tf.float32, [flags.batch_size * flags.unlabeled_ratio, 2] + [32, 32, 3])
    
    train_generator = Generator({
            'labeled_batch_loader' : labeled_batch_loader, 
            'unlabeled_batch_loader' : unlabeled_batch_loader, 
            
            'placeholders' : [x_image_var, x_label_var, u_image_var], 
            'queue_size' : 5, 
    })

    x_image_op, x_label_op, u_image_op = train_generator.dequeue()

    # start
    labeled_loader.start()
    labeled_decoder.start()
    labeled_batch_loader.start()

    unlabeled_loader.start()
    unlabeled_decoder.start()
    unlabeled_batch_loader.start()

    sess = tf.Session()
    coord = tf.train.Coordinator()

    train_generator.set_session(sess)
    train_generator.set_coordinator(coord)
    train_generator.start()

    batch_timer = Timer()

    while True:
        batch_timer.tik()
        images, labels, uimages = sess.run([x_image_op, x_label_op, u_image_op])
        ms = batch_timer.tok(ms = True)
        
        print(labels[0])
        print(images.shape, labels.shape, uimages.shape, '{}ms'.format(ms))

        cv2.imshow('labeled', images[0].astype(np.uint8))
        cv2.imshow('weakly_unlabeled', uimages[0, 0].astype(np.uint8))
        cv2.imshow('strongly_unlabeled', uimages[0, 1].astype(np.uint8))
        cv2.waitKey(0)

