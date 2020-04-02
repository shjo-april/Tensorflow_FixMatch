# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import cv2
import glob
import pickle

import numpy as np

from utility.utils import *

def get_data(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def get_dataset_cifar10(n_label, valid_size = 500):
    train_dic = {label : [] for label in range(10)}

    train_labeled_dataset = []
    train_unlabeled_dataset = []

    valid_dataset = []
    test_dataset = []

    dataset_dir = './dataset/'
    
    channel_size = 32 * 32
    n_label_per_class = n_label // 10

    # load dataset for training.
    for file_path in glob.glob(dataset_dir + "data_batch_*"):
        data = get_data(file_path)
        data_length = len(data[b'filenames'])
        
        for i in range(data_length):
            label = int(data[b'labels'][i])
            image_data = data[b'data'][i]
            
            r = image_data[:channel_size].reshape((32, 32)).astype(np.uint8)
            g = image_data[channel_size : channel_size * 2].reshape((32, 32)).astype(np.uint8)
            b = image_data[channel_size * 2 : ].reshape((32, 32)).astype(np.uint8)

            image = cv2.merge((b, g, r))
            train_dic[label].append(image)

    # split labeled dataset, unlabeled dataset and validation dataset.
    for i in range(10):
        image_data = train_dic[i]
        label_data = single_one_hot(i, 10)

        np.random.shuffle(image_data)
        
        train_labeled_dataset += [[image, label_data] for image in image_data[:n_label_per_class]]
        train_unlabeled_dataset += [image for image in image_data[n_label_per_class:-valid_size]]
        valid_dataset += [[image, label_data] for image in image_data[-valid_size:]]
    
    # load dataset for testing.
    data = get_data(dataset_dir + 'test_batch')
    data_length = len(data[b'filenames'])
    
    for i in range(data_length):
        label = int(data[b'labels'][i])
        image_data = data[b'data'][i]

        r = image_data[:channel_size].reshape((32, 32)).astype(np.uint8)
        g = image_data[channel_size : channel_size * 2].reshape((32, 32)).astype(np.uint8)
        b = image_data[channel_size * 2 : ].reshape((32, 32)).astype(np.uint8)

        image = cv2.merge((b, g, r))
        test_dataset.append([image, single_one_hot(label, 10)])
    
    return train_labeled_dataset, train_unlabeled_dataset, valid_dataset, test_dataset