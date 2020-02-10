# Copyright (C) 2020 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import cv2
import glob
import pickle

import numpy as np

def log_print(string, log_path = './log.txt'):
    with open(log_path, 'a+') as f:
        print(string)
        f.write(string + '\n')
    
def get_data(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def one_hot(label, classes):
    v = np.zeros((classes), dtype = np.float32)
    v[label] = 1.
    return v

def get_dataset(dataset_dir, n_labels = None):
    train_dic = {}
    test_dataset = []
    
    # get train dataset
    for file_path in glob.glob(dataset_dir + '/cifar10/data_batch_*'):
        data = get_data(file_path)
        data_length = len(data[b'filenames'])
        
        for i in range(data_length):
            label = int(data[b'labels'][i])
            image_data = data[b'data'][i]

            channel_size = 32 * 32        
            
            r = image_data[:channel_size]
            g = image_data[channel_size : channel_size * 2]
            b = image_data[channel_size * 2 : ]

            r = r.reshape((32, 32)).astype(np.uint8)
            g = g.reshape((32, 32)).astype(np.uint8)
            b = b.reshape((32, 32)).astype(np.uint8)

            image = cv2.merge((b, g, r))
            
            try:
                train_dic[label].append(image)
            except KeyError:
                train_dic[label] = []
                train_dic[label].append(image)

    # get test dataset
    data = get_data(dataset_dir + '/cifar10/test_batch')
    data_length = len(data[b'filenames'])
    
    for i in range(data_length):
        label = int(data[b'labels'][i])
        image_data = data[b'data'][i]

        channel_size = 32 * 32        
        
        r = image_data[:channel_size]
        g = image_data[channel_size : channel_size * 2]
        b = image_data[channel_size * 2 : ]

        r = r.reshape((32, 32)).astype(np.uint8)
        g = g.reshape((32, 32)).astype(np.uint8)
        b = b.reshape((32, 32)).astype(np.uint8)

        image = cv2.merge((b, g, r))
        test_dataset.append([image, one_hot(label, 10)])

    # semi superivsed learning
    if n_labels is not None:
        n_label_per_class = n_labels // 10

        labeled_dataset = []
        unlabeled_image_data = []
    
        for class_index in range(10):
            images = np.asarray(train_dic[class_index], dtype = np.uint8)
            label = one_hot(class_index, 10)

            np.random.shuffle(images)
            
            labeled_dataset += [[image, label] for image in images[:n_label_per_class]]
            unlabeled_image_data += [image for image in images[n_label_per_class:]]
            
        return labeled_dataset, unlabeled_image_data, test_dataset
    # fully supervised learning
    else:
        train_dataset = []

        for class_index in range(10):
            label = one_hot(class_index, 10)
            train_dataset += [[image, label] for image in train_dic[class_index]]

        return train_dataset, test_dataset
