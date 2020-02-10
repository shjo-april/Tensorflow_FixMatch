# Copyright (C) 2020 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import cv2
import copy
import time
import random

import numpy as np

from threading import Thread

from core.DataAugment import *
from core.randaugment.augment import *

from utils.Utils import *
from utils.Timer import *

class Teacher(Thread):
    
    def __init__(self, option):
        Thread.__init__(self)

        self.train = True
        self.timer = Timer()
        self.main_queue = option['main_queue']
        
        self.option = option
        
        self.labeled_batch_size = self.option['batch_size']
        self.unlabeled_batch_size = self.option['batch_size'] * self.option['unlabeled_ratio']

        self.labeled_dataset = copy.deepcopy(self.option['labeled_dataset'])
        self.unlabeled_dataset = copy.deepcopy(self.option['unlabeled_dataset'])

        self.weakly_augment = WeaklyAugment()
        self.rand_augment = RandAugment()

        if self.option['strongly_augment'] == 'randaugment':
            self.strongly_augment = self.rand_augment
        else:
            assert False, "[!] TODO : CTAugment"

    def run(self):
        while self.train:
            while self.main_queue.full() and self.train:
                time.sleep(0.1)
                continue
            
            if not self.train:
                break
            
            batch_x_image_data = []
            batch_x_label_data = []
            batch_u_image_data = []

            for data in random.sample(self.labeled_dataset, self.labeled_batch_size):
                image, label = data
                image = self.weakly_augment(image.copy())

                batch_x_image_data.append(image)
                batch_x_label_data.append(label)
            
            for image in random.sample(self.unlabeled_dataset, self.unlabeled_batch_size):
                w_image = self.weakly_augment(image.copy())
                s_image = self.strongly_augment(image.copy())
                
                batch_u_image_data.append([w_image, s_image])
                
            batch_x_image_data = np.asarray(batch_x_image_data, dtype = np.float32)
            batch_x_label_data = np.asarray(batch_x_label_data, dtype = np.float32)
            batch_u_image_data = np.asarray(batch_u_image_data, dtype = np.float32)

            try:
                self.main_queue.put_nowait([
                    batch_x_image_data, 
                    batch_x_label_data,
                    batch_u_image_data
                ])
            except:
                pass

