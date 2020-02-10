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
        
        self.option = option

        self.batch_size = self.option['batch_size']
        self.main_queue = self.option['main_queue']
        self.train_data_list = copy.deepcopy(self.option['labeled_dataset'])

        self.weakly_augment = WeaklyAugment()
        self.rand_augment = RandAugment()
        
        if option['augment'] == 'randaugment':
            self.augment = self.rand_augment
        else:
            self.augment = self.weakly_augment
        
    def run(self):
        while self.train:
            while self.main_queue.full() and self.train:
                time.sleep(0.1)
                continue

            if not self.train:
                break
            
            batch_image_data = []
            batch_label_data = []
            
            for data in random.sample(self.train_data_list, self.batch_size):
                image, label = data
                image = self.augment(image)

                batch_image_data.append(image)
                batch_label_data.append(label)
            
            batch_image_data = np.asarray(batch_image_data, dtype = np.float32)
            batch_label_data = np.asarray(batch_label_data, dtype = np.float32)
            
            try:
                self.main_queue.put_nowait([batch_image_data, batch_label_data])
            except:
                pass
