# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import cv2
import sys
import time
import random

import numpy as np
import tensorflow as tf

import multiprocessing as mp

from utility.timer import *

class Decoder(mp.Process):
    def __init__(self, main_queue, queue_for_dataset, option):
        super().__init__()
        self.daemon = True
        
        self.main_queue = main_queue
        self.queue_for_dataset = queue_for_dataset
        
        self.use_label = option['use_label']
        self.augment_func = option['augment_func']

        self.debug = option['debug']
        self.timer = Timer()
    
    def run(self):
        while True:
            dataset = self.queue_for_dataset.get()
            
            for data in dataset:
                if self.debug:
                    self.timer.tik()
                
                if self.use_label:
                    image_data, label_data = data
                else:
                    image_data = data

                if image_data is None:
                    continue
                
                if self.use_label:
                    image_data = self.augment_func(image_data)
                else:
                    image_data = [self.augment_func[0](image_data), self.augment_func[1](image_data)]
                
                if self.debug:
                    self.timer.tok()
                    print('[Debug] decode_func : {}ms'.format(self.timer.tok(ms = True)))
                
                if self.use_label:
                    self.main_queue.put([image_data, label_data])
                else:
                    self.main_queue.put([image_data])
                
            del dataset

