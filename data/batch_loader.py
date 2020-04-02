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

class Batch_Loader(mp.Process):
    def __init__(self, main_queue, queue_for_dataset, option):
        super().__init__()
        self.daemon = True
        
        self.main_queue = main_queue
        self.queue_for_dataset = queue_for_dataset
        
        self.batch_size = option['batch_size']
        self.batch_length = option['batch_length']
        
        self.debug = option['debug']
        self.timer = Timer()
        
        self.init_dataset()
    
    def init_dataset(self):
        self.timer.tik()
        self.batch_dataset = [[] for _ in range(self.batch_length)]
    
    def run(self):
        while True:
            dataset = self.queue_for_dataset.get()
            for i in range(self.batch_length):
                self.batch_dataset[i].append(dataset[i])

            if len(self.batch_dataset[0]) == self.batch_size:
                if self.debug:
                    self.timer.tok()
                    print('[Debug] load batch dataset : {}ms'.format(self.timer.tok(ms = True)))
                
                self.main_queue.put(self.batch_dataset)
                self.init_dataset()

                    

