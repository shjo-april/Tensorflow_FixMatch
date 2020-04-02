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

class Loader(mp.Process):
    def __init__(self, main_queue, option):
        super().__init__()
        self.daemon = True
        
        self.main_queue = main_queue
        
        self.dataset = option['dataset']
        
        self.number_of_loading_dataset = option['number_of_loading_dataset']
        if len(self.dataset) <= self.number_of_loading_dataset:
            self.number_of_loading_dataset = -1

        self.bShuffle = option['bShuffle']

        self.debug = option['debug']
        self.timer = Timer()
    
    def run(self):
        while True:
            if self.debug:
                self.timer.tik()
                # print('[Debug] load pickle start')
            
            np.random.shuffle(self.dataset)
            if self.number_of_loading_dataset > 0:
                dataset = self.dataset[:self.number_of_loading_dataset]
            else:
                dataset = self.dataset.copy()
            
            if self.debug:
                print('[Debug] load pickle end = {}sec'.format(self.timer.tok()))
                self.timer.tik()
            
            if self.bShuffle:
                np.random.shuffle(dataset)
            
            self.main_queue.put(dataset)
