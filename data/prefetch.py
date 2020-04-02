# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import cv2
import copy
import time
import random

import numpy as np
import multiprocessing as mp

from utility.timer import *
from utility.utils import *

class Prefetch_using_queue:
    def __init__(self, class_func, use_cores, max_size):
        self.main_queue = mp.Queue(maxsize = max_size)
        self.instances = [class_func(self.main_queue) for _ in range(use_cores)]
    
    def start(self):
        for instance in self.instances:
            print(instance)
            instance.start()

    def get_size(self):
        return self.main_queue.qsize()

class Prefetch_using_list:
    def __init__(self, class_func, use_cores, max_size):
        self.manager = mp.Manager()

        self.main_list = self.manager.list()
        self.instances = [class_func(self.main_list) for _ in range(use_cores)]
        
    def start(self):
        for instance in self.instances:
            print(instance)
            instance.start()

    def get_size(self):
        return len(self.main_list)