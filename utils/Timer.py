# Copyright (C) 2020 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import time

class Timer:
    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0
    
    def tik(self):
        self.start_time = time.time()
    
    def tok(self):
        self.end_time = time.time()
        self.ms = int((self.end_time - self.start_time) * 1000)
        return self.ms

