# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import time

class Timer:
    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0
    
    def tik(self):
        self.start_time = time.time()
    
    def tok(self, ms = False):
        self.end_time = time.time()

        if ms:
            duration = int((self.end_time - self.start_time) * 1000)
        else:
            duration = int(self.end_time - self.start_time)

        return duration

