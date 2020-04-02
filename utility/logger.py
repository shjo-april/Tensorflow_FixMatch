# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import numpy as np

class Logger:
    def __init__(self, names, formats):
        self.names = names

        self.log_format = ''
        for _format in formats:
            self.log_format += (_format + ' ')
        
        self.log_dic = {name : [] for name in self.names}
    
    def update(self, values):
        for name, value in zip(self.names, values):
            self.log_dic[name].append(value)
    
    def get_data(self):
        data = [np.mean(self.log_dic[name]) for name in self.names]
        return data
    
    def log(self):
        data = [np.mean(self.log_dic[name]) for name in self.names]
        return self.log_format.format(*data)
    
    def clear(self):        
        self.log_dic = {name : [] for name in self.names}
