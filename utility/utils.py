# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import cv2
import time
import json
import pickle
import datetime

import numpy as np
import xml.etree.ElementTree as ET

class Colors:
    class BlackBackground:
        # TextColour BlackBackground
        Dark_Gray = "\033[1;30;40m"
        Bright_Red = "\033[1;31;40m"
        Bright_Green = "\033[1;32;40m"
        Yellow = "\033[1;33;40m"
        Bright_Blue = "\033[1;34;40m"
        Bright_Magenta = "\033[1;35;40m"
        Bright_Cyan = "\033[1;36;40m"
        White = "\033[1;37;40m"
    
    class WhiteBackground:
        # WhiteText ColouredBackground
        Red = "\033[0;37;41m"
        Green = "\033[0;37;42m"
        Yellow = "\033[0;37;43m"
        Blue = "\033[0;37;44m"
        Magenta = "\033[0;37;45m"
        Cyan = "\033[0;37;46m"
        White = "\033[0;37;47m"
        Black = "\033[0;37;48m"
    
    RED   = "\033[1;31m"  
    BLUE  = "\033[1;34m"
    CYAN  = "\033[1;36m"
    GREEN = "\033[0;32m"
    RESET = "\033[0;0m"
    BOLD    = "\033[;1m"
    REVERSE = "\033[;7m"
    
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def load_pickle(pickle_path):
    return pickle.load(open(pickle_path, 'rb'))

def dump_pickle(pickle_path, dataset):
    return pickle.dump(dataset, open(pickle_path, 'wb'))

def encode_image(image_data):
    _, image_data = cv2.imencode('.jpg', image_data)
    return image_data

def decode_image(image_data):
    image_data = np.fromstring(image_data, dtype = np.uint8)
    image_data = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    return image_data

def read_xml(xml_path, class_names):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find('size')
    image_width = int(size.find('width').text)
    image_height = int(size.find('height').text)
    
    bboxes = []
    classes = []
    
    for obj in root.findall('object'):
        label = obj.find('name').text
        if not label in class_names:
            continue
            
        bbox = obj.find('bndbox')
        
        bbox_xmin = max(min(int(bbox.find('xmin').text.split('.')[0]), image_width - 1), 0)
        bbox_ymin = max(min(int(bbox.find('ymin').text.split('.')[0]), image_height - 1), 0)
        bbox_xmax = max(min(int(bbox.find('xmax').text.split('.')[0]), image_width - 1), 0)
        bbox_ymax = max(min(int(bbox.find('ymax').text.split('.')[0]), image_height - 1), 0)

        if (bbox_xmax - bbox_xmin) == 0 or (bbox_ymax - bbox_ymin) == 0:
            continue
        
        bboxes.append([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax])
        classes.append(label)

    return np.asarray(bboxes, dtype = np.float32), np.asarray(classes)

def get_today():
    now = time.localtime()
    s = "%04d-%02d-%02d-%02dh%02dm%02ds" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    return s

def log_print(string, log_path = './log.txt'):
    print(string)
    
    if log_path is not None:
        f = open(log_path, 'a+')
        f.write(string + '\n')
        f.close()

def csv_print(data_list, log_path = './log.csv'):
    string = ''
    for data in data_list:
        if type(data) != type(str):
            data = str(data)
        string += (data + ',')
    
    if log_path is not None:
        f = open(log_path, 'a+')
        f.write(string + '\n')
        f.close()

def read_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def write_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent = '\t')

def single_one_hot(label, classes):
    v = np.zeros((classes), dtype = np.float32)
    v[label] = 1.
    return v

def multiple_one_hot(labels, classes):
    v = np.zeros([classes], dtype = np.float32)
    for label in labels:
        v[label] = 1.
    return v