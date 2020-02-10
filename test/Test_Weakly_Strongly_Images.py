# Copyright (C) 2020 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import sys
sys.path.insert(1, './')

import cv2
import numpy as np

from core.DataAugment import *
from core.randaugment.augment import *

from utils.Utils import *
from utils.Tensorflow_Utils import *

weakly_augment = WeaklyAugment()
strongly_augment = RandAugment()

labeled_dataset, unlabeled_image_data, test_dataset = get_dataset('./dataset/', n_labels = 4000)

for image, label in labeled_dataset:
    weakly_image = weakly_augment(image.copy())
    strongly_image = strongly_augment(image.copy())
    
    image = cv2.resize(image, (112, 112))
    weakly_image = cv2.resize(weakly_image, (112, 112))
    strongly_image = cv2.resize(strongly_image, (112, 112))

    cv2.imshow('original', image)
    cv2.imshow('weakly_image', weakly_image)
    cv2.imshow('strongly_image', strongly_image)
    cv2.waitKey(0)
