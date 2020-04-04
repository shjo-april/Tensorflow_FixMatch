# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

from core.augment.functions import *
from core.augment.randaugment_function import *

class DataAugmentation:
    def __init__(self, augment_functions = []):
        self.augment_functions = augment_functions
    
    def __call__(self, image):
        aug_image = image.copy()
        for augment_func in self.augment_functions:
            aug_image = augment_func(aug_image)
        return aug_image

class BaseAugment(DataAugmentation):
    def __init__(self, crop_size):
        augment_functions = [
            Random_HorizontalFlip(),
            Random_ColorJitter(brightness = 0.3, contrast = 0.3, saturation = 0.3, hue = 0.1),
            Random_Crop(crop_size),
        ]
        super().__init__(augment_functions)

class Flip_and_Crop(DataAugmentation):
    def __init__(self, crop_size):
        w, h = crop_size

        augment_functions = [
            Random_HorizontalFlip(),
            Padding(int(w * 0.125)),
            Random_Crop(crop_size),
        ]
        super().__init__(augment_functions)

class RandAugmentation(DataAugmentation):
    def __init__(self):
        augment_functions = [
            RandAugment(),
        ]
        super().__init__(augment_functions)

