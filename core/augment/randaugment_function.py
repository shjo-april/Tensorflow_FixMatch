# Copyright (C) 2020 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import random
import numpy as np

import core.augment.randaugment.policies as found_policies
import core.augment.randaugment.augmentation_transforms as transform

class RandAugment:
    def __init__(self):
        self.mean, self.std = transform.get_mean_and_std()
        self.polices = found_policies.randaug_policies()
    
    def __call__(self, image):
        # norm
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        
        # randaugment
        chosen_policy = random.choice(self.polices)
        aug_image = transform.apply_policy(chosen_policy, image)
        aug_image = transform.cutout_numpy(aug_image)
        
        # denorm
        aug_image = (aug_image * self.std) + self.mean
        aug_image = aug_image * 255.

        return aug_image.astype(np.uint8)
