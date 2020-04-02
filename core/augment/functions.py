# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

from core.augment.utils import *

class Random_HorizontalFlip:
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            return hflip(x)
        return x

class Random_VerticalFlip:
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            return vflip(x)
        return x

class Random_Crop:
    def __init__(self, crop_size):
        self.crop_w, self.crop_h = crop_size

    def __call__(self, x):
        h, w, c = x.shape
        
        xmin = random.randint(0, w - self.crop_w)
        ymin = random.randint(0, h - self.crop_h)

        return x[ymin : ymin + self.crop_h, xmin : xmin + self.crop_w, :]

class Padding:
    def __init__(self, size = 4):
        self.size = size

    def __call__(self, x):
        return add_padding(x, self.size)

class Random_ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, x):
        transforms = []

        if self.brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            transforms.append(lambda img: adjust_brightness(img, brightness_factor))

        if self.contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            transforms.append(lambda img: adjust_contrast(img, contrast_factor))

        if self.saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            transforms.append(lambda img: adjust_saturation(img, saturation_factor))

        if self.hue > 0:
            hue_factor = random.uniform(-self.hue, self.hue)
            transforms.append(lambda img: adjust_hue(img, hue_factor))

        random.shuffle(transforms)

        for transform in transforms:
            x = transform(x)

        return x

class Center_Crop:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, x):
        return center_crop(x, self.crop_size)

class Resize:
    def __init__(self, image_size):
        self.interpolation_mode = {
            'NEAREST' : cv2.INTER_NEAREST,
            'BILINEAR' : cv2.INTER_LINEAR,
            'BICUBIC' : cv2.INTER_CUBIC,
        }
        
        self.image_size = image_size
        self.interpolation_names = list(self.interpolation_mode.keys())
    
    def __call__(self, image, name = None):
        if image.shape[:2] == self.image_size:
            return image

        if name is None:
            name = random.choice(self.interpolation_names)

        return cv2.resize(image, self.image_size, self.interpolation_mode[name])

