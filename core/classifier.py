# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import numpy as np
import tensorflow as tf

from core.backbone.wideresnet import *

class Classifier(WideResNet):
    def __init__(self, option):
        super().__init__(option)

    def forward(self, x):
        return super().forward(x)