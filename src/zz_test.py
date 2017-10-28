#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 00:24:01 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import cv2
import numpy as np

from torch.optim import SGD
from torchvision import models
from torch.nn import functional

from misc_functions import preprocess_image, recreate_image


org_im = cv2.imread('../input_images/cat_dog.png', 1)
noise = cv2.imread('../results/fooling_sample_class_340.jpg', 1)

new_im = org_im*0.9 + noise*0.1
cv2.imwrite('new_im.jpg', new_im)
prep_im = preprocess_image(new_im)
pretrained_model = models.alexnet(pretrained=True)
pretrained_model.eval()

out = pretrained_model(prep_im)
