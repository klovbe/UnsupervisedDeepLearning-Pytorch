import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as data
from PIL import Image

import numpy as np
from PIL import Image
#import cv2


import torchvision.transforms as transforms
import pandas as pd

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
#img = transform(img)

class mydataset(data.Dataset):
    def __init__(self, datapath, labelpath, input_dim, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        datadf = pd.read_csv(datapath)
        self.data = datadf.data
        labelsdf = pd.read_csv(labelpath)
        self.labels = labelsdf.data

    def __getitem__(self, index):
        cell, target = self.data[index], self.labels[index]
        # img = Image.fromarray(img)
        if self.transform is not None:
            cell = self.transform(cell)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return cell, target

    def __len__(self):
        return len(self.labels)