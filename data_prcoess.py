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
    def __init__(self, model_name, input_dim, gene_select, transform=None, target_transform=None, data_path='F:/project/data/'):
        self.transform = transform
        self.target_transform = target_transform
        datapath = data_path + 'h_' + model_name + '.train'
        datadf = pd.read_csv(datapath)
        x = self.extract_features(datadf.values, gene_select)
        # scale to [0,1]
        from sklearn.preprocessing import MinMaxScaler
        features = MinMaxScaler().fit_transform(x)
        self.data = features
        labelpath = data_path + model_name + '_label.csv'
        labelsdf = pd.read_csv(labelpath, header=None, index_col=None)
        self.labels = labelsdf.values.transpose().squeeze()

    def get_n_centroids(self):
        n_clusters = len(np.unique(self.labels))
        return n_clusters

    def extract_features(self, data, gene_select=1000):
        # sheng xu pai lie qu biggest variational gens, then dao xu
        selected = np.std(data, axis=0)
        selected = selected.argsort()[-gene_select:][::-1]
        h_data = data[:, selected]
        return h_data

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

