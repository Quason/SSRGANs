import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as scio
import random
from torch.utils.data import Dataset, DataLoader


class Baseline(nn.Module):
    ''' baseline network: BP
    '''
    def __init__(self, in_channels, classes):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(in_channels, 1024), # hidden layer
            nn.ReLU(),
            nn.Linear(1024, classes), # output layer
        )

    def forward(self, x):
        x = self.regressor(x)
        return x


class WaterNet(nn.Module):
    def __init__(self, in_channels, classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=1, padding=1), # 4*11*11 -> 4*11*11
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # 64*5*5
            nn.Conv2d(64, 128, 3, stride=1, padding=1), # 128*5*5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # 128*2*2
        )
        self.regressor = nn.Sequential(
            nn.Linear(2*2*128, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x


class NetQin(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.conv2d_1_3x3 = nn.Conv2d(1, 128, (3, 3), stride=1, padding=1)

        self.conv2d_2_1x1 = nn.Conv3d(128, 256, (1, 1), stride=1, padding=0)
        self.conv2d_2_3x3 = nn.Conv3d(128, 256, (3, 3), stride=1, padding=1)
        self.conv2d_2_5x5 = nn.Conv3d(128, 256, (5, 5), stride=1, padding=2)

        self.conv2d_3_1x1 = nn.Conv3d(128, 256, (1, 1), stride=1, padding=0)
        self.conv2d_3_3x3 = nn.Conv3d(128, 256, (3, 3), stride=1, padding=1)
        self.conv2d_3_5x5 = nn.Conv3d(128, 256, (5, 5), stride=1, padding=2)

        self.conv2d1 = nn.Conv2d(256, 128, 1)
        self.conv2d2 = nn.Conv2d(128, 128, 1)
        self.conv2d3 = nn.Conv2d(128, 128, 1)
        self.conv2d4 = nn.Conv2d(128, 128, 1)
        self.conv2d5 = nn.Conv2d(128, 128, 1)
        self.conv2d6 = nn.Conv2d(128, 128, 1)
        self.conv2d7 = nn.Conv2d(128, 128, 1)
        self.conv2d8 = nn.Conv2d(128, n_classes, 1)
        self.lrn1 = nn.LocalResponseNorm(256)
        self.lrn2 = nn.LocalResponseNorm(128)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x_3x3 = self.conv3d_3x3(x)
        x_1x1 = self.conv3d_1x1(x)
        x = torch.cat([x_3x3, x_1x1], dim=1)
        x = torch.squeeze(x, dim=2)
        x = F.relu(self.lrn1(x))
        x = self.conv2d1(x)
        x = F.relu(self.lrn2(x))
        x_res = F.relu(self.conv2d2(x))
        x_res = self.conv2d3(x_res)
        x = F.relu(x + x_res)
        x_res = F.relu(self.conv2d4(x))
        x_res = self.conv2d5(x_res)
        x = F.relu(x + x_res)
        x = F.relu(self.conv2d6(x))
        x = self.dropout(x)
        x = F.relu(self.conv2d7(x))
        x = self.dropout(x)
        x = self.conv2d8(x)
        x = torch.sum(torch.sum(x, dim=2), dim=2)
        return x