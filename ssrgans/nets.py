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
    def __init__(self, classes):
        super().__init__()
        self.classes = classes
        
        self.classifier = nn.Sequential(
            nn.Linear(5, 2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, 1),
        )

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, 200)
        x = self.classifier(x)
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