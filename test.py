import sys
import os
import time
import argparse
import uuid
import shutil

import numpy as np
import scipy.io as scio
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from ssrgans import preprocess, models


def load_data(src_fn):
    data = scio.loadmat(src_fn)
    for key in data.keys():
        if key[:2] != '__':
            return data[key]
    return None


class Baseline(nn.Module):
    ''' baseline network: BP
    '''
    def __init__(self, classes):
        super().__init__()
        self.classes = classes
        
        self.classifier = nn.Sequential(
            nn.Linear(200, 2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, classes),
        )

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, 200)
        x = self.classifier(x)
        return x


def BP_test():
    input_ds = load_data('./data/houseInputs.mat')
    input_ds = input_ds.T
    scaler = StandardScaler()
    scaler.fit(input_ds)
    input_ds= scaler.transform(input_ds)
    target_ds = load_data('./data/houseTargets.mat')
    target_ds = target_ds.flatten()
    bp = MLPRegressor(
        hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001,
        batch_size='auto', learning_rate='constant', learning_rate_init=0.01,
        power_t=0.5, max_iter=500, shuffle=True, random_state=None, tol=0.0001,
        verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
        epsilon=1e-08, n_iter_no_change=10, max_fun=15000
    )
    train_percent = 0.7
    train_data, test_data, train_label, test_label = train_test_split(
        input_ds, target_ds, random_state=1, train_size=train_percent, test_size=1-train_percent)
    bp.fit(train_data, train_label)
    test_predict = bp.predict(test_data)
    train_predict = bp.predict(train_data)
    print('R_train=%.2f' % (np.corrcoef(train_label, train_predict)[0][1]))
    print('R_test=%.2f' % (np.corrcoef(test_label, test_predict)[0][1]))
    # figure
    data_max = max([np.max(train_label), np.max(train_predict), np.max(test_label), np.max(test_predict)])
    data_max = (np.ceil(data_max / 5) + 1) * 5
    plt.plot(train_label, train_predict, 'b*')
    plt.plot(test_label, test_predict, 'r*')
    plt.plot([0, data_max], [0, data_max], '--', color='#aaaaaa')
    plt.xlabel('GT')
    plt.ylabel('predict')
    plt.axis('equal')
    plt.xlim(0, data_max)
    plt.ylim(0, data_max)
    plt.show()


def preprocess_s2():
    # user input args
    parser = argparse.ArgumentParser(
        description='ACOLITE acmospheric correction')
    parser.add_argument('--ifile', type=str, help='source L1C file')
    parser.add_argument('--opath', type=str, help='output path')
    parser.add_argument('--vector', type=str, help='laker vector for mask')
    args = parser.parse_args()
    opath = args.opath
    acolite_dir = preprocess.main(args.ifile, opath, args.vector)
    # acolite_dir = os.path.join(opath, '__temp__/ACOLITE')

    # Rrs extraction
    if 'S2A' in os.path.split(args.ifile[0:-1])[1]:
        acolite = models.AcoliteModel(acolite_dir, opath, sensor='S2A')
    else:
        acolite = models.AcoliteModel(acolite_dir, opath, sensor='S2B')
    acolite.cloud_detect()
    acolite.rrs_extractor()
    print('kmean...')
    acolite.kmean_extractor(5)
    print('kmean done!')
    # shutil.rmtree(acolite_dir)


if __name__ == '__main__':
    preprocess_s2()
