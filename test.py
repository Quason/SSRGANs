import sys
import os
import time
import argparse
import uuid
import shutil
import pickle

import numpy as np
import scipy.io as scio
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import linear_model
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


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
    print('input shape:' % input_ds.shape)
    print('target shape:' % target_ds.shape)
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


def reg_models_train(datasets, method='BP', train_percent=0.5, show_fig=False,
                    save_net=''):
    ''' regression models
    '''
    input_ds = None
    target_ds = None
    for item in datasets:
        data_tmp = np.load(item)
        input_tmp = data_tmp['x_all']
        target_tmp = data_tmp['y_all']
        if input_ds is None:
            input_ds = input_tmp
            target_ds = target_tmp
        else:
            input_ds = np.vstack((input_ds, input_tmp))
            target_ds = np.hstack((target_ds, target_tmp))
    # normalize
    scaler_i = StandardScaler()
    scaler_i.fit(input_ds)
    input_ds= scaler_i.transform(input_ds)
    std_target = np.std(target_ds)
    mean_target = np.mean(target_ds)
    target_ds = (target_ds - mean_target) / std_target
    if method == 'BP':
        print('-- BP neural network --')
        net = MLPRegressor(
            hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001,
            batch_size='auto', learning_rate='constant', learning_rate_init=0.01,
            power_t=0.5, max_iter=500, shuffle=True, random_state=None, tol=0.0001,
            verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
            epsilon=1e-08, n_iter_no_change=10, max_fun=15000
        )
    elif method == 'SVM':
        print('-- SVM --')
        net = svm.SVR()
    elif method == 'RF':
        print('-- Random Forest --')
        net = linear_model.Lasso(alpha=0.1)
    elif method == 'LR':
        print('-- Linear Regression --')
        net = linear_model.LinearRegression()
    train_data, test_data, train_label, test_label = train_test_split(
        input_ds, target_ds, random_state=1, train_size=train_percent, test_size=1-train_percent)
    net.fit(train_data, train_label)
    test_predict = net.predict(test_data)
    train_predict = net.predict(train_data)
    # reverse
    train_predict = (train_predict * std_target) + mean_target
    train_label = (train_label * std_target) + mean_target
    test_predict = (test_predict * std_target) + mean_target
    test_label = (test_label * std_target) + mean_target
    print('R_train=%.2f' % (np.corrcoef(train_label, train_predict)[0][1]))
    print('R_test=%.2f' % (np.corrcoef(test_label, test_predict)[0][1]))
    print('MRE_train=%.3f' % (np.mean(np.abs((train_predict-train_label)/train_label))))
    print('MRE_test=%.3f' % (np.mean(np.abs((test_predict-test_label)/test_label))))
    if show_fig:
        # figure
        data_max = max([np.max(train_label), np.max(train_predict), np.max(test_label), np.max(test_predict)]) + 0.01
        data_min = min([np.min(train_label), np.min(train_predict), np.min(test_label), np.min(test_predict)]) - 0.01
        plt.plot(train_label, train_predict, 'bo')
        plt.plot(test_label, test_predict, 'r*')
        plt.plot([data_min, data_max], [data_min, data_max], '--', color='#aaaaaa')
        plt.xlabel('GT')
        plt.ylabel('predict')
        plt.axis('square')
        plt.xlim(data_min, data_max)
        plt.ylim(data_min, data_max)
        plt.show()
    if save_net:
        ssrn_model = {
            'net': net,
            'inputScale': scaler_i,
            'targetMean': mean_target,
            'targetStd': std_target
        }
        with open('./data/%s' % save_net, 'wb') as fn:
            pickle.dump(ssrn_model, fn)
    return net, scaler_i, mean_target, std_target


def reg_models_apply(net, datasets, *, input_scale, target_mean, target_std, show_fig=False):
    data = np.load(datasets)
    input_ds = data['x_all']
    target_ds = data['y_all']
    # normalize
    input_ds= input_scale.transform(input_ds)
    target_ds = (target_ds - target_mean) / target_std
    data_predict = net.predict(input_ds)
    # reverse
    data_predict = (data_predict * target_std) + target_mean
    data_label = (target_ds * target_std) + target_mean
    print('R_apply=%.2f' % (np.corrcoef(data_label, data_predict)[0][1]))
    print('MRE_apply=%.3f' % (np.mean(np.abs((data_predict-data_label)/data_label))))
    if show_fig:
        # figure
        data_max = max([np.max(data_label), np.max(data_predict)]) + 0.01
        data_min = min([np.min(data_label), np.min(data_predict)]) - 0.01
        plt.plot(data_label, data_predict, 'g*')
        plt.plot([data_min, data_max], [data_min, data_max], '--', color='#aaaaaa')
        plt.xlabel('GT')
        plt.ylabel('predict')
        plt.axis('square')
        plt.xlim(data_min, data_max)
        plt.ylim(data_min, data_max)
        plt.show()


def preprocess_s2(ifile, opath, vector, dstfile, kmean_cnt=5):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    acolite_dir = preprocess.main(ifile, opath, vector)
    # Rrs extraction
    if 'S2A' in os.path.split(ifile[0:-1])[1]:
        acolite = models.AcoliteModel(acolite_dir, opath, sensor='S2A')
    else:
        acolite = models.AcoliteModel(acolite_dir, opath, sensor='S2B')
    acolite.cloud_detect(vector=vector)
    acolite.rrs_extractor()
    #  0:443, 1:492, 2:560, 3:665, 4:704, 5:740, 6:783, 7:833, 8:865
    data_all = acolite.kmean_extractor(kmean_cnt)
    x_all = data_all[:, [0,1,2,3,8]]
    y_all = data_all[:, 4]
    target_fn = os.path.join(root_dir, 'data/%s'%(dstfile))
    np.savez(target_fn, x_all=x_all, y_all=y_all)
    shutil.rmtree(acolite_dir)
    return target_fn


if __name__ == '__main__':
    # BP_test()

    # ifile = '/mnt/d/data/L1/with-insitu/dianchi/S2B_MSIL1C_20171108T033929_N0206_R061_T48RTN_20171108T104340.SAFE'
    # opath = '/mnt/d/tmp/pip-test/'
    # vector = '/mnt/d/data/vector/dianchi.geojson'
    # dstfile = 'dataset_dianchi.npz'
    # opath = os.path.join(opath, str(uuid.uuid1()))
    # preprocess_s2(ifile, opath, vector, dstfile, kmean_cnt=3)

    train_fns = [
        './data/dataset_taihu.npz',
        './data/dataset_dianchi.npz',
        './data/dataset_qiandaohu.npz'
    ]
    net, input_scale, target_mean, target_std = reg_models_train(
        train_fns, method='LR', show_fig=False, save_net='ssrn_704.pkl'
    )
    reg_models_apply(
        net, './data/dataset_xingyunhu.npz',
        input_scale=input_scale,
        target_mean=target_mean,
        target_std=target_std,
        show_fig=True
    )
