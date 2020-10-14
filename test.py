import sys
import os
import time
import argparse
import uuid
import shutil
import pickle
import random

import numpy as np
import scipy.io as scio
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn import linear_model
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


from ssrgans import preprocess, models
from ssrgans import nets


def load_data(src_fn):
    data = scio.loadmat(src_fn)
    for key in data.keys():
        if key[:2] != '__':
            return data[key]
    return None


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


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_t = self.data[index]
        data_t = np.reshape(data_t, (1, -1))
        data_t = torch.from_numpy(data_t).type(torch.FloatTensor)
        label_t = self.label[index]
        label_t = torch.tensor(label_t).type(torch.FloatTensor)
        return data_t, label_t


def myLoader1d(dataset, label, train_perc):
    batch_size_train = 4
    batch_size_test = 4
    dsize = dataset.shape
    valid_index = []
    for i in range(dsize[1]):
        for j in range(dsize[2]):
            valid_index.append([i,j])
    random.shuffle(valid_index)
    # train dataset
    train_size = int(len(valid_index) * train_perc)
    train_index = valid_index[0:train_size]
    train_data = []
    train_label = []
    for item in train_index:
        t_data = dataset[:, item[0], item[1]]
        t_data = t_data.reshape(1, dsize[0])
        t_data = torch.from_numpy(t_data)
        t_data = t_data.type(torch.FloatTensor)
        t_label = torch.tensor(label[item[0], item[1]])
        train_data.append(t_data)
        train_label.append(t_label.type(torch.LongTensor))
    trainloader = torch.utils.data.DataLoader(
        MyDataset(train_data,train_label), batch_size=batch_size_train, shuffle=True)
    # test dataset
    test_index = valid_index[train_size:]
    test_data = []
    test_label = []
    for item in test_index:
        t_data = dataset[:, item[0], item[1]]
        t_data = t_data.reshape(1, dsize[0])
        t_data = torch.from_numpy(t_data)
        t_data = t_data.type(torch.FloatTensor)
        t_label = torch.tensor(label[item[0], item[1]])
        test_data.append(t_data)
        test_label.append(t_label.type(torch.LongTensor))
    testloader = torch.utils.data.DataLoader(
        MyDataset(test_data,test_label), batch_size=batch_size_test, shuffle=True)
    return trainloader, testloader


def myLoader3d(train_datasets, train_perc=0.5):
    datasets = np.load(train_datasets)
    all_data = list(datasets['x_all'])
    all_label = list(datasets['y_all'])
    batch_size_train = len(all_label)
    batch_size_test = 10
    random.shuffle(all_data)
    random.shuffle(all_label)
    train_size = int(len(all_data) * train_perc)
    # train dataset
    train_data = all_data[0:train_size]
    train_label = all_label[0:train_size]
    trainloader = torch.utils.data.DataLoader(
        MyDataset(train_data,train_label), batch_size=batch_size_train, shuffle=True)
    # test dataset
    test_data = all_data[train_size:]
    test_label = all_label[train_size:]
    testloader = torch.utils.data.DataLoader(
        MyDataset(test_data,test_label), batch_size=batch_size_train, shuffle=True)
    return trainloader, testloader


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
        net = RandomForestRegressor(max_depth=10, random_state=0)
    elif method == 'LR':
        print('-- Linear Regression --')
        net = linear_model.LinearRegression()
    elif method == 'LASSO':
        print('-- LASSO --')
        net = linear_model.Lasso(alpha=0.01)
    else:
        print('[Error] unmatched method!')
        sys.exit(0)
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


def dl_models_train(net, datasets_fn):
    print('train...')
    rrs_max = 0.07
    rrs_min = 0
    trainloader, testloader = myLoader3d(datasets_fn, train_perc=0.3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(500):
        running_loss_sum = 0
        print('epoch %d...' % (epoch+1))
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            print(inputs)
            labels = torch.reshape(labels, (-1, 1))
            inputs = (inputs - rrs_min) / (rrs_max - rrs_min)
            labels = (labels - rrs_min) / (rrs_max - rrs_min)
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            # print(torch.sum(torch.abs(torch.flatten(outputs) - labels)))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # if (i+1) % 50 == 0:
            if True:
                print('epoch %d loss: %.3f' % (epoch+1, running_loss))
                running_loss = 0
    # test
    print('test...')
    predict_data = []
    labels_list = []
    with torch.no_grad():
        for i, data in enumerate(testloader):
            inputs, labels = data
            inputs = (inputs - rrs_min) / (rrs_max - rrs_min)
            labels = (labels - rrs_min) / (rrs_max - rrs_min)
            labels_list += list(labels.numpy())
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)
            outputs = torch.flatten(net(inputs))
            predict_data += list(outputs.cpu().numpy())
    R2 = np.corrcoef(predict_data, labels_list)[0, 1]
    # figure
    data_max = max([max(labels_list), max(predict_data)])
    data_min = max([min(labels_list), min(predict_data)])
    plt.plot(labels_list, predict_data, 'b*')
    plt.plot([data_min, data_max], [data_min, data_max], '--', color='#aaaaaa')
    plt.xlabel('GT')
    plt.ylabel('predict')
    plt.axis('square')
    plt.xlim(data_min, data_max)
    plt.ylim(data_min, data_max)
    plt.show()
    return net


def dl_models_apply(net, datasets_fn):
    print('apply ...')
    trainloader, testloader = myLoader3d(datasets_fn, train_perc=0.3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs = inputs.to(device=device)
        labels = labels.to(device=device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(torch.flatten(outputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i+1) % 50 == 0:
            print('epoch %d loss: %.3f' % (epoch+1, running_loss/10))
            running_loss = 0

    return net


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


def preprocess_s2(
    ifile, opath, vector, dstfile, *,
    kmean_cnt=5, target_model='sklearn', twave=704, kernel=1):
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
    if target_model == 'sklearn':
        data_all = acolite.kmean_extractor(kmean_cnt)
        x_all = data_all[:, [0,1,2,3,8]]
        y_all = data_all[:, 4]
    elif target_model == 'cnn':
        x_all, y_all = acolite.kmean_extractor_3d(
            classes=kmean_cnt, target_cnt=10000, target_wave=twave, kernel=kernel)
    np.savez(dstfile, x_all=x_all, y_all=y_all)
    shutil.rmtree(acolite_dir)
    return dstfile


if __name__ == '__main__':
    # BP_test()

    # ifile = '/mnt/d/data/L1/with-insitu/xingyunLake/S2A_MSIL1C_20181118T034021_N0207_R061_T48QTM_20181118T072005.SAFE'
    # opath = '/mnt/d/tmp/pip-test/'
    # vector = '/mnt/d/data/vector/xingyunLake.geojson'
    # dstfile = './data/__temp__/dataset_xingyunhu_1x1.npz'
    # opath = os.path.join(opath, str(uuid.uuid1()))
    # preprocess_s2(ifile, opath, vector, dstfile, kmean_cnt=3, target_model='cnn', twave=704, kernel=1)

    # train_fns = [
    #     './data/dataset_taihu.npz',
    #     './data/dataset_dianchi.npz',
    #     './data/dataset_qiandaohu.npz',
    #     './data/dataset_hulunhu.npz',
    #     './data/dataset_xingyunhu.npz',
    # ]
    # net, input_scale, target_mean, target_std = reg_models_train(
    #     train_fns, method='BP', show_fig=False, save_net='ssrn_704_BP.pkl'
    # )
    # # 效果较差：星云湖，千岛湖
    # reg_models_apply(
    #     net, './data/dataset_dianchi_test.npz',
    #     input_scale=input_scale,
    #     target_mean=target_mean,
    #     target_std=target_std,
    #     show_fig=True
    # )

    net = nets.Baseline(5, 1)
    dl_models_train(net, './data/__temp__/dataset_xingyunhu_1x1.npz')