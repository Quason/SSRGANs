import sys
import os
import time
import argparse
import uuid
import shutil
import pickle
import random
from glob import glob

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
import skimage.io
from osgeo import gdal, osr, ogr


from ssrgans import preprocess, models
from ssrgans import nets, utils


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
    plt.ylabel('prediction')
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
        data_t = torch.from_numpy(data_t).type(torch.FloatTensor)
        label_t = self.label[index]
        label_t = torch.Tensor(label_t).type(torch.FloatTensor)
        label_t = label_t.reshape(-1, 1)
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
        t_label = torch.Tensor(label[item[0], item[1]])
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
        t_label = torch.Tensor(label[item[0], item[1]])
        test_data.append(t_data)
        test_label.append(t_label.type(torch.LongTensor))
    testloader = torch.utils.data.DataLoader(
        MyDataset(test_data,test_label), batch_size=batch_size_test, shuffle=True)
    return trainloader, testloader


def myLoader3d(train_datasets, train_perc):
    datasets = np.load(train_datasets)
    if len(datasets['x_all'][0].shape) == 1:
        all_data = list(datasets['x_all'][:,[1,2,3,4]] * 100)
    elif len(datasets['x_all'][0].shape) == 3:
        all_data = list(datasets['x_all'][:,[1,2,3,4],:,:] * 100)
    else:
        print('[Error] unrecoganized dataset')
        sys.exit(0)
    # all_label = list(datasets['y_all'][:] * 100)
    all_label = list(datasets['y_all'][:])
    # shuffle
    all_data_label = []
    for i in range(len(all_data)):
        all_data_label.append([all_data[i], all_label[i]])
    random.shuffle(all_data_label)
    all_data, all_label = [], []
    for item in all_data_label:
        all_data.append(item[0])
        all_label.append(item[1])
    batch_size_train = 5
    batch_size_test = 5
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
        if len(input_tmp[0].shape) == 1:
            if input_ds is None:
                input_ds = input_tmp
                target_ds = target_tmp
            else:
                input_ds = np.vstack((input_ds, input_tmp))
                target_ds = np.hstack((target_ds, target_tmp))
        elif len(input_tmp[0].shape) == 3:
            cindex = int(np.shape(input_tmp[0])[0] / 2)
            patch_extract_stack = None
            for patch in input_tmp:
                patch_extract = np.squeeze(patch[cindex,cindex,:])
                if patch_extract_stack is None:
                    patch_extract_stack = patch_extract
                else:
                    patch_extract_stack = np.vstack((patch_extract_stack, patch_extract))
            if input_ds is None:
                input_ds = patch_extract_stack
                target_ds = target_tmp
            else:
                input_ds = np.vstack((input_ds, patch_extract_stack))
                target_ds = np.hstack((target_ds, target_tmp))
    input_ds = input_ds[:, 1:]
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
    elif method == 'SVR':
        print('-- SVR --')
        net = svm.SVR()
    elif method == 'RFR':
        print('-- Random Forest Regression --')
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
        data_max = max([np.max(train_label), np.max(train_predict), np.max(test_label), np.max(test_predict)]) + 5
        data_min = min([np.min(train_label), np.min(train_predict), np.min(test_label), np.min(test_predict)]) - 5
        # plt.plot(train_label, train_predict, 'bo')
        plt.plot(test_label, test_predict, 'b*')
        plt.plot([data_min, data_max], [data_min, data_max], '--', color='#aaaaaa')
        plt.xlabel('GT')
        plt.ylabel('prediction')
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
    input_ds = input_ds[:, 1:]
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
        plt.ylabel('prediction')
        plt.axis('square')
        plt.xlim(data_min, data_max)
        plt.ylim(data_min, data_max)
        plt.show()


def reg_models_apply_img(model_fn, src_dir, dst_fn):
    print('super spectral resolution...')
    water_fn = glob(os.path.join(src_dir, '*water.tif'))[0]
    rrs_fns = glob(os.path.join(src_dir, '*_Rrs_*'))
    water = skimage.io.imread(water_fn) == 1
    with open(model_fn, 'rb') as fn:
        model_data = pickle.load(fn)
        model = model_data['net']
        input_scale = model_data['inputScale']
        target_mean = model_data['targetMean']
        target_std = model_data['targetStd']
    # 0:443, 1:492, 2:560, 3:665, 4:704, 5:740, 6:783, 7:833, 8:865
    rrs_band = ['Rrs_483', 'Rrs_561', 'Rrs_655', 'Rrs_865']
    nband = len(rrs_band)
    water_mask = np.copy(water)
    band_sum = 0
    for i in range(nband):
        rrs_fn = [item for item in rrs_fns if rrs_band[i] in item]
        rrs_fn = rrs_fn[0]
        ds = gdal.Open(rrs_fn)
        band_data = ds.GetRasterBand(1).ReadAsArray()
        band_sum += band_data
    water_mask[np.isnan(band_sum)] = False
    cnt_label = len(water_mask[water_mask])
    extract_data = np.zeros((nband,cnt_label))
    if cnt_label == 0:
        return 0
    geo_trans, proj_ref = None, None
    for i in range(nband):
        rrs_fn = [item for item in rrs_fns if rrs_band[i] in item]
        rrs_fn = rrs_fn[0]
        ds = gdal.Open(rrs_fn)
        if geo_trans is None:
            geo_trans = ds.GetGeoTransform()
            proj_ref = ds.GetProjection()
        band_data = ds.GetRasterBand(1).ReadAsArray()
        extract_data[i,:] = band_data[water_mask]
    extract_data = extract_data.T
    extract_data = input_scale.transform(extract_data)
    predict = model.predict(extract_data)
    predict = predict * target_std + target_mean
    dst_data = water_mask.astype(float) * 0
    dst_data[water_mask] = predict
    if '704' in model_fn:
        rrs_red_fn = [item for item in rrs_fns if 'Rrs_655' in item][0]
        rrs_red = utils.band_math([rrs_red_fn], 'B1')
        ndci = (dst_data - rrs_red) / (dst_data + rrs_red)
        dst_data = 14.039 + 86.115*ndci + 194.325*ndci**2
    dst_data[water < 0.5] = -9999
    utils.raster2tif(dst_data, geo_trans, proj_ref, dst_fn, type='float')


def dl_models_train(net, datasets_fn, train_perc=0.5):
    print('train...')
    trainloader, testloader = myLoader3d(datasets_fn, train_perc=train_perc)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    epoch_list = []
    for epoch in range(10):
        running_loss_sum = 0
        print('epoch %d...' % (epoch+1))
        running_loss = 0.0
        label_cnt = 0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            label_cnt += len(labels)
            labels = torch.reshape(labels, (-1, 1))
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('epoch %d loss: %.5f' % (epoch+1, running_loss/label_cnt))
        epoch_list.append(running_loss/label_cnt)
        running_loss = 0
    f1 = plt.figure()
    plt.plot(epoch_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # test
    print('test...')
    predict_data = []
    labels_list = []
    with torch.no_grad():
        for i, data in enumerate(testloader):
            inputs, labels = data
            inputs = inputs
            labels = labels
            labels_list += list(torch.flatten(labels).numpy())
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)
            outputs = torch.flatten(net(inputs))
            predict_data += list(outputs.cpu().numpy())
    R2 = np.corrcoef(predict_data, labels_list)[0, 1]
    # figure
    data_max = max([max(labels_list), max(predict_data)])
    data_min = min([min(labels_list), min(predict_data)])
    offset = (data_max - data_min) / 20
    data_min -= offset
    data_max += offset
    plt.figure()
    plt.plot(labels_list, predict_data, 'b*')
    r2 = np.corrcoef(labels_list, predict_data)[0][1]
    labels_list = np.array(labels_list)
    predict_data = np.array(predict_data)
    rmse = np.mean((predict_data - labels_list)**2) ** 0.5
    plt.plot([data_min, data_max], [data_min, data_max], '--', color='#aaa')
    step = (data_max - data_min) * 0.01
    plt.text(data_min+step, data_max*0.9, 'R^2=%.2f'%r2, ha='left')
    plt.text(data_min+step, data_max*0.85, 'RMSE=%.2f'%rmse, ha='left')
    plt.xlabel('GT')
    plt.ylabel('prediction')
    plt.axis('square')
    plt.xlim(data_min, data_max)
    plt.ylim(data_min, data_max)
    plt.show()
    return net


def dl_models_apply_img(model_fn, src_dir, dst_fn, scale=0.01, model_kernel=5):
    print('super spectral resolution...')
    water_fn = glob(os.path.join(src_dir, '*water.tif'))[0]
    rrs_fns = glob(os.path.join(src_dir, '*_Rrs_*'))
    water = skimage.io.imread(water_fn) == 1
    img_width = np.shape(water)[1]
    img_height = np.shape(water)[0]
    net = torch.load(model_fn)
    # 0:443, 1:492, 2:560, 3:665, 4:704, 5:740, 6:783, 7:833, 8:865
    rrs_band = ['Rrs_483', 'Rrs_561', 'Rrs_655', 'Rrs_865']
    L8toS2_scale = [1.0, 1.0, 1.0, 1.0] # scale of L8 to S2
    L8toS2_offset = np.array([0, 0, 0, 0]) / 3.14 # offset of L8 to S2
    nband = len(rrs_band)
    water_mask = np.copy(water)
    band_sum = 0
    for i in range(nband):
        rrs_fn = [item for item in rrs_fns if rrs_band[i] in item]
        rrs_fn = rrs_fn[0]
        ds = gdal.Open(rrs_fn)
        band_data = ds.GetRasterBand(1).ReadAsArray()
        band_sum += band_data
    water_mask[np.isnan(band_sum)] = False
    cnt_label = len(water_mask[water_mask])
    # extract_data = np.zeros((nband,cnt_label))
    data_stack = np.zeros((img_height, img_width, nband))
    if cnt_label == 0:
        return 0
    geo_trans, proj_ref = None, None
    for i in range(nband):
        rrs_fn = [item for item in rrs_fns if rrs_band[i] in item]
        rrs_fn = rrs_fn[0]
        ds = gdal.Open(rrs_fn)
        if geo_trans is None:
            geo_trans = ds.GetGeoTransform()
            proj_ref = ds.GetProjection()
        band_data = ds.GetRasterBand(1).ReadAsArray()
        band_data = band_data * L8toS2_scale[i] + L8toS2_offset[i]
        data_stack[:, :, i] = band_data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    predict = np.zeros((img_height, img_width))
    with torch.no_grad():
        for i in range(img_height):
            if i%100==0:
                print('%.1f%%' % (i/img_height*100))
            for j in range(img_width):
                if water[i, j]:
                    x_s = i - int(model_kernel / 2)
                    x_e = i + int(model_kernel / 2) + 1
                    y_s = j - int(model_kernel / 2)
                    y_e = j + int(model_kernel / 2) + 1
                    patch_water = water[x_s:x_e, y_s:y_e]
                    if np.sum(patch_water) == model_kernel**2:
                        patch = data_stack[x_s:x_e, y_s:y_e, :] / scale
                        patch = np.swapaxes(patch, 1, 2)
                        patch = np.swapaxes(patch, 0, 1)
                        inputs = np.reshape(patch, (1, nband, model_kernel, model_kernel))
                        inputs = torch.from_numpy(inputs)
                        inputs = inputs.type(torch.FloatTensor)
                        inputs = inputs.to(device=device)
                        outputs = net(inputs)
                        predict[i, j] = outputs
    # predict = predict * scale
    predict[water < 0.5] = -9999
    if dst_fn is None:
        fn_483 = [item for item in rrs_fns if 'Rrs_483' in item][0]
        dst_fn = fn_483.replace('Rrs_483', 'Rrs_704')
        dst_fn_chla = fn_483.replace('Rrs_483', 'Chla')
    utils.raster2tif(predict, geo_trans, proj_ref, dst_fn, type='float')
    # # Chla
    # rrs_fn_red = [item for item in rrs_fns if 'Rrs_655' in item][0]
    # red = utils.band_math([rrs_fn_red], 'B1')
    # red = red * 0.9902 + 0.0006/3.14
    # ndci = (predict - red) / (predict + red)
    # chla = 14.039 + 86.115*ndci + 194.325*ndci**2
    # chla[water < 0.5] = -9999
    # utils.raster2tif(chla, geo_trans, proj_ref, dst_fn_chla)


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
    # sub-region
    path_L1C = ifile
    jp2_b2_mid = os.path.join(path_L1C, 'GRANULE')
    jp2_b2_mid = glob(os.path.join(jp2_b2_mid, '*'))[0]
    jp2_b2 = glob(os.path.join(jp2_b2_mid, 'IMG_DATA', '*B02.jp2'))[0]
    ds = gdal.Open(jp2_b2)
    geo_trans = ds.GetGeoTransform()
    proj_ref = ds.GetProjection()
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    ds = None
    vds = ogr.Open(vector)
    lyr = vds.GetLayer()
    v_bound = lyr.GetExtent()
    img_bound = (geo_trans[0], geo_trans[0]+geo_trans[1]*xsize,
        geo_trans[3]+geo_trans[5]*ysize, geo_trans[3])
    src_epsg = int(utils.get_epsg(jp2_b2))
    if src_epsg != 4326:
        point_SW = utils.coord_trans(src_epsg, 4326, img_bound[0], img_bound[2])
        point_NE = utils.coord_trans(src_epsg, 4326, img_bound[1], img_bound[3])
        # China only
        if point_SW[1] > point_SW[0]:
            point_SW = [point_SW[1], point_SW[0]]
        if point_NE[1] > point_NE[0]:
            point_NE = [point_NE[1], point_NE[0]]
        img_bound = (point_SW[0], point_NE[0], point_SW[1], point_NE[1])
    # lat_min, lon_min, lat_max, lon_max
    sub_lim = (
        round(max(v_bound[2], img_bound[2])-0.005, 4),
        round(max(v_bound[0], img_bound[0])-0.005, 4),
        round(min(v_bound[3], img_bound[3])+0.005, 4),
        round(min(v_bound[1], img_bound[1])+0.005, 4)
    )
    center_lonlat = [
        (sub_lim[1] + sub_lim[3]) / 2,
        (sub_lim[0] + sub_lim[2]) / 2
    ]
    # date
    date_str0 = os.path.split(jp2_b2)[1]
    date_str = date_str0.split('_')[1]
    year = int(date_str[0:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    hour = int(date_str[9:11])
    minute = int(date_str[11:13])
    second = int(date_str[13:15])
    date = '%d/%d/%d %d:%d:%d' % (year, month, day, hour, minute, second)
    [solz, sola] = utils.calc_sola_position(center_lonlat[0], center_lonlat[1], date)
    ozone, vapour = 0.6, 1.6
    acolite.cloud_detect(vector=vector, sola=sola, solz=solz)
    acolite.rrs_extractor()
    #  0:443, 1:492, 2:560, 3:665, 4:704, 5:740, 6:783, 7:833, 8:865
    if target_model == 'sklearn':
        data_all = acolite.kmean_extractor(kmean_cnt)
        x_all = data_all[:, [0,1,2,3,8]]
        y_all = data_all[:, 4]
    elif target_model == 'cnn':
        x_all, y_all = acolite.kmean_extractor_3d(
            classes=kmean_cnt, target_cnt=10000, target_wave=twave, kernel=kernel)
    if dstfile is not None:
        np.savez(dstfile, x_all=x_all, y_all=y_all)
    shutil.rmtree(acolite_dir)
    return dstfile


def preprocess_LC08(ifile, opath, vector):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    acolite_dir = preprocess.main(ifile, opath, vector)
    acolite = models.AcoliteModelL8(acolite_dir, opath)
    acolite.cloud_detect(vector=vector)
    acolite.rrs_extractor()
    shutil.rmtree(acolite_dir)


def chla_nir2red(rrs_red, rrs_nir, water_fn, dst_fn=None):
    chla_gons = utils.band_math([rrs_nir, rrs_red], '79.386*(B1/B2)-16.092')
    ds = gdal.Open(rrs_red)
    geo_trans = ds.GetGeoTransform()
    proj_ref = ds.GetProjection()
    water = skimage.io.imread(water_fn) == 0
    chla_gons[chla_gons < 0] = -9999
    chla_gons[water] = -9999
    if dst_fn is None:
        dst_fn = rrs_red[:-11] + 'chla.tif'
    utils.raster2tif(chla_gons, geo_trans, proj_ref, dst_fn, type='float')


if __name__ == '__main__':
    # 训练数据：20200322
    # 测试数据：20171108
    # BP_test()

    # # 数据提取
    # ifile = 'D:/data/L1/with-insitu/dianchi/S2A_MSIL1C_20200322T033531_N0209_R061_T48RTN_20200322T064402.SAFE'
    # opath = 'D:/tmp/pip-test/'
    # vector = 'D:/data/vector/dianchi.geojson'
    # # dstfile = './data/__temp__/dataset_dianchi_5x5.npz'
    # dstfile = None
    # opath = os.path.join(opath, str(uuid.uuid1()))
    # preprocess_s2(ifile, opath, vector, dstfile, kmean_cnt=3, target_model='cnn', twave='chla', kernel=5)

    # # 测试数据提取
    # ifile = 'D:/data/L1/with-insitu/dianchi/LC08_L1TP_129043_20200322_20200326_01_T1'
    # opath = 'D:/tmp/pip-test/dianchi'
    # vector = 'D:/data/vector/dianchi.geojson'
    # opath = os.path.join(opath, str(uuid.uuid1()))
    # preprocess_LC08(ifile, opath, vector)

    # # ML模型训练
    # train_fns = [
    #     './data/dataset_dianchi.npz',
    #     './data/dataset_hulunhu.npz',
    #     './data/dataset_qiandaohu.npz',
    #     './data/dataset_taihu.npz',
    #     './data/dataset_xingyunhu.npz',
    # ]
    # net, input_scale, target_mean, target_std = reg_models_train(
    #     train_fns, method='LR', show_fig=False, save_net='ssrn_704_LR.pkl', train_percent=0.5
    # )
    # # 效果较差：星云湖，千岛湖
    # reg_models_apply(
    #     net, './data/dataset_dianchi_test.npz',
    #     input_scale=input_scale,
    #     target_mean=target_mean,
    #     target_std=target_std,
    #     show_fig=True
    # )

    # # 模型应用: ML
    # model_fn = './data/ssrn_704_LR.pkl'
    # src_dir = r'D:/tmp/pip-test/dianchi/20200322-S2-fake'
    # dst_fn = os.path.join(src_dir, 'chla_LR.tif')
    # reg_models_apply_img(model_fn, src_dir, dst_fn)

    # # CNN模型训练和保存
    # net = nets.WaterNet(4, 1)
    # net = dl_models_train(net, './data/__temp__/dataset_dianchi_5x5.npz', train_perc=0.5)
    # save_net = './data/ssrn_chla_cnn_dianchi.pt'
    # torch.save(net, save_net)

    # # chla_nir2red
    # rrs_red = r'D:\tmp\pip-test\dianchi\20200322-S2-fake\S2A_MSI_2020_03_22_03_35_31_T48RTN_Rrs_655.tif'
    # rrs_nir = r'D:\tmp\pip-test\dianchi\20200322-S2-fake\S2A_MSI_2020_03_22_03_35_31_T48RTN_Rrs_865.tif'
    # water = r'D:\tmp\pip-test\dianchi\20200322-S2-fake\S2A_MSI_2020_03_22_03_35_31_T48RTN_water.tif'
    # chla_nir2red(rrs_red, rrs_nir, water)

    # 模型应用: CNN
    model_fn = './data/ssrn_chla_cnn_dianchi.pt'
    src_dir = r'D:/tmp/pip-test/dianchi/20200322-S2-fake'
    dst_fn = None
    dl_models_apply_img(model_fn, src_dir, dst_fn, scale=0.01, model_kernel=5)
