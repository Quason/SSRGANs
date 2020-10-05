from glob import glob
import os
import shutil
import random

import numpy as np
from osgeo import gdal
from sklearn.cluster import MiniBatchKMeans

from ssrgans import utils

class AcoliteModel():
    def __init__(self, acolite_dir, res_dir, sensor='S2A'):
        self.acolite_dir = acolite_dir
        self.rrs_fns = glob(os.path.join(acolite_dir, '*Rrs*'))
        self.rhos_fns = glob(os.path.join(acolite_dir, '*rhos*'))
        self.rhot_fns = glob(os.path.join(acolite_dir, '*rhot*'))
        self.kd = glob(os.path.join(acolite_dir, '*kd*'))
        self.pre_name = os.path.split((self.rrs_fns)[0])[1][0:-12]
        self.res_dir = res_dir
        if 'S2A' in os.path.split(self.rrs_fns[1])[1]:
            self.sensor ='S2A'
        else:
            self.sensor ='S2B'
        rrs_red = [item for item in self.rrs_fns if '665' in item][0]  
        ds = gdal.Open(rrs_red)
        self.proj_ref = ds.GetProjection()
        self.geo_trans = ds.GetGeoTransform()
        data = ds.GetRasterBand(1).ReadAsArray()
        self.width = np.shape(data)[1]
        self.height = np.shape(data)[0]
        self.water = None

    def cloud_detect(self):
        rhot_red = [item for item in self.rhot_fns if 'rhot_665' in item][0]
        if self.sensor == 'S2A':
            rhos_green = [item for item in self.rhos_fns if 'rhos_560' in item][0]
            rhos_swir1 = [item for item in self.rhos_fns if 'rhos_1614' in item][0]
        else:
            rhos_green = [item for item in self.rhos_fns if 'rhos_559' in item][0]
            rhos_swir1 = [item for item in self.rhos_fns if 'rhos_1610' in item][0]
        ndsi = utils.band_math([rhos_green,rhos_swir1], '(B1-B2)/(B1+B2)')
        cloud_prob1 = utils.band_math([rhot_red], 'B1>0.15')
        cloud = cloud_prob1
        dst_fn = os.path.join(self.res_dir, self.pre_name+'classification.tif')
        scene_class = np.asarray(cloud, np.uint8) * 0
        scene_class[ndsi>-0.01] = 1
        scene_class[cloud] = 2
        utils.raster2tif(scene_class, self.geo_trans, self.proj_ref, dst_fn, type='uint8')
        self.water = (ndsi>-0.01) * (cloud<0.5)

    def rrs_extractor(self):
        for item in self.rrs_fns:
            shutil.copy(item, os.path.join(self.res_dir, os.path.split(item)[1]))

    def chla_mishra(self):
        rrs_red = [item for item in self.rrs_fns if '665' in item][0]
        rrs_red_edge = [item for item in self.rrs_fns if '704' in item][0]
        ndci = utils.band_math([rrs_red_edge, rrs_red], '(B1-B2)/(B1+B2)')
        chla_mishra = 14.039 + 86.115*ndci + 194.325*ndci**2
        chla_mishra[chla_mishra < 0] = -9999
        chla_mishra[self.water < 0.5] = -9999
        dst_fn = os.path.join(self.res_dir, self.pre_name+'chla_mishra.tif')
        utils.raster2tif(chla_mishra, self.geo_trans, self.proj_ref, dst_fn, type='float')
        return dst_fn

    def kmean_extractor(self, classes=5, target_cnt=10000):
        if self.sensor == 'S2A':
            RrsB1 = [item for item in self.rrs_fns if 'Rrs_443' in item][0]
            RrsB2 = [item for item in self.rrs_fns if 'Rrs_492' in item][0]
            RrsB3 = [item for item in self.rrs_fns if 'Rrs_560' in item][0]
            RrsB4 = [item for item in self.rrs_fns if 'Rrs_665' in item][0]
            RrsB5 = [item for item in self.rrs_fns if 'Rrs_704' in item][0]
            RrsB6 = [item for item in self.rrs_fns if 'Rrs_740' in item][0]
            RrsB7 = [item for item in self.rrs_fns if 'Rrs_783' in item][0]
            RrsB8 = [item for item in self.rrs_fns if 'Rrs_833' in item][0]
            RrsB8A = [item for item in self.rrs_fns if 'Rrs_865' in item][0]
        else:
            RrsB1 = [item for item in self.rrs_fns if 'Rrs_442' in item][0]
            RrsB2 = [item for item in self.rrs_fns if 'Rrs_492' in item][0]
            RrsB3 = [item for item in self.rrs_fns if 'Rrs_559' in item][0]
            RrsB4 = [item for item in self.rrs_fns if 'Rrs_665' in item][0]
            RrsB5 = [item for item in self.rrs_fns if 'Rrs_704' in item][0]
            RrsB6 = [item for item in self.rrs_fns if 'Rrs_739' in item][0]
            RrsB7 = [item for item in self.rrs_fns if 'Rrs_780' in item][0]
            RrsB8 = [item for item in self.rrs_fns if 'Rrs_833' in item][0]
            RrsB8A = [item for item in self.rrs_fns if 'Rrs_864' in item][0]
        nband_stack = [RrsB1, RrsB2, RrsB3, RrsB4, RrsB5, RrsB6, RrsB7, RrsB8, RrsB8A]
        data_stack = np.zeros((self.height, self.width, len(nband_stack)))
        x_train = np.zeros((len(self.water[self.water]), len(nband_stack)))
        for i in range(len(nband_stack)):
            data_stack[:,:,i] = utils.band_math([nband_stack[i]], 'B1')
        n = 0
        for i in range(self.height):
            for j in range(self.width):
                if self.water[i, j]:
                    x_train[n, :] = np.squeeze(data_stack[i, j, :])
                    n += 1
        mb_kmeans = MiniBatchKMeans(classes)
        mb_kmeans.fit(x_train)
        y_train = mb_kmeans.predict(x_train)
        kmean_res = np.zeros((self.height, self.width), np.uint8)
        kmean_res[self.water] = y_train + 1
        dst_fn = os.path.join(self.res_dir, self.pre_name+'kmean.tif')
        utils.raster2tif(kmean_res, self.geo_trans, self.proj_ref, dst_fn, type='uint8')
        per_choice = int(target_cnt/classes)
        data_choice = []
        for i in range(classes):
            data_t = x_train[y_train==i, :]
            if np.shape(data_t)[0] * 0.5 > per_choice:
                a = [x for x in range(np.shape(data_t)[0])]
                data_choice_t = data_t[random.sample(a, per_choice), :]
            else:
                choice_cnt = np.shape(data_t)[0] * 0.5
                data_choice_t = data_t[random.sample(a, choice_cnt), :]
            if i==0:
                data_choice = data_choice_t
            else:
                data_choice = np.vstack((data_choice, data_choice_t))
        return(data_choice)
