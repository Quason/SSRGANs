from glob import glob
import os
import shutil
import random

import numpy as np
from osgeo import gdal
from sklearn.cluster import MiniBatchKMeans
from cv2 import cv2

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

    def cloud_detect0(self):
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

    def cloud_detect(self, vector=None, sola=None, solz=None):
        if self.sensor == 'S2A':
            blue_fn = [item for item in self.rhot_fns if 'rhot_492' in item][0]
            green_fn = [item for item in self.rhot_fns if 'rhot_560' in item][0]
            red_fn = [item for item in self.rhot_fns if 'rhot_665' in item][0]
            nir_fn = [item for item in self.rhot_fns if 'rhot_833' in item][0]
            swir1_fn = [item for item in self.rhot_fns if 'rhot_1614' in item][0]
            swir2_fn = [item for item in self.rhot_fns if 'rhot_2202' in item][0]
            cirrus_fn = [item for item in self.rhot_fns if 'rhot_1373' in item][0]
        else:
            blue_fn = [item for item in self.rhot_fns if 'rhot_492' in item][0]
            green_fn = [item for item in self.rhot_fns if 'rhot_559' in item][0]
            red_fn = [item for item in self.rhot_fns if 'rhot_665' in item][0]
            nir_fn = [item for item in self.rhot_fns if 'rhot_833' in item][0]
            swir1_fn = [item for item in self.rhot_fns if 'rhot_1610' in item][0]
            swir2_fn = [item for item in self.rhot_fns if 'rhot_2186' in item][0]
            cirrus_fn = [item for item in self.rhot_fns if 'rhot_1377' in item][0]
        blue = utils.band_math([blue_fn], 'B1')
        green = utils.band_math([green_fn], 'B1')
        red = utils.band_math([red_fn], 'B1')
        ndsi = utils.band_math([green_fn, swir1_fn], '(B1-B2)/(B1+B2)')
        swir2 = utils.band_math([swir2_fn], 'B1')
        ndvi = utils.band_math([nir_fn, red_fn], '(B1-B2)/(B1+B2)')
        blue_swir = utils.band_math([blue_fn, swir1_fn], 'B1/B2')
        # step 1
        cloud_prob1 = (swir2 > 0.03) * (ndsi<0.8) * (ndvi<0.5) * (red>0.15)
        mean_vis = (blue + green + red) / 3
        cloud_prob2 = (
            np.abs(blue - mean_vis)/mean_vis
            + np.abs(green - mean_vis)/mean_vis
            + np.abs(red - mean_vis)/mean_vis) < 0.7
        cloud_prob3 = (blue - 0.5*red) > 0.08
        cloud_prob4 = utils.band_math([nir_fn,swir1_fn], 'B1/B2>0.75')
        cloud = cloud_prob1 * cloud_prob2 * cloud_prob3 * cloud_prob4
        cloud = cloud.astype(np.uint8)
        cnt_cloud = len(cloud[cloud==1])
        cloud_level = cnt_cloud / np.shape(cloud)[0] / np.shape(cloud)[1]
        print('cloud level:%.3f' % cloud_level)
        # cloud shadow detection
        cloud_large = np.copy(cloud) * 0
        # -- only the cloud over water was saved --
        # labels_struct[0]: count;
        # labels_struct[1]: label matrix;
        # labels_struct[2]: [minY,minX,block_width,block_height,cnt]
        labels_struct = cv2.connectedComponentsWithStats(cloud, connectivity=4)
        img_h, img_w = cloud.shape
        for i in range(1, labels_struct[0]):
            patch = labels_struct[2][i]
            if patch[4] > 2000:
                cloud_large[labels_struct[1]==i] = 1
        PI = 3.1415
        shadow_dire = sola + 180.0
        if shadow_dire > 360.0:
            shadow_dire -= 360.0
        cloud_height = [100+100*i for i in range(100)]
        shadow_mean = []
        for item in cloud_height:
            shadow_dist = item * np.tan(solz/180.0*PI) / 10.0
            w_offset = np.sin(shadow_dire/180.0*PI) * shadow_dist
            h_offset = np.cos(shadow_dire/180.0*PI) * shadow_dist * -1
            affine_m = np.array([[1,0,w_offset], [0,1,h_offset]])
            cloud_shadow = cv2.warpAffine(cloud_large, affine_m, (img_w,img_h))
            cloud_shadow = (cloud_shadow==1) * (cloud_large!=1)
            shadow_mean.append(np.mean(green[cloud_shadow]))
        cloud_hight_opt = cloud_height[shadow_mean.index(min(shadow_mean))]
        shadow_dist_metric = cloud_hight_opt * np.tan(solz/180.0*PI)
        if cloud_hight_opt>200 and cloud_hight_opt<10000 and shadow_dist_metric<5000:
            print('cloud height: %dm' % cloud_hight_opt)
            shadow_dist = cloud_hight_opt * np.tan(solz/180.0*PI) / pixel_size
            w_offset = np.sin(shadow_dire/180.0*PI) * shadow_dist
            h_offset = np.cos(shadow_dire/180.0*PI) * shadow_dist * -1
            affine_m = np.array([[1,0,w_offset], [0,1,h_offset]])
            cloud_shadow = cv2.warpAffine(cloud_large, affine_m, (img_w,img_h))
            cloud_shadow = (cloud_shadow==1) * (cloud_large!=1)

            cloud1 = np.copy(cloud)
            cloud1[cloud_shadow] = 2
            dst_fn = os.path.join(self.res_dir, self.pre_name+'cloud.tif')
            utils.raster2tif(cloud1, self.geo_trans, self.proj_ref, dst_fn, type='uint8')

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cloud_shadow = cv2.morphologyEx(
                cloud_shadow.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            cloud_shadow_key = True
        else:
            cloud_shadow_key = False
        cirrus = utils.band_math([cirrus_fn], 'B1')
        # step 1
        cloud_prob1 = (red - 0.07) / (0.25 - 0.07)
        cloud_prob1[cloud_prob1<0] = 0
        cloud_prob1[cloud_prob1>1] = 1
        # step 2
        cloud_prob2 = (ndsi + 0.1) / (0.2 + 0.1)
        cloud_prob2[cloud_prob2<0] = 0
        cloud_prob2[cloud_prob2>1] = 1
        cloud_prob = cloud_prob1 * cloud_prob2
        # step 3: water
        cloud_prob[blue_swir>2.5] = 0
        cloud_prob = (cloud_prob * 100).astype(np.int)
        self.water = (ndsi > -0.1) * (cloud_prob == 0) * (cirrus<0.012)
        
        # ice mask
        if self.sensor == 'S2A':
            blue_rrs_fn = [item for item in self.rrs_fns if '492' in item][0]
            green_rrs_fn = [item for item in self.rrs_fns if '560' in item][0]
            red_rrs_fn = [item for item in self.rrs_fns if '665' in item][0]
        else:
            blue_rrs_fn = [item for item in self.rrs_fns if '492' in item][0]
            green_rrs_fn = [item for item in self.rrs_fns if '559' in item][0]
            red_rrs_fn = [item for item in self.rrs_fns if '665' in item][0]
        rrs_blue = utils.band_math([blue_rrs_fn], 'B1')
        rrs_green = utils.band_math([green_rrs_fn], 'B1')
        rrs_red = utils.band_math([red_rrs_fn], 'B1')
        mean_vis_rrs = (rrs_blue + rrs_green + rrs_red) / 3
        ice = mean_vis_rrs > 0.04 # 海冰在可见光波段的反射率普遍大于0.15
        self.water *= (ice==0)
        # vector mask
        if vector is not None:
            vector_mask = utils.vector2mask(blue_fn, vector)
            self.water = self.water * (vector_mask==0)
        dst_fn = os.path.join(self.res_dir, self.pre_name+'water.tif')
        utils.raster2tif(self.water, self.geo_trans, self.proj_ref, dst_fn, type='uint8')

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
        return data_choice

    def kmean_extractor_3d(self, classes, target_cnt, target_wave, kernel):
        if kernel % 2 == 0:
            print('[Error] kernel size needs to be odd!!')
            return None
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
        x_train = []
        index_train = []
        for i in range(len(nband_stack)):
            data_stack[:,:,i] = utils.band_math([nband_stack[i]], 'B1')
        n = 0
        line_start = int(kernel/2)
        line_end = self.height - int(kernel/2)
        colm_start = int(kernel/2)
        colm_end = self.width - int(kernel/2)
        water_cp = np.zeros((self.height, self.width), np.uint8)
        for i in range(line_start, line_end):
            for j in range(colm_start, colm_end):
                if self.water[i, j]:
                    i_start = i - int(kernel/2)
                    i_end = i + int(kernel/2) + 1
                    j_start = j - int(kernel/2)
                    j_end = j + int(kernel/2) + 1
                    patch = self.water[i_start:i_end, j_start:j_end]
                    if np.sum(patch) == kernel**2:
                        x_train.append(np.squeeze(data_stack[i, j, :]))
                        index_train.append(i*self.width+j)
                        n += 1
                        water_cp[i, j] = 1
        mb_kmeans = MiniBatchKMeans(classes)
        x_train = np.array(x_train)
        mb_kmeans.fit(x_train)
        y_train = mb_kmeans.predict(x_train)
        y_train += 1 # 使标号从1开始，0为mask区域
        kmean_res = np.zeros((self.height, self.width), np.uint8)
        kmean_res[water_cp==1] = y_train
        dst_fn = os.path.join(self.res_dir, self.pre_name+'kmean.tif')
        utils.raster2tif(kmean_res, self.geo_trans, self.proj_ref, dst_fn, type='uint8')
        per_choice = int(target_cnt/classes) # 每个类别需要选取的样本个数

        index_train = np.array(index_train)
        data_choice = []
        label_choice = []
        rrs_red = [item for item in self.rrs_fns if '665' in item][0]
        rrs_red_edge = [item for item in self.rrs_fns if '704' in item][0]
        ndci = utils.band_math([rrs_red_edge, rrs_red], '(B1-B2)/(B1+B2)')
        chla_mishra = 14.039 + 86.115*ndci + 194.325*ndci**2
        for i in range(classes):
            index_t = index_train[y_train==i+1]
            if len(index_t) * 0.5 > per_choice:
                a = [x for x in range(len(index_t))]
                index_choice_t = index_t[random.sample(a, per_choice)]
            else:
                choice_cnt = np.shape(data_t)[0] * 0.5
                index_choice_t = index_t[random.sample(a, choice_cnt)]
            for item in index_choice_t:
                index_line = int(item / self.width)
                index_colm = item % self.width
                index_line_s = index_line - int(kernel/2)
                index_line_e = index_line + int(kernel/2) + 1
                index_colm_s = index_colm - int(kernel/2)
                index_colm_e = index_colm + int(kernel/2) + 1
                # 0:443, 1:492, 2:560, 3:665, 4:704, 5:740, 6:783, 7:833, 8:865
                data_t = data_stack[
                    index_line_s:index_line_e,
                    index_colm_s:index_colm_e,
                    [0,1,2,3,8]
                ]
                # 调换坐标顺序
                data_t = np.swapaxes(data_t, 1, 2)
                data_t = np.swapaxes(data_t, 0, 1)
                data_choice.append(np.squeeze(data_t))
                if target_wave == 704:
                    label_t = data_stack[index_line, index_colm, 4]
                elif target_wave == 740:
                    label_t = data_stack[index_line, index_colm, 5]
                elif target_wave == 783:
                    label_t = data_stack[index_line, index_colm, 6]
                elif target_wave == 'chla':
                    label_t = chla_mishra[index_line, index_colm]
                label_choice.append(label_t)
        return data_choice, label_choice


class AcoliteModelL8():
    def __init__(self, acolite_dir, res_dir):
        self.acolite_dir = acolite_dir
        self.rrs_fns = glob(os.path.join(acolite_dir, '*Rrs*'))
        self.rhos_fns = glob(os.path.join(acolite_dir, '*rhos*'))
        self.rhot_fns = glob(os.path.join(acolite_dir, '*rhot*'))
        self.kd = glob(os.path.join(acolite_dir, '*kd*'))
        self.pre_name = os.path.split((self.rrs_fns)[0])[1][0:-12]
        self.res_dir = res_dir
        rrs_red = [item for item in self.rrs_fns if '655' in item][0]  
        ds = gdal.Open(rrs_red)
        self.proj_ref = ds.GetProjection()
        self.geo_trans = ds.GetGeoTransform()
        data = ds.GetRasterBand(1).ReadAsArray()
        self.width = np.shape(data)[1]
        self.height = np.shape(data)[0]
        self.water = None

    def cloud_detect(self, vector=None):
        blue_fn = [item for item in self.rhot_fns if 'rhot_483' in item][0]
        green_fn = [item for item in self.rhot_fns if 'rhot_561' in item][0]
        red_fn = [item for item in self.rhot_fns if 'rhot_655' in item][0]
        nir_fn = [item for item in self.rhot_fns if 'rhot_865' in item][0]
        swir1_fn = [item for item in self.rhot_fns if 'rhot_1609' in item][0]
        swir2_fn = [item for item in self.rhot_fns if 'rhot_2201' in item][0]
        cirrus_fn = [item for item in self.rhot_fns if 'rhot_1373' in item][0]
        blue = utils.band_math([blue_fn], 'B1')
        green = utils.band_math([green_fn], 'B1')
        red = utils.band_math([red_fn], 'B1')
        ndsi = utils.band_math([green_fn, swir1_fn], '(B1-B2)/(B1+B2)')
        swir2 = utils.band_math([swir2_fn], 'B1')
        ndvi = utils.band_math([nir_fn, red_fn], '(B1-B2)/(B1+B2)')
        blue_swir = utils.band_math([blue_fn, swir1_fn], 'B1/B2')
        # step 1
        cloud_prob1 = (swir2 > 0.03) * (ndsi<0.8) * (ndvi<0.5) * (red>0.15)
        mean_vis = (blue + green + red) / 3
        cloud_prob2 = (
            np.abs(blue - mean_vis)/mean_vis
            + np.abs(green - mean_vis)/mean_vis
            + np.abs(red - mean_vis)/mean_vis) < 0.7
        cloud_prob3 = (blue - 0.5*red) > 0.08
        cloud_prob4 = utils.band_math([nir_fn,swir1_fn], 'B1/B2>0.75')
        cloud = cloud_prob1 * cloud_prob2 * cloud_prob3 * cloud_prob4
        cloud = cloud.astype(np.uint8)
        cnt_cloud = len(cloud[cloud==1])
        cloud_level = cnt_cloud / np.shape(cloud)[0] / np.shape(cloud)[1]
        print('cloud level:%.3f' % cloud_level)
        # cloud shadow detection
        cirrus = utils.band_math([cirrus_fn], 'B1')
        # step 1
        cloud_prob1 = (red - 0.07) / (0.25 - 0.07)
        cloud_prob1[cloud_prob1<0] = 0
        cloud_prob1[cloud_prob1>1] = 1
        # step 2
        cloud_prob2 = (ndsi + 0.1) / (0.2 + 0.1)
        cloud_prob2[cloud_prob2<0] = 0
        cloud_prob2[cloud_prob2>1] = 1
        cloud_prob = cloud_prob1 * cloud_prob2
        # step 3: water
        cloud_prob[blue_swir>2.5] = 0
        cloud_prob = (cloud_prob * 100).astype(np.int)
        self.water = (ndsi > -0.1) * (cloud_prob == 0) * (cirrus<0.012)
        # ice mask
        blue_rrs_fn = [item for item in self.rrs_fns if '483' in item][0]
        green_rrs_fn = [item for item in self.rrs_fns if '561' in item][0]
        red_rrs_fn = [item for item in self.rrs_fns if '655' in item][0]
        rrs_blue = utils.band_math([blue_rrs_fn], 'B1')
        rrs_green = utils.band_math([green_rrs_fn], 'B1')
        rrs_red = utils.band_math([red_rrs_fn], 'B1')
        mean_vis_rrs = (rrs_blue + rrs_green + rrs_red) / 3
        ice = mean_vis_rrs > 0.04 # 海冰在可见光波段的反射率普遍大于0.15
        self.water *= (ice==0)
        # vector mask
        if vector is not None:
            vector_mask = utils.vector2mask(blue_fn, vector)
            self.water = self.water * (vector_mask==0)
        dst_fn = os.path.join(self.res_dir, self.pre_name+'water.tif')
        utils.raster2tif(self.water, self.geo_trans, self.proj_ref, dst_fn, type='uint8')

    def rrs_extractor(self):
        for item in self.rrs_fns:
            shutil.copy(item, os.path.join(self.res_dir, os.path.split(item)[1]))
