import os

from osgeo import gdal, ogr, osr
import numpy as np


def vector2mask(raster_fn, vector_fn):
    """make mask layer with vector file (.shp)

    Args:
        raster_fn (str): raster file name
        vector_fn (str): vector file name
    Return:
        ndarray
    """    
    raster = gdal.Open(raster_fn)
    tifData = raster.GetRasterBand(1).ReadAsArray()
    raster_fn_out = vector_fn.replace('.shp', '') + '_mask.tif'
    target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn_out, tifData.shape[1], tifData.shape[0], 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(raster.GetGeoTransform())
    target_ds.SetProjection(raster.GetProjectionRef())
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(255)
    source_ds = ogr.Open(vector_fn)
    source_layer = source_ds.GetLayer()
    gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[0]) # 栅格化函数
    target_ds = None
    raster = None
    return raster_fn_out


def band_math(band_list, expression, scale=1.0):
    """execute band math
    """

    for i in range(len(band_list)):
        raster = gdal.Open(band_list[i])
        band_data = (raster.GetRasterBand(1).ReadAsArray()).astype(float) * scale
        exec('B%d=band_data' % (i+1))
    return eval(expression)


def raster2tif(raster, geo_trans, proj_ref, dst_fn, type='float', mask=True):    
    """save ndarray as GeoTiff
    """
    driver = gdal.GetDriverByName('GTiff')
    if len(raster.shape) == 2:
        nbands = 1
    else:
        nbands = raster.shape[2]
    if type == 'uint8':
        target_ds = driver.Create(
            dst_fn, raster.shape[1], raster.shape[0], nbands, gdal.GDT_Byte)
        mask_value = 0
    elif type == 'int':
        target_ds = driver.Create(
            dst_fn, raster.shape[1], raster.shape[0], nbands, gdal.GDT_Int16)
        mask_value = -9999
    else:
        target_ds = driver.Create(
            dst_fn, raster.shape[1], raster.shape[0], nbands, gdal.GDT_Float32)
        mask_value = -9999
    target_ds.SetGeoTransform(geo_trans)
    target_ds.SetProjection(proj_ref)
    if nbands == 1:
        target_ds.GetRasterBand(1).WriteArray(raster)
        if mask and (mask_value is not None):
            target_ds.GetRasterBand(1).SetNoDataValue(mask_value)
    else:
        for i in range(nbands):
            target_ds.GetRasterBand(i+1).WriteArray(raster[:,:,i])
            if mask and (mask_value is not None):
                target_ds.GetRasterBand(i+1).SetNoDataValue(mask_value)
    target_ds = None


def auto_tone(data, threshold):
    xmin_global = threshold[0]
    xmax_global = threshold[1]
    y = 255 * (data-xmin_global) / (xmax_global-xmin_global)
    y[y < 0] = 0
    y[y > 255] = 255
    y[np.isnan(data)] = 255
    return(y.astype(np.uint8))


def rgb_generator(red, green, blue, res_dir='', scale=1e-4):
    ds = gdal.Open(red)
    red_data = (ds.GetRasterBand(1).ReadAsArray()).astype(float) * scale
    ds = gdal.Open(green)
    green_data = (ds.GetRasterBand(1).ReadAsArray()).astype(float) * scale
    ds = gdal.Open(blue)
    blue_data = (ds.GetRasterBand(1).ReadAsArray()).astype(float) * scale
    red_data = auto_tone(red_data, [0, 0.06])
    green_data = auto_tone(green_data, [0, 0.06])
    blue_data = auto_tone(blue_data, [0, 0.06])
    rgb = np.zeros((np.shape(red_data)[0], np.shape(red_data)[1], 3), np.uint8)
    rgb[:, :, 0] = red_data
    rgb[:, :, 1] = green_data
    rgb[:, :, 2] = blue_data
    if res_dir:
        rgb_fn = os.path.join(res_dir, 'true_color.tif')
    else:
        rgb_fn = os.path.join(os.path.split(red)[0], 'true_color.tif')
    raster2tif(
        rgb,
        ds.GetGeoTransform(),
        ds.GetProjection(),
        rgb_fn,
        type='uint8',
        mask=False
    )
    return rgb_fn


def coord_trans(EPSGs, EPSGt, x, y):
    """coordinate transition

    Args:
        EPSGs (int): source coordinate system
        EPSGt (int): target coordinate system
        x (float): source x index
        y (float): source y index
    """
    sys0 = osr.SpatialReference()
    sys0.ImportFromEPSG(EPSGs) # 投影坐标系
    sys1 = osr.SpatialReference()
    sys1.ImportFromEPSG(EPSGt) # 地理坐标系
    ct = osr.CoordinateTransformation(sys0, sys1)
    coords = ct.TransformPoint(x, y)
    return coords[:2]


def get_epsg(src_fn):
    ds = gdal.Open(src_fn)
    proj_ref = ds.GetProjection()
    srs = osr.SpatialReference(wkt=proj_ref)
    assert srs.IsProjected, "raster has no projection"
    return srs.GetAttrValue("authority", 1)