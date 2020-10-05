import os
import subprocess
import zipfile
from glob import glob

from osgeo import gdal, osr, ogr

from ssrgans import utils

def atms_corr_acolite(
    src_dir,
    export_dir,
    src_config_fn,
    aerosol_corr,
    l2w_parameters,
    sub_lim=''):
    """atmospheric correction with ACOLITE

    Args:
        src_dir (str): L1C file directory
        export_dir (str): export directory
        src_config_fn (str): configuration file
        config_fn (str): original configure file (.txt)
        aerosol_corr (str): aerosor correction method
    """
    old_list = glob(os.path.join(export_dir, '*'))
    for item in old_list:
        os.remove(item)
    dst_config_fn = os.path.join(export_dir, os.path.split(src_config_fn)[1])
    fp_dst = open(dst_config_fn, 'w')
    with open(src_config_fn, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            line_user = line
            line_split = line.split('=')
            if len(line_split) == 2:
                if line_split[0] == 'l2w_parameters':
                    if sub_lim:
                        line_add = 'limit=%s\n' % sub_lim
                        line_user = '%s%s=%s\n' % (line_add, line_split[0], l2w_parameters)
                    else:
                        line_user = '%s=%s\n' % (line_split[0], l2w_parameters)
                elif line_split[0] == 'aerosol_correction':
                    line_user = '%s=%s\n' % (line_split[0], aerosol_corr)
                elif line_split[0] == 'inputfile':
                    line_user = '%s=%s\n' % (line_split[0], src_dir)
                elif line_split[0] == 'output':
                    line_user = '%s=%s\n' % (line_split[0], export_dir)
            fp_dst.write(line_user)
    fp_dst.close()
    # run in command line
    print('ACOLITE is running...')
    try:
        process_flag = subprocess.run(
            ['acolite', '--cli', '--settings=%s' % dst_config_fn],
            stdout=subprocess.PIPE
        )
        if process_flag.returncode == 0:
            print('acolite process success!')
        else:
            print('acolite process failed!')
    except:
        print('acolite process failed!')


def main(ifile, opath, vector):
    if os.path.isfile(ifile):
        # extract file
        fz = zipfile.ZipFile(ifile, 'r')
        for file in fz.namelist():
            fz.extract(file, os.path.split(ifile)[0])
        path_name = os.path.split(fz.namelist()[0])[0]
        path_L1C = os.path.join(os.path.split(ifile)[0], path_name)
    else:
        path_L1C = ifile
    if path_L1C[-1] == '\\' or path_L1C[-1] == '/':
        path_L1C = path_L1C[0:-1]
    root_dir = os.path.dirname(os.path.abspath(__file__))
    acolite_dir = os.path.join(opath, '__temp__', 'ACOLITE')
    os.makedirs(acolite_dir, exist_ok=True)
    acolite_config = os.path.join(root_dir, 'resource/acolite_python_settings.txt')
    if 'S2A' in os.path.split(path_L1C)[1]:
        l2_list = 'Rrs_443,Rrs_492,Rrs_560,Rrs_665,Rrs_704,Rrs_740,Rrs_783,Rrs_833,Rrs_865,Rrs_1614'
    elif 'S2B' in os.path.split(path_L1C)[1]:
        l2_list = 'Rrs_442,Rrs_492,Rrs_559,Rrs_665,Rrs_704,Rrs_739,Rrs_780,Rrs_833,Rrs_864,Rrs_1610'
    # sub-region
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
        img_bound = (point_SW[0], point_NE[0], point_SW[1], point_NE[1])
    sub_lim = (
        round(max(v_bound[2], img_bound[2]), 4),
        round(max(v_bound[0], img_bound[0]), 4),
        round(min(v_bound[3], img_bound[3]), 4),
        round(min(v_bound[1], img_bound[1]), 4)
    )
    sub_lim_str = ','.join([str(i) for i in sub_lim])
    atms_corr_acolite(
        src_dir = path_L1C,
        export_dir = acolite_dir,
        src_config_fn = acolite_config,
        aerosol_corr = 'dark_spectrum',
        l2w_parameters = l2_list,
        sub_lim=sub_lim_str
    )
    return acolite_dir
