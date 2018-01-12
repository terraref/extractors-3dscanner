'''
Created on Nov 17, 2016

@author: Zongyang
'''

import os, sys, terra_common, argparse, shutil, math, colorsys
import numpy as np
from numpy import linspace
from glob import glob
from PIL import Image
from plyfile import PlyData, PlyElement
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
import matplotlib.pyplot as plt
from lmfit import Model
from datetime import date
from terrautils import betydb

PLOT_RANGE_NUM = 54
PLOT_COL_NUM = 32

convt = terra_common.CoordinateConverter()

def options():
    
    parser = argparse.ArgumentParser(description='Angle Distribution Extractor in Roger',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-m", "--mode", help="all day flag, all for all day process, given parent directory as input, one for one day process")
    parser.add_argument("-p", "--ply_dir", help="ply directory")
    parser.add_argument("-j", "--json_dir", help="json directory")
    parser.add_argument("-o", "--out_dir", help="output directory")
    parser.add_argument("-v", "--save_dir", help="integrate result in another output directory")
    parser.add_argument("-y", "--year", help="which year to process")
    parser.add_argument("-d", "--month", help="which month to process")
    parser.add_argument("-s", "--start_date", help="start date")
    parser.add_argument("-e", "--end_date", help="end date")

    args = parser.parse_args()

    return args

def main():
    
    print("start...")
    
    args = options()
    
    
    if args.mode == 'all':
        process_all_scanner_data(args.ply_dir, args.json_dir, args.out_dir)

    if args.mode == 'one':
        full_day_gen_angle_data(args.ply_dir, args.json_dir, args.out_dir)
        
        full_day_summary(args.out_dir)
        
    if args.mode == 'date':
        process_one_month_data(args.ply_dir, args.json_dir, args.out_dir, args.year, args.month, args.start_date, args.end_date, args.save_dir)
    
    
    return

def process_all_scanner_data(ply_parent, json_parent, out_parent):
    
    
    
    return

def process_one_month_data(ply_parent, json_parent, out_parent, str_year, str_month, str_start_date, str_end_date, save_dir_parent):
    
    if not os.path.isdir(save_dir_parent):
        os.makedirs(save_dir_parent)
        
    for day in range(int(str_start_date), int(str_end_date)+1):
        target_date = date(int(str_year), int(str_month), day)
        str_date = target_date.isoformat()
        print(str_date)
        ply_path = os.path.join(ply_parent, str_date)
        json_path = os.path.join(json_parent, str_date)
        out_path = os.path.join(out_parent, str_date)
        save_path = os.path.join(save_dir_parent, str_date)
        if not os.path.isdir(ply_path):
            continue
        if not os.path.isdir(json_path):
            continue
        try:
            
            q_flag = convt.bety_query(str_date)
            #q_flag = convt.bety_query('2017-01-18')
            if not q_flag:
                continue
            
            full_day_gen_angle_data(ply_path, json_path, out_path)
    
            full_day_summary(out_path, save_path)
            
            insert_leafAngle_traits_into_betydb(save_path, save_path, str_date)
        except Exception as ex:
            fail(str_date + str(ex))
    
    return

def full_day_gen_angle_data(ply_path, json_path, out_path):
    
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    
    list_dirs = os.walk(ply_path)
    
    for root, dirs, files in list_dirs:
        for d in dirs:
            print("Start processing "+ d)
            p_path = os.path.join(ply_path, d)
            j_path = os.path.join(json_path, d)
            o_path = os.path.join(out_path, d)
            if not os.path.isdir(p_path):
                continue
            
            if not os.path.isdir(j_path):
                continue
            
            create_angle_data(p_path, j_path, o_path)
            #color_code_depth_img(p_path, j_path, o_path)
    
    
    return

def color_code_depth_img(ply_path, json_path, out_dir):
    
    windowSize = 4
    xyScale = 3
    
    jsonf, plyf, gImf, pImf = find_files(ply_path, json_path)
    if jsonf == [] or plyf == [] or gImf == [] or pImf == []:
        return
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    metadata = terra_common.lower_keys(terra_common.load_json(jsonf))
    yOffset = get_offset_from_metadata(metadata)
    
    gIm = Image.open(gImf)
    
    [gWid, gHei] = gIm.size
    
    # color code gray image
    codeImg = Image.new("RGB", gIm.size, "black")
    
    pix = np.array(gIm).ravel()
    
    gIndex = (np.where(pix>32))
    nonZeroSize = gIndex[0].size
    
    plydata = PlyData.read(plyf)
    
    pointSize = plydata.elements[0].count
    
    if nonZeroSize != pointSize:
        return
    
    gIndexImage = np.zeros(gWid*gHei)
    
    gIndexImage[gIndex[0]] = np.arange(1,pointSize+1)
    
    gIndexImage_ = np.reshape(gIndexImage, (-1, gWid))
    
    angle_data = []
    for i in range(0,32):
        angle_data.append(np.zeros((1,6)))
        
    icount = 0
    # get top angle
    for i in np.arange(0+windowSize*xyScale, gWid-windowSize*xyScale, windowSize*xyScale*2):
        icount = icount+1
        jcount = 0
        for j in np.arange(0+windowSize, gHei-windowSize, windowSize*2):
            jcount = jcount + 1
            plyIndices = gIndexImage_[j-windowSize:j+windowSize+1, i-windowSize*xyScale:i+windowSize*xyScale+1]
            plyIndices = plyIndices.ravel()
            plyIndices_ = np.where(plyIndices>0)
            
            localIndex = plyIndices[plyIndices_[0]].astype('int64')
            if plyIndices_[0].size < 100:
                continue
            localP = plydata.elements[0].data[localIndex-1]
            yCoord = np.mean(localP["y"])
            area_ind = get_area_index(yCoord, yOffset) - 1
            localNormal = calcAreaNormalSurface(localP)
            if localNormal != [] :
                angle_data[area_ind] = np.append(angle_data[area_ind],[localNormal], axis = 0)
    
    hist_data = np.zeros((32, 90))
    pix_height = np.zeros(32)
    disp_window = np.zeros(32)
    min_z  = np.zeros(32)
    ind = 0
    for meta_angle in angle_data:
        if meta_angle.size < 10:
            continue
        
        pix_height[ind] = get_scanned_height(meta_angle)
        leaf_angle = remove_soil_points(meta_angle)
        hist_data[ind] = gen_angle_hist_from_raw(meta_angle)
        disp_window[ind] = np.argmax(hist_data[ind])
        min_z[ind] = np.amin(meta_angle[1:,5])+55
        ind = ind + 1
    
    # color code
    for i in np.arange(0+windowSize*xyScale, gWid-windowSize*xyScale, windowSize*xyScale*2):
        icount = icount+1
        jcount = 0
        for j in np.arange(0+windowSize, gHei-windowSize, windowSize*2):
            jcount = jcount + 1
            plyIndices = gIndexImage_[j-windowSize:j+windowSize+1, i-windowSize*xyScale:i+windowSize*xyScale+1]
            plyIndices = plyIndices.ravel()
            plyIndices_ = np.where(plyIndices>0)
            
            localIndex = plyIndices[plyIndices_[0]].astype('int64')
            if plyIndices_[0].size < 100:
                continue
            localP = plydata.elements[0].data[localIndex-1]
            localNormal = calcAreaNormalSurface(localP)
            
            yCoord = np.mean(localP["y"])
            area_ind = get_area_index(yCoord, yOffset) - 1
            if localNormal == [] :
                continue
            #if localNormal[5] < min_z[area_ind]:
            #    continue
            #if angle_in_range(disp_window[area_ind], localNormal):
            rgb = normals_to_rgb_2(localNormal)
            codeImg.paste(rgb, (i-windowSize*xyScale, j-windowSize, i+windowSize*xyScale+1, j+windowSize+1))
    
            #save_points(localP, '/Users/nijiang/Desktop/normal_plot.png', 4)
    
    img1 = Image.open(gImf)
    img1 = img1.convert('RGB')
    
    img3 = Image.blend(img1, codeImg, 0.5)
    save_png_file = os.path.join(out_dir, os.path.basename(gImf))
    img3.save(save_png_file)
    
    file_ind = 0
    for save_data in angle_data:
        file_ind = file_ind + 1
        out_basename = str(file_ind) + '.npy'
        out_file = os.path.join(out_dir, out_basename)
        np.save(out_file, save_data)
    
    json_basename = os.path.basename(jsonf)
    json_dst = os.path.join(out_dir, json_basename)
    shutil.copyfile(jsonf, json_dst)
    return

def angle_in_range(disp_window, normals):
    
    min_angle = disp_window-3
    min_angle = min_angle if min_angle > 0 else 0
    max_angle = disp_window+3
    max_angle = max_angle if max_angle < 90 else 90
    #state = "fat" if is_fat else "not fat"
    
    r_min = math.cos(math.radians(max_angle))
    r_max = math.cos(math.radians(min_angle))
    
    if normals[2] > r_min and normals[2] < r_max:
        return True
    
    return False

def normals_to_rgb(normals):
    
    h = math.acos(normals[0])/math.pi
    s = math.acos(normals[1])/math.pi
    v = math.acos(normals[2])/math.pi
    rgb = colorsys.hsv_to_rgb(h, s, v)
    r = int(round(rgb[0]*255))
    g = int(round(rgb[1]*255))
    b = int(round(rgb[2]*255))
    
    return (r,g,b)

def normals_to_rgb_2(normals):
    
    val = math.atan2(normals[1], normals[0])
    
    h = (val+math.pi)/(math.pi*2)
    s = math.acos(normals[2])/(math.pi)
    v = 0.7
    rgb = colorsys.hsv_to_rgb(h, s, v)
    r = int(round(rgb[0]*255))
    g = int(round(rgb[1]*255))
    b = int(round(rgb[2]*255))
    
    return (r,g,b)

def convert_to_rgb(minval, maxval, val, colors):
    max_index = len(colors)-1
    v = float(val-minval) / float(maxval-minval) * max_index
    i1, i2 = int(v), min(int(v)+1, max_index)
    (r1, g1, b1), (r2, g2, b2) = colors[i1], colors[i2]
    f = v - i1
    return int(r1 + f*(r2-r1)), int(g1 + f*(g2-g1)), int(b1 + f*(b2-b1))

def create_angle_data(ply_path, json_path, out_dir):
    
    windowSize = 4
    xyScale = 3
    
    jsonf, plyf, gImf, pImf = find_files(ply_path, json_path)
    if jsonf == [] or plyf == [] or gImf == [] or pImf == []:
        return
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
        
    
    json_basename = os.path.basename(jsonf)
    json_dst = os.path.join(out_dir, json_basename)
    if os.path.exists(json_dst):
        return
    
    
    metadata = terra_common.lower_keys(terra_common.load_json(jsonf))
    center_position = get_position(metadata)
    yOffset = get_offset_from_metadata(metadata)
    
    gIm = Image.open(gImf)
    
    [gWid, gHei] = gIm.size
    
    pix = np.array(gIm).ravel()
    
    gIndex = (np.where(pix>33))
    nonZeroSize = gIndex[0].size
    
    plydata = PlyData.read(plyf)
    
    pointSize = plydata.elements[0].count
    
    if nonZeroSize != pointSize:
        return
    
    gIndexImage = np.zeros(gWid*gHei)
    
    gIndexImage[gIndex[0]] = np.arange(1,pointSize+1)
    
    gIndexImage_ = np.reshape(gIndexImage, (-1, gWid))
    
    angle_data = []
    for i in range(0,PLOT_COL_NUM):
        angle_data.append(np.zeros((1,6)))
        
    icount = 0
    for i in np.arange(0+windowSize*xyScale, gWid-windowSize*xyScale, windowSize*xyScale*2):
        icount = icount+1
        jcount = 0
        for j in np.arange(0+windowSize, gHei-windowSize, windowSize*2):
            jcount = jcount + 1
            plyIndices = gIndexImage_[j-windowSize:j+windowSize+1, i-windowSize*xyScale:i+windowSize*xyScale+1]
            plyIndices = plyIndices.ravel()
            plyIndices_ = np.where(plyIndices>0)
            
            localIndex = plyIndices[plyIndices_[0]].astype('int64')
            if plyIndices_[0].size < 100:
                continue
            localP = plydata.elements[0].data[localIndex-1]
            xCoord = np.mean(localP["x"])
            yCoord = np.mean(localP["y"])
            area_ind = get_area_index(xCoord, yCoord, yOffset, center_position[0]) - 1
            if area_ind < 0:
                continue
            localNormal = calcAreaNormalSurface(localP)
            if localNormal != [] :
                angle_data[area_ind] = np.append(angle_data[area_ind],[localNormal], axis = 0)
    
            #save_points(localP, '/Users/nijiang/Desktop/normal_plot.png', 4)
    file_ind = 0
    for save_data in angle_data:
        file_ind = file_ind + 1
        out_basename = str(file_ind) + '.npy'
        out_file = os.path.join(out_dir, out_basename)
        np.save(out_file, save_data)
        
    shutil.copyfile(jsonf, json_dst)
    
    return

def full_day_summary(in_dir, out_dir):
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    list_dirs = os.walk(in_dir)
    angleHist = np.zeros((PLOT_COL_NUM*PLOT_RANGE_NUM, 90))
    relHist = np.zeros((PLOT_COL_NUM*PLOT_RANGE_NUM, 4))
    HeightHist = []
    
    for root, dirs, files in list_dirs:
        for d in dirs:
            full_path = os.path.join(in_dir, d)
            if not os.path.isdir(full_path):
                continue
            
            plotNum, hist_data, pix_height = get_angle_data(full_path)
            
            if len(plotNum) < PLOT_COL_NUM:
                continue
            
            HeightHist.append(pix_height)
            for j in range(0,plotNum.size):
                angleHist[plotNum[j]] = angleHist[plotNum[j]]+hist_data[j]
            
    hist_out = os.path.join(out_dir, 'angleHist.npy')
    np.save(hist_out, angleHist)
    
    hist_out_txt = os.path.join(out_dir, 'angleHist.txt')
    np.savetxt(hist_out_txt, angleHist, delimiter="\t")
    
    # generate beta fit
    for i in range(PLOT_COL_NUM*PLOT_RANGE_NUM):
        relHist[i, :] = calc_beta_distribution_value(angleHist[i])
    chiHistHandle = os.path.join(out_dir, 'beta-distribution.txt')
    np.savetxt(chiHistHandle, relHist, delimiter="\t")
    betaNpyHandle = os.path.join(out_dir, 'beta-distribution.npy')
    np.save(betaNpyHandle, relHist)
    
    return

def get_angle_data(in_dir):
    
    if not os.path.isdir(in_dir):
        fail('Could not find input directory: ' + in_dir)
        
    plotNum = np.zeros(PLOT_COL_NUM)
    hist_data = np.zeros((PLOT_COL_NUM,90))
    pix_height = np.zeros(PLOT_COL_NUM)
    # parse json file
    metafile, angle_files = find_result_files(in_dir)
    if metafile == [] or angle_files == [] :
        return plotNum, hist_data, pix_height
    
    metadata = terra_common.lower_keys(terra_common.load_json(metafile))
    center_position = get_position(metadata)
    
    
    for i in range(0,PLOT_COL_NUM):
        plotNum[i] = field_2_plot(center_position[0], i+1)
        file_path = os.path.join(in_dir, str(i+1)+'.npy')
        if not os.path.exists(file_path):
            continue
        
        meta_angle = np.load(file_path, 'r')
        
        #out_file = os.path.join(in_dir, str(i+1)+'.png')
        #create_angle_visualization(meta_angle, out_file)
        if meta_angle.size < 10:
            continue
        
        #pix_height[i] = get_scanned_height(meta_angle)
        
        leaf_angle = remove_soil_points(meta_angle)
        
        hist_data[i] = gen_angle_hist_from_raw(leaf_angle)
    
    
    return plotNum.astype('int'), hist_data, pix_height

def remove_soil_points(meta_angle):
    
    point_z = meta_angle[1:,5]
    pix_min = np.amin(point_z)
    z_min = pix_min+55
    
    specifiedIndex = np.where(meta_angle[:,5]>z_min)
    target = meta_angle[specifiedIndex]
    
    return target

# get initial soil height for season 2
def get_scanned_height(meta_angle):
    
    point_z = meta_angle[1:,5]
    pix_min = np.amin(point_z)
    pix_max = np.amax(point_z)
    
    return pix_max-pix_min

def create_angle_visualization(dataset, out_file):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    #colors = cm.rainbow(np.linspace(0, 1, 32))
    #ax.scatter(X,Y,Z,color=colors[5], s=2)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    for metadata in dataset:
        if metadata[0] == 0:
            continue
        ax.quiver(metadata[3], metadata[4], metadata[5], metadata[0], metadata[1], metadata[2], length=15)
    
    plt.draw()
    plt.savefig(out_file)
    plt.close()
    
    
    return

def gen_angle_hist_from_raw(raw_data):
    
    plot_angle_hist = np.zeros(90)
    
    zVal = raw_data[:, 2]
    
    for i in range(0, 90):
        r_max = math.cos(math.radians(i))
        r_min = math.cos(math.radians(i+1))
        
        histObj = np.where(np.logical_and(zVal > r_min, zVal < r_max))
        
        plot_angle_hist[i] = histObj[0].size
        
    return plot_angle_hist

def field_2_plot(x_position, y_row):

    xRange = 0
    
    for i in range(PLOT_RANGE_NUM):
        xmin = convt.np_bounds[i][0][0]
        xmax = convt.np_bounds[i][0][1]
        if (x_position > xmin) and (x_position <= xmax):
            xRange = i + 1
            
            plotNum = convt.fieldPartition_to_plotNum_32(xRange, y_row)
            
            return plotNum
    
    return 0

def get_position(metadata):
    try:
        gantry_meta = metadata['lemnatec_measurement_metadata']['gantry_system_variable_metadata']
        gantry_x = gantry_meta["position x [m]"]
        gantry_y = gantry_meta["position y [m]"]
        gantry_z = gantry_meta["position z [m]"]
        
        sensor_fix_meta = metadata['lemnatec_measurement_metadata']['sensor_fixed_metadata']
        
        camera_x = '2.070'#sensor_fix_meta['scanner west location in camera box x [m]']
        # season 1 data don't have this parameter

    except KeyError as err:
        fail('Metadata file missing key: ' + err.args[0])

    try:
        x = float(gantry_x) + float(camera_x)
        y = float(gantry_y)
        z = float(gantry_z)
    except ValueError as err:
        fail('Corrupt positions, ' + err.args[0])
    return (x, y, z)

def find_result_files(in_dir):
    
    json_suffix = os.path.join(in_dir, '*_metadata.json')
    jsons = glob(json_suffix)
    if len(jsons) == 0:
        print in_dir
        fail('Could not find .json file')
        return [], []
    
    npy_suffix = os.path.join(in_dir, '*.npy')
    npys = glob(npy_suffix)
    if len(npys) == 0:
        fail('Could not find .npy files')
        return [], []
    
    return jsons[0], npys
    
    
    return




def get_offset_from_metadata(metadata):
    
    try:
        gantry_meta = metadata['lemnatec_measurement_metadata']['gantry_system_variable_metadata']
        scan_direction = gantry_meta["scanisinpositivedirection"]
        
    except KeyError as err:
        fail('Metadata file missing key: ' + err.args[0])
        
    if scan_direction == 'False':
        yShift = -25.711
    else:
        yShift = -3.60
    
    return yShift

def get_area_index(xCoord, yCoord, yOffset, gantry_x):
    
    xPosition = xCoord/1000 + gantry_x
    yPosition = yCoord/1000 - yOffset
    
    count = 0
    
    xRange = 0
    for i in range(PLOT_RANGE_NUM):
        xmin = convt.np_bounds[i][0][0]
        xmax = convt.np_bounds[i][0][1]
        if (xPosition > xmin) and (xPosition <= xmax):
            xRange = i + 1
        
        for j in range(PLOT_COL_NUM):
            ymin = convt.np_bounds[xRange][j][2]
            ymax = convt.np_bounds[xRange][j][3]
            if (yPosition > ymin) and (yPosition <= ymax):
                count = j + 1
    
    return count

# beta distribution function
def beta_distribution(t, tMean, tVar):
    
    delta0 = tMean*(1-tMean)
    deltaT = tVar
    
    u = (1-tMean)*(delta0/deltaT-1)
    v = tMean*(delta0/deltaT-1)
    
    if u < 0 or v < 0 :
        return -100
    
    try:
        B = math.gamma(u)*math.gamma(v)/math.gamma(u+v)
    except OverflowError:
        B = float('inf')

    #B = math.gamma(u)*math.gamma(v)/math.gamma(u+v)
    
    a = np.power(1-t,u-1)
    b = np.power(t, v-1)
    c = a * b
    ret = c/B
    
    return ret

def beta_mu_nu(tMean, tVar):
    
    delta0 = tMean*(1-tMean)
    deltaT = tVar
    
    u = (1-tMean)*(delta0/deltaT-1)
    v = tMean*(delta0/deltaT-1)
    
    return u, v

def get_mean_and_variance_from_angleHist(angleHist):
    
    tSum = 0
    for i in range(angleHist.size):
        tSum += (i/90.0) * angleHist[i]
    tMean = tSum
    
    tVar = 0
    for i in range(angleHist.size):
        tVar += (i/90.0-tMean)*(i/90.0-tMean)*angleHist[i]
    
    return tMean, tVar

def calc_beta_distribution_value(np_x):
    
    if np.amax(np_x) < 5:
        return -1
    
    if np.isnan(np.min(np_x)):
        return -1
    
    angleHist = np_x / np.sum(np_x)
    tMean, tVar = get_mean_and_variance_from_angleHist(angleHist)
    x = linspace(0.01, 0.99, 90)
    y = angleHist
    delta_y = 90
    y = y * delta_y
    gmod = Model(beta_distribution)
    try:
        result = gmod.fit(y, t=x, tMean=tMean, tVar=tVar)
    except ValueError as err:
        print(err.args[0])
    
    rel = np.zeros(4)
    rel[0], rel[1] = beta_mu_nu(tMean, tVar)
    rel[2], rel[3] = beta_mu_nu(result.best_values['tMean'], result.best_values['tVar'])
    
    return rel

def load_leaf_angle_parameter(in_dir):
    
    betaNpyHandle = os.path.join(in_dir, 'beta-distribution.npy')
    if not os.path.exists(betaNpyHandle):
        return []
    relHist = np.load(betaNpyHandle)
    
    return relHist

def get_traits_table_leafAngle():
    
    fields = ('local_datetime', 'access_level', 'species', 'site',
              'citation_author', 'citation_year', 'citation_title', 'method',
              'leaf_angle_alpha_src', 'leaf_angle_beta_src', 'leaf_angle_alpha_fit', 'leaf_angle_beta_fit', 'entity')
    traits = {'local_datetime' : '',
              'access_level': '2',
              'species': 'Sorghum bicolor',
              'site': [],
              'citation_author': '"Zongyang, Li"',
              'citation_year': '2018',
              'citation_title': 'Maricopa Field Station Data and Metadata',
              'method': 'Scanner 3d ply data to leaf angle distribution',
              'leaf_angle_alpha_src': [],
              'leaf_angle_beta_src': [],
              'leaf_angle_alpha_fit': [],
              'leaf_angle_beta_fit': [],
              'entity': ''}

    return (fields, traits)

def parse_site_from_plotNum_1728(plotNum):

    plot_row = 0
    plot_col = 0
    
    cols = 32
    col = (plotNum-1) % cols + 1
    row = (plotNum-1) / cols + 1
    
    
    if (row % 2) != 0:
        plot_col = col
    else:
        plot_col = cols - col + 1
    
    Range = row
    Column = (plot_col + 1) / 2
    if (plot_col % 2) != 0:
        subplot = 'W'
    else:
        subplot = 'E'
        
    seasonNum = convt.seasonNum
        
    rel = 'MAC Field Scanner Season {} Range {} Column {} {}'.format(str(seasonNum), str(Range), str(Column), subplot)
    
    return rel

def generate_traits_list_height(traits):
    # compose the summary traits
    trait_list = [  traits['local_datetime'],
                    traits['access_level'],
                    traits['species'],
                    traits['site'],
                    traits['citation_author'],
                    traits['citation_year'],
                    traits['citation_title'],
                    traits['method'],
                    traits['leaf_angle_alpha_src'],
                    traits['leaf_angle_beta_src'],
                    traits['leaf_angle_alpha_fit'],
                    traits['leaf_angle_beta_fit'],
                    traits['entity'],
                ]

    return trait_list

def insert_leafAngle_traits_into_betydb(in_dir, out_dir, str_date):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    hist = load_leaf_angle_parameter(in_dir)
    
    out_file = os.path.join(out_dir, str_date+'_betaD.csv')
    csv = open(out_file, 'w')
    
    (fields, traits) = get_traits_table_leafAngle()
        
    csv.write(','.join(map(str, fields)) + '\n')
        
    for j in range(0, PLOT_COL_NUM*PLOT_RANGE_NUM):
        targetHist = hist[j,:]
        plotNum = j+1
        if (targetHist.max() == -1):
            continue
        else: 
            str_time = str_date+'T12:00:00'
            traits['local_datetime'] = str_time
            traits['leaf_angle_alpha_src'] = targetHist[0]
            traits['leaf_angle_beta_src'] = targetHist[1]
            traits['leaf_angle_alpha_fit'] = targetHist[2]
            traits['leaf_angle_beta_fit'] = targetHist[3]
            traits['site'] = parse_site_from_plotNum_1728(plotNum)
            trait_list = generate_traits_list_height(traits)
            csv.write(','.join(map(str, trait_list)) + '\n')
    
    
    csv.close()
    #submitToBety(out_file)
    betydb.submit_traits(out_file, filetype='csv', betykey=betydb.get_bety_key(), betyurl=betydb.get_bety_url())
    
    
    return

def draw_beta_distribution(in_dir, out_dir):
    
    filePath = os.path.join(in_dir, 'angleHist.npy')
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    oneDayHist = np.load(filePath, 'r')
    
    for i in range(oneDayHist.size-1):
        if np.amax(oneDayHist[i]) < 5:
            continue
        
        if np.isnan(np.min(oneDayHist[i])):
            continue
    
        angleHist = oneDayHist[i] / np.sum(oneDayHist[i])
        
        tMean, tVar = get_mean_and_variance_from_angleHist(angleHist)
        
        x = linspace(0.01, 0.99, 90)
        y = angleHist
        delta_y = 90
        y = y * delta_y
        
        gmod = Model(beta_distribution)
        try:
            result = gmod.fit(y, t=x, tMean=tMean, tVar=tVar)
        except ValueError as err:
            print(err.args[0])
        
        print(result.fit_report())
        
        plt.plot(x,y, 'bo')
        plt.plot(x, result.init_fit, 'k--')
        plt.plot(x, result.best_fit, 'r-')
        plt.title('plot num: %d' % (i+1))
        textline = 'reduced chi-square: %f\ndotted line: initial fit by beta distribution\nred line: best fitted\nblue plot: LAD from laser scanner'%(result.redchi)
        plt.annotate(textline, xy=(1, 1), xycoords='axes fraction',horizontalalignment='right', verticalalignment='top')
        out_file = os.path.join(out_dir, str(i)+'.png')
        plt.savefig(out_file)
        plt.close()
    
    
    return

def calcAreaNormalSurface(Q):
    
    nd3points = np.zeros((Q.size, 3))
    centerPoint = np.zeros(3)
    centerPoint[0] = np.mean(Q["x"])
    centerPoint[1] = np.mean(Q["y"])
    centerPoint[2] = np.mean(Q["z"])
    
    nd3points[:,0] = Q["x"] - np.mean(Q["x"])
    nd3points[:,1] = Q["y"] - np.mean(Q["y"])
    nd3points[:,2] = Q["z"] - np.mean(Q["z"])
    
    U, s, V = np.linalg.svd(nd3points, full_matrices=False)
    
    normals = V[2,:]
    normals = np.sign(normals[2])*normals
    
    s1 = s[0]/s[1]
    s2 = s[1]/s[2]
    #show_points_and_normals(nd3points, normals, centerPoint)
    if s1 < s2:
        reval = np.append(normals, centerPoint)
        return reval
    else:
        return []
    
    return normals

def show_points_and_normals(ply_data, normals, centerPoint):
    
    X = ply_data[:,0]
    Y = ply_data[:,1]
    Z = ply_data[:,2]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    colors = cm.rainbow(np.linspace(0, 1, 32))
    ax.scatter(X,Y,Z,color=colors[5], s=2)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    ax.quiver(0, 0, 0, normals[0], normals[1], normals[2], length=2)
    
    plt.draw()
    #plt.savefig(out_file)
    plt.close()
    
    return

def find_files(ply_path, json_path):
    json_suffix = os.path.join(json_path, '*_metadata.json')
    jsons = glob(json_suffix)
    if len(jsons) == 0:
        print json_path
        fail('Could not find .json file')
        return [], [], [], []
    
    ply_suffix = os.path.join(ply_path, '*-west_0.ply')
    plys = glob(ply_suffix)
    if len(plys) == 0:
        print ply_path
        fail('Could not find west ply file')
        return [], [], [], []
    
    gIm_suffix = os.path.join(json_path, '*-west_0_g.png')
    gIms = glob(gIm_suffix)
    if len(gIms) == 0:
        fail('Could not find -west_0_g.png file')
        return [], [], [], []
    
    pIm_suffix = os.path.join(json_path, '*-west_0_p.png')
    pIms = glob(pIm_suffix)
    if len(pIms) == 0:
        fail('Could not find -west_0_p.png file')
        return [], [], [], []
    
    return jsons[0], plys[0], gIms[0], pIms[0]

def fail(reason):
    print >> sys.stderr, reason


if __name__ == "__main__":

    main()