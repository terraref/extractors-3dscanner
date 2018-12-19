'''
Created on Oct 18, 2016

@author: Zongyang
'''
import os, sys, terra_common, argparse, math, colorsys
from PIL import Image, ImageOps
from glob import glob
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import time

convt = terra_common.CoordinateConverter()

def options():
    
    parser = argparse.ArgumentParser(description='Height Data Visualization in Roger',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    #parser.add_argument("-m", "--mode", help="all day flag, all for all day process, given parent directory as input, one for one day process")
    parser.add_argument("-i", "--in_dir", help="png directory")
    parser.add_argument("-o", "--out_dir", help="output directory")
    parser.add_argument("-t", "--txt_file", help="folder list")

    args = parser.parse_args()

    return args

def main():
    
    args = options()
    
    #plot_sub_points('/Users/Desktop/Scanner3D/8/plys', '/Users/Desktop/Scanner3D/8/pngs')
    
    get_time_lapse_points(args.txt_file, args.out_dir, 8)
    
    #list_target_scans(args.in_dir, args.out_dir, 152.7)
    
    #count_scan_times('/Users/Desktop/Scanner3D/season2scan')

    #scan_3d_data_for_x_position(args.in_dir, args.out_dir)
    
    
    #one_day_process(args.in_dir, args.out_dir)
    
    return

def plot_sub_points(in_dir, out_dir):
    
    list_files = os.walk(in_dir)
    for root, dirs, files in list_files:
        for f in files:
            if not f.endswith('.ply'):
                continue
            
            str_date = f[:-4]
            out_file = os.path.join(out_dir, str_date+'.png')
            ply_data = PlyData.read(os.path.join(in_dir, f))
            save_points(ply_data.elements[0].data, out_file, 8)
    
    
    
    
    return

def get_time_lapse_points(json_list, out_dir, colsID):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    xPos = os.path.basename(json_list)
    xPos = xPos[:5]
    
    out_path = os.path.join(out_dir, xPos)
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
        
    out_path = os.path.join(out_path, str(colsID))
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
        os.mkdir(os.path.join(out_path, 'plys'))
        os.mkdir(os.path.join(out_path, 'pngs'))
        os.mkdir(os.path.join(out_path, 'angleVis'))
    
    txt_handle = open(json_list, 'r')
    
    while 1:
        txt_line = txt_handle.readline()
        if not txt_line:
            break
        
        txt_line = txt_line[:-1]
        json_file, png_file = find_files(txt_line)
        if json_file == [] or png_file == []:
            continue
        
        ply_dir = txt_line.replace('raw_data', 'Level_1')
        ply_w_file = find_ply_file(ply_dir)
        
        parentDir = os.path.abspath(os.path.join(txt_line, os.pardir))
        str_date = os.path.basename(parentDir)
        ltime = time.strptime(str_date, '%Y-%m-%d')
        if ltime.tm_mon < 8 or ltime.tm_mon > 11:
            continue
        
        save_target_points(ply_w_file, json_file, png_file, out_path, colsID, str_date)
    
    return

def save_target_points(ply_w_file, json_file, png_file, out_path, colsID, str_date):
    
    metadata = terra_common.lower_keys(terra_common.load_json(json_file))
    x_position, scan_direction = parse_metadata(metadata)
    plywest = PlyData.read(ply_w_file)
    
    yRange = 32
    yShift = -3
    zOffset = 10
    zRange = [-2000, 2000]
    scaleParam = 1000
    hist = np.zeros((yRange, (zRange[1]-zRange[0])/zOffset))
    heightest = np.zeros((yRange, 1))
    data = plywest.elements[0].data
    
    if data.size == 0:
        return hist, heightest
    
    if scan_direction == 'False':
        yShift = -25.1
    
    ymin = (terra_common._y_row_s2[colsID][0]+yShift) * scaleParam
    ymax = (terra_common._y_row_s2[colsID][1]+yShift) * scaleParam
    specifiedIndex = np.where((data["y"]>ymin) & (data["y"]<ymax))
    target = data[specifiedIndex]
    '''    
    out_png_file = os.path.join(out_path, 'pngs', str_date+'.png')
    save_points(target, out_png_file, colsID)
    out_ply_file = os.path.join(out_path, 'plys', str_date+'.ply')
    save_sub_ply(target, plywest, out_ply_file)
    '''
    y_upper = np.amax(data["y"])
    y_lower = np.amin(data["y"])
    
    min_y = (ymin - y_lower)/(y_upper - y_lower)
    max_y = (ymax - y_lower)/(y_upper - y_lower)
    
    out_angle_file = os.path.join(out_path, 'angleVis', str_date+'.png')
    creat_angle_visualization(ply_w_file, png_file, min_y, max_y, out_angle_file)
    
    return

def color_code_top_angle_pixels(ply_w_file, json_file, png_file, out_path, colsID, str_date):
    
    metadata = terra_common.lower_keys(terra_common.load_json(json_file))
    x_position, scan_direction = parse_metadata(metadata)
    plywest = PlyData.read(ply_w_file)
    
    yRange = 32
    yShift = -3
    zOffset = 10
    zRange = [-2000, 2000]
    scaleParam = 1000
    hist = np.zeros((yRange, (zRange[1]-zRange[0])/zOffset))
    heightest = np.zeros((yRange, 1))
    data = plywest.elements[0].data
    
    if data.size == 0:
        return hist, heightest
    
    if scan_direction == 'False':
        yShift = -25.1
    
    ymin = (terra_common._y_row_s2[colsID][0]+yShift) * scaleParam
    ymax = (terra_common._y_row_s2[colsID][1]+yShift) * scaleParam
    specifiedIndex = np.where((data["y"]>ymin) & (data["y"]<ymax))
    target = data[specifiedIndex]
    '''    
    out_png_file = os.path.join(out_path, 'pngs', str_date+'.png')
    save_points(target, out_png_file, colsID)
    out_ply_file = os.path.join(out_path, 'plys', str_date+'.ply')
    save_sub_ply(target, plywest, out_ply_file)
    '''
    y_upper = np.amax(data["y"])
    y_lower = np.amin(data["y"])
    
    min_y = (ymin - y_lower)/(y_upper - y_lower)
    max_y = (ymax - y_lower)/(y_upper - y_lower)
    
    out_angle_file = os.path.join(out_path, 'angleVis', str_date+'.png')
    create_top_angle_vis(ply_w_file, png_file, min_y, max_y, out_angle_file)
    
    return

def create_top_angle_vis(ply_w_file, png_file, min_y, max_y, out_angle_file, min_angle, max_angle, metadata):
    
    windowSize = 4
    xyScale = 3
    
    plyf = ply_w_file
    gImf = png_file
    
    gIm = Image.open(gImf)
    
    [gWid, gHei] = gIm.size
    upper = round(min_y * gHei)
    lower = round(max_y * gHei)
    
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
    
    yOffset = get_offset_from_metadata(metadata)
    
    angle_data = []
    for i in range(0,32):
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
            localNormal = calcAreaNormalSurface(localP)
            yCoord = np.mean(localP["y"])
            area_ind = get_area_index(yCoord, yOffset) - 1
            
            if localNormal != [] :
                angle_data[area_ind] = np.append(angle_data[area_ind])
                rgb = normals_to_rgb_2(localNormal)
                codeImg.paste(rgb, (i-windowSize*xyScale, j-windowSize, i+windowSize*xyScale+1, j+windowSize+1))
    
    img1 = Image.open(png_file)
    img1 = img1.convert('RGB')
    
    img2 = Image.blend(img1, codeImg, 0.5)
    img3 = img2.crop((0,upper,gWid, lower))
    img3.save(out_angle_file)
    
    
    return

def get_area_index(yCoord, yOffset):
    
    yPosition = yCoord/1000 - yOffset
    
    count = 0
        
    for (ymin, ymax) in terra_common._y_row_s2:
        count = count + 1
        if (yPosition > ymin) and (yPosition <= ymax):
            
            return count
    
    return count

def get_offset_from_metadata(metadata):
    
    try:
        gantry_meta = metadata['lemnatec_measurement_metadata']['gantry_system_variable_metadata']
        scan_direction = gantry_meta["scanisinpositivedirection"]
        
    except KeyError as err:
        fail('Metadata file missing key: ' + err.args[0])
        
    if scan_direction == 'False':
        yShift = -25.1
    else:
        yShift = -3
    
    return yShift

def creat_angle_visualization(ply_w_file, png_file, min_y, max_y, out_angle_file):
    
    windowSize = 4
    xyScale = 3
    
    plyf = ply_w_file
    gImf = png_file
    
    gIm = Image.open(gImf)
    
    [gWid, gHei] = gIm.size
    upper = round(min_y * gHei)
    lower = round(max_y * gHei)
    
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
            if localNormal != [] :
                rgb = normals_to_rgb_2(localNormal)
                codeImg.paste(rgb, (i-windowSize*xyScale, j-windowSize, i+windowSize*xyScale+1, j+windowSize+1))
    
    img1 = Image.open(png_file)
    img1 = img1.convert('RGB')
    
    img2 = Image.blend(img1, codeImg, 0.5)
    img3 = img2.crop((0,upper,gWid, lower))
    img3.save(out_angle_file)
    
    return

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
    if s1 < s2:
        reval = np.append(normals, centerPoint)
        return reval
    else:
        return []

def save_points(ply_data, out_file, id):
    
    X = ply_data["x"]
    Y = ply_data["y"]
    Z = ply_data["z"]
    data_size = X.size
    
    index = (np.linspace(0,data_size-1,5000)).astype('int')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if data_size < 10:
        plt.savefig(out_file)
        plt.close()
        return
    
    colors = cm.rainbow(np.linspace(0, 1, 32))
    X = X[index]
    Y = Y[index]
    Z = Z[index]
    ax.scatter(X,Y,Z,color=colors[id], s=2)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    ax.view_init(10, 0)
    plt.draw()
    plt.savefig(out_file)
    plt.close()
    
    return

def save_sub_ply(subData, src, outFile):
    
    src.elements[0].data = subData
    src.write(outFile)
    
    return

def list_target_scans(in_dir, out_dir, xPos):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    list_dirs = os.walk(in_dir)
    
    out_file = open(os.path.join(out_dir, str(xPos)+'.txt'), 'w')
    
    for root, dirs, files in list_dirs:
        for d in dirs:
            dir_path = os.path.join(in_dir, d)
            if not os.path.isdir(dir_path):
                continue
            
            tar_dir = find_target_scan(dir_path, xPos)
            if tar_dir != '':
                out_file.write('%s\n' % tar_dir)
    
    out_file.close()
    
    return

def find_target_scan(in_dir, xPos):
    
    list_dirs = os.walk(in_dir)
    for root, dirs, files in list_dirs:
        for d in dirs:
            dir_path = os.path.join(in_dir, d)
            if not os.path.isdir(dir_path):
                continue
            
            json_file, png_file = find_files(dir_path)
            if json_file == [] or png_file == []:
                continue
            
            metadata = terra_common.lower_keys(terra_common.load_json(json_file))
            x_position, scan_direction = parse_metadata(metadata)
            if abs(xPos - x_position) < 0.15:
                return dir_path
            
    return ''

def count_scan_times(in_dir):
    
    list_files = os.walk(in_dir)
    
    pos_list = []
    
    for root, dirs, files in list_files:
        for f in files:
            if not f.endswith('.txt'):
                continue
            
            file_path = os.path.join(in_dir, f)
            
            if not os.path.exists(file_path):
                continue
            
            file_handle = open(file_path, 'r')
            ind = 0
            npArr = np.zeros(700)
            while 1:
                line = file_handle.readline()
                if not line:
                    break
                pos = round(float(line), 1)*10
                npArr[ind] = int(pos)
                ind = ind + 1
            file_handle.close()
            
            np.unique(npArr)
            pos_list.append(npArr.astype('int'))
    
    hist = np.zeros(2400)
    for npArr in pos_list:
        for i in range(0,npArr.size):
            hist[npArr[i]] = hist[npArr[i]] + 1
    
    count_file = open(os.path.join(in_dir, 'posCount.txt'), 'w')
    for i in range(hist.size):
        if hist[i] !=  0:
            count_file.write('%d, %d\n' % (i, hist[i]))
            
    count_file.close()
    
    return

def scan_3d_data_for_x_position(in_dir, out_dir):
    
    list_dirs = os.walk(in_dir)
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    for root, dirs, files in list_dirs:
        for d in dirs:
            dir_path = os.path.join(in_dir, d)
            if not os.path.isdir(dir_path):
                continue
            
            full_day_scanned_x_position(dir_path, out_dir)
    
    
    return

def full_day_scanned_x_position(in_dir, out_dir):
    
    str_date = os.path.basename(in_dir)
    
    txt_file = open(os.path.join(out_dir, str_date+'.txt'), 'w')
    
    list_dirs = os.walk(in_dir)
    for root, dirs, files in list_dirs:
        for d in dirs:
            dir_path = os.path.join(in_dir, d)
            if not os.path.isdir(dir_path):
                continue
            
            json_file, png_file = find_files(dir_path)
            if json_file == [] or png_file == []:
                continue
            
            metadata = terra_common.lower_keys(terra_common.load_json(json_file))
            x_position, scan_direction = parse_metadata(metadata)
            txt_file.write('%f\n' % x_position)
            
    txt_file.close()
    
    return

def test():
    
    
    plydata = PlyData.read('/Users/Desktop/Scanner3D/2016-09-29/2016-09-29__04-09-29-758/d2009394-fde3-40b8-9731-2e5187a82370__Top-heading-west_0.ply')
    data = plydata.elements[0].data
    scaleParam = 1000
    
    for i in range(0, 16):
        ymin = terra_common._y_row_s2[i*2+1][0] * scaleParam
        ymax = terra_common._y_row_s2[i*2][1] * scaleParam
        specifiedIndex = np.where((data["y"]>ymin) & (data["y"]<ymax))
        target = data[specifiedIndex]
        
        
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    
        x = target["x"]
        y = target["y"]
        
        c = ('r')
        ax.scatter(x, y, 0, zdir='y', c=c)
        
        ax.legend()
        
        plt.show()
        plt.close()
    
    
    return


def one_day_process(png_dir, out_dir):
    
    if not os.path.isdir(png_dir):
        return
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    scan_date = os.path.basename(os.path.normpath(png_dir))
    
    list_dirs = os.listdir(png_dir)
    
    saved_flag = np.zeros(865)
    
    for sub_dir in list_dirs:
        png_path = os.path.join(png_dir, sub_dir)
        if not os.path.isdir(png_path):
            continue
        
        plotNums = meta_save_sample_image(png_path, out_dir, saved_flag, scan_date)
        
        for i in plotNums:
            saved_flag[i] = 1
    
    
    
    return

def meta_save_sample_image(in_dir, out_dir, saved_flag, scan_date):
    
    json_file, png_file = find_files(in_dir)
    
    if json_file == [] or png_file == []:
        return []
    
    if not os.path.exists(png_file) or not os.path.exists(json_file):
        return []
    
    metadata = terra_common.lower_keys(terra_common.load_json(json_file))
    
    x_position, scan_direction = parse_metadata(metadata)
    src_img = Image.open(png_file)
    in_data = np.array(src_img).astype('float')
    dmin = in_data.min()
    dmax = in_data.max()
    out_data = (in_data-dmin)/(dmax-dmin)*255
    dst_img = Image.fromarray(out_data)
    dst_img = dst_img.convert('P')
    width, height = src_img.size
    scaleParam = 1000
    yRange = 16
    
    plotNums = []
    
    for i in range(0,yRange):
        plotNum = int(field_2_plot_for_season_two(x_position, i+1))
        if saved_flag[plotNum] == 1:
            continue
        
        y = i
        if scan_direction == 'False':
            y = yRange - 1 - i
        
        ymin = terra_common._y_row_s2[y*2+1][0] * scaleParam
        ymax = terra_common._y_row_s2[y*2][1] * scaleParam
            
        if ymin > height or ymax < 0:
            continue
        
        if ymin < 0:
            ymin = 0;
        if ymax > height:
            ymax = height
            
        sub_path = os.path.join(out_dir, str(plotNum))
        if not os.path.isdir(sub_path):
            os.mkdir(sub_path)
        
        out_path = os.path.join(out_dir, str(plotNum), scan_date+'.png')
        img_save = dst_img.crop((0,int(ymin),width,int(ymax)))
        img_save.save(out_path)
        plotNums.append(plotNum)
    
    return plotNums

def find_files(in_dir):
    
    json_suffix = os.path.join(in_dir, '*_metadata.json')
    jsons = glob(json_suffix)
    if len(jsons) == 0:
        terra_common.fail('Could not find .json file')
        return [], []
        
        
    png_suffix = os.path.join(in_dir, '*west_0_g.png')
    pngs = glob(png_suffix)
    if len(pngs) == 0:
        terra_common.fail('Could not find .bin file')
        return [], []
    
    return jsons[0], pngs[0]

def find_ply_file(in_dir):
    
    plyW_suffix = os.path.join(in_dir, '*west_0.ply')
    plyW = glob(plyW_suffix)
    if len(plyW) == 0:
        terra_common.fail('Could not find *west_0.ply file')
        return []
    
    return plyW[0]

def parse_metadata(metadata):
    
    try:
        gantry_meta = metadata['lemnatec_measurement_metadata']['gantry_system_variable_metadata']
        gantry_x = gantry_meta["position x [m]"]
        scan_direction = gantry_meta["scanisinpositivedirection"]

    except KeyError as err:
        terra_common.fail('Metadata file missing key: ' + err.args[0])
        return 0, 'True'
        
    x_position = float(gantry_x)
    
    return x_position, scan_direction

def field_2_plot_for_season_two(x_position, y_row):

    xRange = 0
    count = 0
        
    for (xmin, xmax) in terra_common._x_range_s2:
        count = count + 1
        if (x_position > xmin) and (x_position <= xmax):
            xRange = 55 - count
            
            plotNum = convt.fieldPartition_to_plotNum(xRange, y_row)
            
            return plotNum
    
    return 0



def fail(reason):
    print >> sys.stderr, reason

if __name__ == "__main__":

    main()
