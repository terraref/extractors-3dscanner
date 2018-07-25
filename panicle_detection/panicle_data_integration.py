'''
Created on Mar 28, 2018

@author: zli
'''

import os, shutil, terra_common
from glob import glob
from datetime import date, timedelta
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
from matplotlib import cm

def crop_reflectance_image(img, out_img_size=(1024,1024)):
    
    width, height, channels = img.shape
    
    i_wid_max = int(round(width/out_img_size[0]))
    i_hei_max = int(round(height/out_img_size[1]))
    
    x_ind = []
    y_ind = []
    img_vec = []
            
            
    for i in range(i_wid_max):
        for j in range(i_hei_max):
            cropped_img = img[i*out_img_size[1]:(i+1)*out_img_size[1], j*out_img_size[0]:(j+1)*out_img_size[0]]
            #img_path = os.path.join(out_dir, base_name+'_'+str(i)+'_'+str(j)+'.jpg')
            #cv2.imwrite(img_path, crop_img)
            x_ind.append(i)
            y_ind.append(j)
            img_vec.append(cropped_img)
    
    return x_ind, y_ind, img_vec

# integrage all the txt output in a super directory
def full_day_output_integrate(super_dir, out_dir):
    
    # list initialization
    totalPlot = 1728
    list_dirs = os.listdir(super_dir)
    volList = [[] for i in range(totalPlot+1)]
    cntList = [[] for i in range(totalPlot+1)]
    areaList = [[] for i in range(totalPlot+1)]
    densityList = [[] for i in range(totalPlot+1)]
    
    # scan super directory
    for d in list_dirs:
            full_path = os.path.join(super_dir, d)
            if not os.path.isdir(full_path):
                continue
            
            # load txt file
            vol_file = os.path.join(full_path, 'volume.txt')
            counting_file = os.path.join(full_path, 'counting.txt')
            area_file = os.path.join(full_path, 'area.txt')
            den_file = os.path.join(full_path, 'density.txt')
            if not os.path.isfile(vol_file) or not os.path.isfile(counting_file) or not os.path.isfile(area_file) or not os.path.isfile(den_file):
                continue
            
            # data into list
            vol_handle = open(vol_file, 'r')
            for line in vol_handle:
                fields = line.split(',')
                plotNum = int(fields[0])
                vol = float(fields[1])
                volList[plotNum].append(vol)
            vol_handle.close()
            
            cnt_handle = open(counting_file, 'r')
            for line in cnt_handle:
                fields = line.split(',')
                plotNum = int(fields[0])
                cnt = int(fields[1])
                cntList[plotNum].append(cnt)
            cnt_handle.close()
            
            area_handle = open(area_file, 'r')
            for line in area_handle:
                fields = line.split(',')
                plotNum = int(fields[0])
                area = float(fields[1])
                areaList[plotNum].append(area)
            area_handle.close()
            
            den_handle = open(den_file, 'r')
            for line in den_handle:
                fields = line.split(',')
                plotNum = int(fields[0])
                den = float(fields[1])
                densityList[plotNum].append(den)
            den_handle.close()
            
    # throw out outliers
    #volList, cntList, areaList, densityList = data_integration(volList, cntList, areaList, densityList)
            
    
    # calculate mean and median for each plot
    plot_counting = np.zeros((totalPlot+1))
    vol_mean = np.zeros((totalPlot+1))
    vol_median = np.zeros((totalPlot+1))
    for i in range(1, len(volList)):
        target_vols = volList[i]
        if len(target_vols) == 0:
            continue
        vol_mean[i] = np.mean(target_vols)
        vol_median[i] = np.median(target_vols)
        plot_counting[i] = len(target_vols)
        
    cnt_mean = np.zeros((totalPlot+1))
    cnt_median = np.zeros((totalPlot+1))
    plot_total_cnt = np.zeros((totalPlot+1))
    
    for i in range(1, len(cntList)):
        target_cnts = cntList[i]
        if len(target_cnts) == 0:
            continue
        cnt_mean[i] = np.mean(target_cnts)
        cnt_median[i] = np.median(target_cnts)
        plot_total_cnt[i] = np.sum(target_cnts)
        
        
    area_mean = np.zeros((totalPlot+1))
    area_median = np.zeros((totalPlot+1))
    for i in range(1, len(areaList)):
        target_areas = areaList[i]
        if len(target_areas) == 0:
            continue
        area_mean[i] = np.mean(target_areas)
        area_median[i] = np.median(target_areas)
        
    density_mean = np.zeros((totalPlot+1))
    density_median = np.zeros((totalPlot+1))
    for i in range(1, len(cntList)):
        target_dens = densityList[i]
        if len(target_dens) == 0:
            continue
        density_mean[i] = np.mean(target_dens)
        density_median[i] = np.median(target_dens)
    
        
    # save value to new files
    vol_mean_file = os.path.join(out_dir, 'vol_mean.txt')
    np.savetxt(vol_mean_file, vol_mean)
    
    vol_median_file = os.path.join(out_dir, 'vol_median.txt')
    np.savetxt(vol_median_file, vol_median)
    
    cnt_mean_file = os.path.join(out_dir, 'cnt_mean.txt')
    np.savetxt(cnt_mean_file, cnt_mean)
    
    cnt_median_file = os.path.join(out_dir, 'cnt_median.txt')
    np.savetxt(cnt_median_file, cnt_median)
    
    area_mean_file = os.path.join(out_dir, 'area_mean.txt')
    np.savetxt(area_mean_file, area_mean)
    
    area_median_file = os.path.join(out_dir, 'area_median.txt')
    np.savetxt(area_median_file, area_median)
    
    plot_counting_file = os.path.join(out_dir, 'plot_counting.txt')
    np.savetxt(plot_counting_file, plot_counting)
    
    plot_total_cnt_file = os.path.join(out_dir, 'cnt_total.txt')
    np.savetxt(plot_total_cnt_file, plot_total_cnt)
    
    density_mean_file = os.path.join(out_dir, 'density_mean.txt')
    np.savetxt(density_mean_file, density_mean)
    
    density_median_file = os.path.join(out_dir, 'density_median.txt')
    np.savetxt(density_median_file, density_median)
    
    return

def single_plant_visualization(ply_path, json_path, out_dir, sensor_d):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    metadata = terra_common.lower_keys(terra_common.load_json(json_path))
    xShift, yShift = offset_choice(metadata, sensor_d)
    scaleParam = 1000
    
    base_name = 'test'
    yRange = 32
    ply_data = PlyData.read(ply_path)
    src_data = PlyData.read(ply_path)
    data = ply_data.elements[0].data
    for i in range(0, yRange):
        ymin = (terra_common._y_row_s2[i][0]-yShift) * scaleParam
        ymax = (terra_common._y_row_s2[i][1]-yShift) * scaleParam
        specifiedIndex = np.where((data["y"]>ymin) & (data["y"]<ymax))
        target = data[specifiedIndex]
        
        out_png_file = os.path.join(out_dir, base_name+'_'+str(i)+'.png')
        save_sub_ply(target, src_data, os.path.join(out_dir, base_name+'_'+str(i)+'.ply'))
        save_points(target, out_png_file, 5)
    
    return

def save_sub_ply(subData, src, outFile):
    
    src.elements[0].data = subData
    src.write(outFile)
    
    return

def save_points(ply_data, out_file, id):
    
    X = ply_data["x"]
    Y = ply_data["y"]
    Z = ply_data["z"]
    data_size = X.size
    
    index = (np.linspace(0,data_size-1,10000)).astype('int')
    
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

def offset_choice(metadata, sensor_direction='w'):
    
    scanDirectory = get_direction(metadata)
    center_position = get_position(metadata)
    
    
    if scanDirectory == 'True':
        yShift = 3.45
    else:
        yShift = 25.711
        
    xShift = 0.082 + center_position[0]
    
    return xShift, yShift

def get_direction(metadata):
    try:
        gantry_meta = metadata['lemnatec_measurement_metadata']['gantry_system_variable_metadata']
        scan_direction = gantry_meta["scanisinpositivedirection"]
        
    except KeyError as err:
        terra_common.fail('Metadata file missing key: ' + err.args[0])
        
    return scan_direction

def get_position(metadata):
    try:
        gantry_meta = metadata['lemnatec_measurement_metadata']['gantry_system_variable_metadata']
        gantry_x = gantry_meta["position x [m]"]
        gantry_y = gantry_meta["position y [m]"]
        gantry_z = gantry_meta["position z [m]"]
        
        sensor_fix_meta = metadata['lemnatec_measurement_metadata']['sensor_fixed_metadata']
        camera_x = '2.070'#sensor_fix_meta['scanner west location in camera box x [m]']
        

    except KeyError as err:
        terra_common.fail('Metadata file missing key: ' + err.args[0])

    try:
        x = float(gantry_x) + float(camera_x)
        y = float(gantry_y)
        z = float(gantry_z)
    except ValueError as err:
        terra_common.fail('Corrupt positions, ' + err.args[0])
    return (x, y, z)

# plot image visualization, split saved plot images into referencing directories
def split_images(image_dir, plot_dir, out_dir, plotNum):
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
        
    plot_out_dir = os.path.join(out_dir, str(plotNum))
    if not os.path.isdir(plot_out_dir):
        os.mkdir(plot_out_dir)
        
        
    plot_search_name = os.path.join(plot_dir, '*_'+str(plotNum)+'.png')
    fileList = glob(plot_search_name)
    for fileName in fileList:
        dst_path = os.path.join(plot_out_dir, os.path.basename(fileName))
        shutil.copyfile(fileName, dst_path)
    
    d1 = date(2016, 9, 23)
    d2 = date(2016, 11, 20)
    
    delta = d2 - d1
    
    for i in range(delta.days +1):
        str_date = str(d1 + timedelta(days=i))
        full_path = os.path.join(image_dir, str_date)
        if not os.path.isdir(full_path):
            continue
        list_dirs = os.listdir(full_path)
        for d in list_dirs:
            plot_in_dir = os.path.join(full_path, d, 'plotImg')
            if not os.path.isdir(plot_in_dir):
                continue
            
            file_name = os.path.join(plot_in_dir, '*_'+str(plotNum)+'.jpg')
            f = glob(file_name)
            if f == []:
                continue
            
            dst_path = os.path.join(plot_out_dir, str_date+'_'+str(plotNum)+'.jpg')
            shutil.copyfile(f[0], dst_path)
            
    return

# reviewing function, for testing
def plot_data_visualization(super_dir, out_dir, pre_suffix):
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
        
    target_data, date_list = load_target_data(super_dir, pre_suffix)
    day_len = len(date_list)
    
    for i in range(257, 288):
        
        plot_data = []
        for j in range(day_len):
            plot_data.append(target_data[j][i])
        
        data_vis(plot_data, date_list, i, out_dir, pre_suffix)
    
    
    return

# load target data from super directory
def load_target_data(super_dir, pre_suffix):
    
    target_data = []
    date_list = []
    
    d1 = date(2016, 9, 23)
    d2 = date(2016, 11, 20)
    
    delta = d2 - d1
    
    for i in range(delta.days +1):
        str_date = str(d1 + timedelta(days=i))
        full_path = os.path.join(super_dir, str_date)
        np_data = np.zeros(1728)
        if os.path.isdir(full_path):
            file_path = os.path.join(full_path, pre_suffix+'.txt')
            if not os.path.isfile(file_path):
                continue
            
            np_data = np.loadtxt(file_path)
        
        target_data.append(np_data)
        date_list.append(str_date)
    
    return target_data, date_list

# draw plot images using perticular pre_suffix data
def data_vis(plot_data, date_list, i, out_dir, pre_suffix):
    
    x = range(len(plot_data))
    
    for item,ind in zip(plot_data, x):
        if item==0:
            plot_data.remove(item)
            x.remove(ind)
    
    if len(plot_data)==0:
        return
    
    plt.plot(x, plot_data, 'b-')
    for a,b in zip(x, plot_data):
        txt = '(%s,%s)'%(a,b)
        plt.annotate(txt, (a,b), fontsize=6)
    
    plt.title('%s Plot Number %d'%(pre_suffix, i))
    
    out_file = os.path.join(out_dir, '%s_%d.png'%(pre_suffix, i))
    plt.savefig(out_file)
    plt.close()
    return

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


def write_ply(fn, verts):
    #verts = verts.reshape(-1, 3)
    #colors = colors.reshape(-1, 3)
    verts = np.hstack([verts])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d')
        
        
        