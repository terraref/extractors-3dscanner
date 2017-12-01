'''
Created on Aug 9, 2016

@author: Zongyang Li
'''

import os, sys, json, argparse, shutil, terra_common
import numpy as np
from glob import glob
from plyfile import PlyData
from matplotlib import cm
import matplotlib.pyplot as plt
from datetime import date
from terrautils import betydb

convt = terra_common.CoordinateConverter()

PLOT_RANGE_NUM = 54
PLOT_COL_NUM = 32
HIST_BIN_NUM = 400

# [(35th percentile from E + 35th percentile from W) / 2] cm * 0.97841 + 25.678cm
B_TH_PERC = 0.35
B_F_SLOPE = 0.97841
B_F_OFFSET = 25.678

# [37th percentile from E sensor] cm * 0.94323 + 26.41cm
E_TH_PERC = 0.37
E_F_SLOPE = 0.94323
E_F_OFFSET = 26.41

# [41st percentile from W sensor] cm * 0.98132 + 24.852cm
W_TH_PERC = 0.41
W_F_SLOPE = 0.98132
W_F_OFFSET = 24.852


def process_all_scanner_data(ply_parent, json_parent, out_parent):
    
    list_dirs = os.listdir(ply_parent)
    
    start_ind = 0
    ind = 0
    
    for dir in list_dirs:
        ply_path = os.path.join(ply_parent, dir)
        json_path = os.path.join(json_parent, dir)
        out_path = os.path.join(out_parent, dir)
        if not os.path.isdir(ply_path):
            continue
        if not os.path.isdir(json_path):
            continue
        
        ind = ind + 1
        if ind < start_ind:
            continue
        print('start processing' + out_path)
        full_day_gen_hist(ply_path, json_path, out_path)
        try:
            create_normalization_hist(out_path, out_path)
        except Exception as ex:
            fail(str(ex))

    return


def process_one_month_data(ply_parent, json_parent, out_parent, str_year, str_month, str_start_date, str_end_date):
    
    for day in range(int(str_start_date), int(str_end_date)+1):
        target_date = date(int(str_year), int(str_month), day)
        str_date = target_date.isoformat()
        print(str_date)
        ply_path = os.path.join(ply_parent, str_date)
        json_path = os.path.join(json_parent, str_date)
        out_path = os.path.join(out_parent, str_date)
        if not os.path.isdir(ply_path):
            continue
        if not os.path.isdir(json_path):
            continue
        try:
            q_flag = convt.bety_query(str_date)
            if not q_flag:
                continue
            
            full_day_gen_hist(ply_path, json_path, out_path)
    
            full_day_array_to_xlsx(out_path)
            
            insert_height_traits_into_betydb(out_path, out_path, str_date, B_TH_PERC, 'e')
        except Exception as ex:
            fail(str_date + str(ex))
    
    return


def full_day_gen_hist(ply_path, json_path, out_path):
    
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    
    list_dirs = os.walk(ply_path)
    
    for root, dirs, files in list_dirs:
        for d in dirs:
            print("Start processing " + d)
            p_path = os.path.join(ply_path, d)
            j_path = os.path.join(json_path, d)
            o_path = os.path.join(out_path, d)
            if not os.path.isdir(p_path):
                continue
            
            if not os.path.isdir(j_path):
                continue
            
            gen_hist(p_path, j_path, o_path)
    
    return
    

def gen_hist(ply_path, json_path, out_dir):
    
    if not os.path.isdir(ply_path):
        fail('Could not find input directory: ' + ply_path)
        
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
        
    # parse json file
    metas, ply_file_wests, ply_file_easts = find_input_files(ply_path, json_path)
    
    for meta, ply_file_west, ply_file_east in zip(metas, ply_file_wests, ply_file_easts):
        json_dst = os.path.join(out_dir, meta)
        if os.path.exists(json_dst):
            return
        
        metadata = lower_keys(load_json(os.path.join(json_path, meta)))
        # make all our keys lowercase since keys appear to change case (???)
        
        center_position = get_position(metadata)  # (x, y, z) in meters
        scanDirection = get_direction(metadata)  # scan direction
        
        plywest = PlyData.read(os.path.join(ply_path, ply_file_west))
        hist_w, heightest_w = gen_height_histogram(plywest, scanDirection, out_dir, 'w', center_position)
        
        histPath = os.path.join(out_dir, 'hist_w.npy')
        np.save(histPath, hist_w)
        heightestPath = os.path.join(out_dir, 'top_w.npy')
        np.save(heightestPath, heightest_w)
        
        plyeast = PlyData.read(os.path.join(ply_path, ply_file_east))
        hist_e, heightest_e = gen_height_histogram(plyeast, scanDirection, out_dir, 'e', center_position)
        
        histPath = os.path.join(out_dir, 'hist_e.npy')
        np.save(histPath, hist_e)
        heightestPath = os.path.join(out_dir, 'top_e.npy')
        np.save(heightestPath, heightest_e)
        
        shutil.copyfile(os.path.join(json_path, meta), json_dst)
    
    return

def get_height_result(in_dir, sensor_d):
    
    if not os.path.isdir(in_dir):
        fail('Could not find input directory: ' + in_dir)
    
    plotNum = np.zeros((PLOT_COL_NUM,1))
    hist_data = np.zeros((PLOT_COL_NUM,HIST_BIN_NUM))
    top_data = np.zeros((PLOT_COL_NUM,1))
    
    # parse json file
    metafile, hist, top = find_result_files(in_dir, sensor_d)
    if metafile == [] or hist == []:
        return plotNum, hist_data, top_data

    metadata = lower_keys(load_json(metafile))
    center_position = get_position(metadata)
    hist_data = np.load(hist, 'r')
    top_data = np.load(top, 'r')
        
    for i in range(0,PLOT_COL_NUM):
        plotNum[i] = field_2_plot(center_position[0], i+1)
    
    return plotNum.astype('int'), hist_data, top_data


def create_normalization_hist(in_dir, out_dir):
    
    list_dirs = os.walk(in_dir)
    heightHist = np.zeros((PLOT_COL_NUM*PLOT_RANGE_NUM, HIST_BIN_NUM))
    plotScanCount = np.zeros((PLOT_COL_NUM*PLOT_RANGE_NUM))
    
    for root, dirs, files in list_dirs:
        for d in dirs:
            full_path = os.path.join(in_dir, d)
            if not os.path.isdir(full_path):
                continue
            
            plotNum, hist, top = get_height_result(full_path)
            if len(plotNum) < PLOT_COL_NUM:
                continue
            
            for j in range(0,plotNum.size):
                heightHist[plotNum[j]-1] = heightHist[plotNum[j]-1]+hist[j]
                plotScanCount[plotNum[j]-1] = plotScanCount[plotNum[j]-1] + 1
                
    for i in range(0, PLOT_COL_NUM*PLOT_RANGE_NUM):
        if plotScanCount[i] != 0:
            heightHist[i] = heightHist[i]/plotScanCount[i]
    
    histfile = os.path.join(in_dir, 'heightHist.npy')
    np.save(histfile, heightHist)
    
    hist_out_file = os.path.join(in_dir, 'hist.txt')
    np.savetxt(hist_out_file, np.array(heightHist), delimiter="\t")
    
    return

def full_day_array_to_xlsx(in_dir):
    
    for sensor_d in ['e','w']:
        list_dirs = os.walk(in_dir)
        heightHist = np.zeros((PLOT_COL_NUM*PLOT_RANGE_NUM, HIST_BIN_NUM))
        topMat = np.zeros((PLOT_COL_NUM*PLOT_RANGE_NUM))
        for root, dirs, files in list_dirs:
            for d in dirs:
                full_path = os.path.join(in_dir, d)
                if not os.path.isdir(full_path):
                    continue
                
                plotNum, hist, top = get_height_result(full_path, sensor_d)
                if len(plotNum) < PLOT_COL_NUM:
                    continue
                
                for j in range(0,plotNum.size):
                    if plotNum[j] == 0:
                        continue
                    
                    heightHist[plotNum[j]-1] = heightHist[plotNum[j]-1]+hist[j]
                    
                    if topMat[plotNum[j]-1] < top[j]:
                        topMat[plotNum[j]-1] = top[j]
        
        histfile = os.path.join(in_dir, 'heightHist_'+sensor_d+'.npy')
        topfile = os.path.join(in_dir, 'topHist_'+sensor_d+'.npy')
        np.save(histfile, heightHist)
        np.save(topfile, topMat)
        '''
        hist_out_file = os.path.join(in_dir, 'hist_'+sensor_d+'.txt')
        np.savetxt(hist_out_file, np.array(heightHist), delimiter="\t")
        top_out_file = os.path.join(in_dir, 'top_'+sensor_d+'.txt')
        np.savetxt(top_out_file, np.array(topMat), delimiter="\t")
        '''
    
    return

def insert_height_traits_into_betydb(in_dir, out_dir, str_date, param_percentile, sensor_d):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    hist_e, hist_w = load_histogram_from_both_npy(in_dir)
    
    out_file = os.path.join(out_dir, str_date+'_height.csv')
    csv = open(out_file, 'w')
    
    (fields, traits) = get_traits_table_height()
        
    csv.write(','.join(map(str, fields)) + '\n')
        
    for j in range(0, PLOT_COL_NUM*PLOT_RANGE_NUM):
        targetHist_e = hist_e[j,:]
        targetHist_w = hist_w[j, :]
        plotNum = j+1
        if (targetHist_e.max() == 0) or (targetHist_w.max())==0:
            continue
        else:
            targetHist_e = targetHist_e/np.sum(targetHist_e)
            quantiles_e = np.cumsum(targetHist_e)
            b = np.arange(len(quantiles_e))
            c = b[quantiles_e>param_percentile]
            quantile_e = min(c)
            
            targetHist_w = targetHist_w/np.sum(targetHist_w)
            quantiles_w = np.cumsum(targetHist_w)
            b = np.arange(len(quantiles_w))
            c = b[quantiles_w>param_percentile]
            quantile_w = min(c)
            
            estHeight = (quantile_e + quantile_w)/2
                
            str_time = str_date+'T12:00:00'
            traits['local_datetime'] = str_time
            traits['canopy_height'] = str((B_F_SLOPE*float(estHeight) + B_F_OFFSET)/100.0)
            traits['site'] = parse_site_from_plotNum_1728(plotNum)
            trait_list = generate_traits_list_height(traits)
            csv.write(','.join(map(str, trait_list)) + '\n')
    
    
    csv.close()
    betydb.submit_traits(out_file, filetype='csv')
    
    
    return

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


def load_histogram_from_npy(in_dir, sensor_d):
    
    file_path = os.path.join(in_dir, 'heightHist_' + sensor_d + '.npy')
    hist = np.load(file_path)
    
    return hist

def load_histogram_from_both_npy(in_dir):
    
    file_path_e = os.path.join(in_dir, 'heightHist_e.npy')
    if not os.path.exists(file_path_e):
        return []
    hist_e = np.load(file_path_e)
    
    file_path_w = os.path.join(in_dir, 'heightHist_w.npy')
    if not os.path.exists(file_path_w):
        return []
    hist_w = np.load(file_path_w)
    
    return hist_e, hist_w

def get_traits_table_height():
    
    fields = ('local_datetime', 'canopy_height', 'access_level', 'species', 'site',
              'citation_author', 'citation_year', 'citation_title', 'method')
    traits = {'local_datetime' : '',
              'height' : [],
              'access_level': '2',
              'species': 'Sorghum bicolor',
              'site': [],
              'citation_author': '"Zongyang, Li"',
              'citation_year': '2016',
              'citation_title': 'Maricopa Field Station Data and Metadata',
              'method': 'Scanner 3d ply data to height'}

    return (fields, traits)

def generate_traits_list_height(traits):
    # compose the summary traits
    trait_list = [  traits['local_datetime'],
                    traits['canopy_height'],
                    traits['access_level'],
                    traits['species'],
                    traits['site'],
                    traits['citation_author'],
                    traits['citation_year'],
                    traits['citation_title'],
                    traits['method']
                ]

    return trait_list

def field_x_2_range(x_position):
    
    xRange = 0
    
    for i in range(PLOT_RANGE_NUM):
        xmin = convt.np_bounds[i][0][0]
        xmax = convt.np_bounds[i][0][1]
        if (x_position > xmin) and (x_position <= xmax):
            xRange = i + 1
    
    return xRange

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

def find_result_files(in_dir, sensor_d):
    
    metadata_suffix = os.path.join(in_dir, '*_metadata.json')
    metas = glob(metadata_suffix)
    if len(metas) == 0:
        #fail('No metadata file found in input directory.')
        return [], [], []

    hist_file = os.path.join(in_dir, 'hist_'+sensor_d+'.npy')
    top_file = os.path.join(in_dir, 'top_'+sensor_d+'.npy')
    if os.path.isfile(hist_file) == False | os.path.isfile(top_file) == False:
        #fail('No hist file or top file in input directory')
        return [], [], []

    return metas[0], hist_file, top_file

    
def load_json(meta_path):
    try:
        with open(meta_path, 'r') as fin:
            return json.load(fin)
    except Exception as ex:
        fail('Corrupt metadata file, ' + str(ex))
    
    
def lower_keys(in_dict):
    if type(in_dict) is dict:
        out_dict = {}
        for key, item in in_dict.items():
            out_dict[key.lower()] = lower_keys(item)
        return out_dict
    elif type(in_dict) is list:
        return [lower_keys(obj) for obj in in_dict]
    else:
        return in_dict
    
    
def find_input_files(ply_path, json_path):
    metadata_suffix = '_metadata.json'
    metas = [os.path.basename(meta) for meta in glob(os.path.join(json_path, '*' + metadata_suffix))]
    if len(metas) == 0:
        fail('No metadata file found in input directory.')

    ply_suffix = 'Top-heading-west_0.ply'
    plyWests = [os.path.basename(ply) for ply in glob(os.path.join(ply_path, '*' + ply_suffix))]
    if len(plyWests) == 0:
        fail('No west file found in input directory.')
        
    ply_suffix = 'Top-heading-east_0.ply'
    plyEasts = [os.path.basename(ply) for ply in glob(os.path.join(ply_path, '*' + ply_suffix))]
    if len(plyEasts) == 0:
        fail('No east file found in input directory.')

    return metas, plyWests, plyEasts

def get_position(metadata):
    try:
        gantry_meta = metadata['lemnatec_measurement_metadata']['gantry_system_variable_metadata']
        gantry_x = gantry_meta["position x [m]"]
        gantry_y = gantry_meta["position y [m]"]
        gantry_z = gantry_meta["position z [m]"]
        
        sensor_fix_meta = metadata['lemnatec_measurement_metadata']['sensor_fixed_metadata']
        camera_x = '2.070'#sensor_fix_meta['scanner west location in camera box x [m]']
        camera_z = '1.135'
        

    except KeyError as err:
        fail('Metadata file missing key: ' + err.args[0])

    try:
        x = float(gantry_x) + float(camera_x)
        y = float(gantry_y)
        z = float(gantry_z) + float(camera_z)
    except ValueError as err:
        fail('Corrupt positions, ' + err.args[0])
    return (x, y, z)


def get_direction(metadata):
    try:
        gantry_meta = metadata['lemnatec_measurement_metadata']['gantry_system_variable_metadata']
        scan_direction = gantry_meta["scanisinpositivedirection"]
        
    except KeyError as err:
        fail('Metadata file missing key: ' + err.args[0])
        
    return scan_direction


def gen_height_histogram(plydata, scanDirection, out_dir, sensor_d, center_position):
    
    gantry_z_offset = 0.35
    zGround = (3.445 - center_position[2] + gantry_z_offset)*1000
    yRange = PLOT_COL_NUM
    yShift = offset_choice(scanDirection, sensor_d)
    zOffset = 10
    zRange = [-2000, 2000]
    scaleParam = 1000
    hist = np.zeros((yRange, (zRange[1]-zRange[0])/zOffset))
    heightest = np.zeros((yRange, 1))
    data = plydata.elements[0].data
    
    xRange = field_x_2_range(center_position[0])
    
    if data.size == 0:
        return hist, heightest
    
    for i in range(0, yRange):
        ymin = (convt.np_bounds[xRange][i][2]+yShift) * scaleParam
        ymax = (convt.np_bounds[xRange][i][3]+yShift) * scaleParam
        specifiedIndex = np.where((data["y"]>ymin) & (data["y"]<ymax))
        target = data[specifiedIndex]
        
        for j in range(0, HIST_BIN_NUM):
            zmin = zGround + j * zOffset
            zmax = zGround + (j+1) * zOffset
            zIndex = np.where((target["z"]>zmin) & (target["z"]<zmax));
            num = len(zIndex[0])
            hist[i][j] = num
    
        zTop = 0;
        if len(specifiedIndex[0])!=0:
            zTop = target["z"].max()
        
        heightest[i] = zTop
        
        '''
        out_basename = str(i)+'_'+sensor_d+'.png'
        out_file = os.path.join(out_dir, out_basename)
        save_points(target, out_file, i)
        #save_sub_ply(target, plydata, os.path.join(out_dir, str(i)+'.ply'))
        np.savetxt(os.path.join(out_dir, str(i)+'.txt'), hist[i])
        '''
    
    return hist, heightest

def offset_choice(scanDirection, sensor_d):
    
    if sensor_d == 'w':
        if scanDirection == 'True':
            ret = -3.60#-3.08
        else:
            ret = -25.711#-25.18
            
    if sensor_d == 'e':
        if scanDirection == 'True':
            ret = -3.60#-3.08
        else:
            ret = -25.711#-25.18
    
    return ret

def gen_height_histogram_for_Roman(plydata, scanDirection, out_dir, sensor_d, center_position):
    
    gantry_z_offset = 0.35
    zGround = (3.445 - center_position[2] + gantry_z_offset)*1000
    yRange = PLOT_COL_NUM
    yShift = offset_choice(scanDirection, sensor_d)
    zOffset = 10
    zRange = [-2000, 2000]
    scaleParam = 1000
    hist = np.zeros((yRange, (zRange[1]-zRange[0])/zOffset))
    heightest = np.zeros((yRange, 1))
    data = plydata.elements[0].data
    
    xRange = field_x_2_range(center_position[0])
    
    if data.size == 0:
        return hist, heightest
    
    for i in range(0, yRange):
        # ymin = (terra_common._y_row_s4[i][0]+yShift) * scaleParam
        # ymax = (terra_common._y_row_s4[i][1]+yShift) * scaleParam
        ymin = (convt.np_bounds[xRange][i][2]+yShift) * scaleParam
        ymax = (convt.np_bounds[xRange][i][3]+yShift) * scaleParam
        specifiedIndex = np.where((data["y"]>ymin) & (data["y"]<ymax))
        target = data[specifiedIndex]
        
        for j in range(0, HIST_BIN_NUM):
            zmin = zGround + j * zOffset
            zmax = zGround + (j+1) * zOffset
            zIndex = np.where((target["z"]>zmin) & (target["z"]<zmax));
            num = len(zIndex[0])
            hist[i][j] = num
    
        zTop = 0;
        if len(specifiedIndex[0])!=0:
            zTop = target["z"].max()
        
        heightest[i] = zTop
        '''
        out_basename = str(i)+'_'+sensor_d+'.png'
        out_file = os.path.join(out_dir, out_basename)
        save_points(target, out_file, i)
        #save_sub_ply(target, plydata, os.path.join(out_dir, str(i)+'.ply'))
        np.savetxt(os.path.join(out_dir, str(i)+'.txt'), hist[i])
        '''
    
    return hist, heightest

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

def fail(reason):
    print >> sys.stderr, reason


