'''
Created on Aug 9, 2016

@author: Zongyang Li
'''
import os,sys,json,argparse, shutil, terra_common
import numpy as np
from glob import glob
from plyfile import PlyData, PlyElement
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from datetime import date

convt = terra_common.CoordinateConverter()

def options():
    
    parser = argparse.ArgumentParser(description='Height Distribution Extractor in Roger',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-m", "--mode", help="all day flag, all for all day process, given parent directory as input, one for one day process")
    parser.add_argument("-p", "--ply_dir", help="ply directory")
    parser.add_argument("-j", "--json_dir", help="json directory")
    parser.add_argument("-o", "--out_dir", help="output directory")
    parser.add_argument("-d", "--month", help="a month data process")

    args = parser.parse_args()

    return args


def main():
    print("start...")
    
    args = options()
    
    if args.mode == 'all':
        process_all_scanner_data(args.ply_dir, args.json_dir, args.out_dir)

    if args.mode == 'one':
        full_day_gen_hist(args.ply_dir, args.json_dir, args.out_dir)
            
        full_day_array_to_xlsx_for_roman(args.out_dir)
        
    if args.mode == 'date':
        process_one_month_data(args.ply_dir, args.json_dir, args.out_dir, args.month)
    
    return

def process_all_scanner_data(ply_parent, json_parent, out_parent):
    
    list_dirs = os.listdir( ply_parent )
    
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

def process_one_month_data(ply_parent, json_parent, out_parent, str_month):
    
    for day in range(27, 30):
        target_date = date(2017, int(4), day)
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
            full_day_gen_hist(ply_path, json_path, out_path)
    
            full_day_array_to_xlsx_for_roman(out_path)
        except Exception as ex:
            fail(str_date + str(ex))
    
    return

def gen_fraction_hist(in_dir, plotNum):
    
    list_dirs = os.walk(in_dir)
    
    fraction_data = np.zeros((100))
    DateHist = []
    
    data_count = 0
    
    for root, dirs, files in list_dirs:
        for d in dirs:
            full_path = os.path.join(in_dir, d)
            if not os.path.isdir(full_path):
                continue
            
            frac_suffix = os.path.join(full_path, 'fraction.npy')
            fracfiles = glob(frac_suffix)
            if len(fracfiles) == 0:
                continue
            
            one_day_hist = np.load(fracfiles[0], 'r')
            date_str = full_path[-10:]
            
            DateHist.append(date_str)
            fraction_data[data_count] = one_day_hist[plotNum]
            data_count = data_count + 1
    '''        
    if data_count != 0:       
        draw_field_scanned_in_grid.plot_fraction(fraction_data[0:data_count], DateHist, plotNum, in_dir)
    else:
        print 'No data in this plot'
    '''
    
    return

def gen_plot_heatmap(in_dir, plotNum):
    
    list_dirs = os.walk(in_dir)
    
    plotHist = np.zeros((100, 400))
    DateHist = []
    
    data_count = 0
    
    for root, dirs, files in list_dirs:
        for d in dirs:
            full_path = os.path.join(in_dir, d)
            if not os.path.isdir(full_path):
                continue
            
            hist_suffix = os.path.join(full_path, 'heightHist.npy')
            histfiles = glob(hist_suffix)
            if len(histfiles) == 0:
                continue
            
            one_day_hist = np.load(histfiles[0], 'r')
            date_str = full_path[-10:]
            
            DateHist.append(date_str)
            if one_day_hist[plotNum, :].max() == 0:
                plotHist[data_count, :] = np.NAN
            else:
                plotHist[data_count, :] = one_day_hist[plotNum, :]
            data_count = data_count + 1
    
    '''        
    if data_count != 0:       
        #draw_field_scanned_in_grid.draw_heatmap(plotHist[0:data_count, :], DateHist, plotNum, in_dir)
    else:
        print 'No data in this plot'
    '''
    
    
    return

def full_day_gen_hist(ply_path, json_path, out_path):
    
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    
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
        metadata = lower_keys(load_json(os.path.join(json_path, meta))) # make all our keys lowercase since keys appear to change case (???)
        
        center_position = get_position(metadata) # (x, y, z) in meters
        scanDirection = get_direction(metadata) # scan direction
        
        plywest = PlyData.read(os.path.join(ply_path, ply_file_west))
        hist_w, heightest_w = gen_height_histogram_for_Roman(plywest, scanDirection, out_dir, 'w', center_position)
        
        histPath = os.path.join(out_dir, 'hist_w.npy')
        np.save(histPath, hist_w)
        heightestPath = os.path.join(out_dir, 'top_w.npy')
        np.save(heightestPath, heightest_w)
        
        plyeast = PlyData.read(os.path.join(ply_path, ply_file_east))
        hist_e, heightest_e = gen_height_histogram_for_Roman(plyeast, scanDirection, out_dir, 'e', center_position)
        
        histPath = os.path.join(out_dir, 'hist_e.npy')
        np.save(histPath, hist_e)
        heightestPath = os.path.join(out_dir, 'top_e.npy')
        np.save(heightestPath, heightest_e)
        
        json_dst = os.path.join(out_dir, meta)
        shutil.copyfile(os.path.join(json_path, meta), json_dst)
    
    return

def get_height_result_for_roman(in_dir, sensor_d):
    
    if not os.path.isdir(in_dir):
        fail('Could not find input directory: ' + in_dir)
    
    plotNum = np.zeros((32,1))
    hist_data = np.zeros((32,400))
    top_data = np.zeros((32,1))
    
    # parse json file
    metafile, hist, top = find_result_files(in_dir, sensor_d)
    if metafile == [] or hist == [] :
        return plotNum, hist_data, top_data
    
    
    metadata = lower_keys(load_json(metafile))
    center_position = get_position(metadata)
    hist_data = np.load(hist, 'r')
    top_data = np.load(top, 'r')
        
    for i in range(0,32):
        #plotNum[i] = field_2_plot(center_position[0], i+1)
        plotNum[i] = field_2_plot_for_roman(center_position[0], i+1)
    
    return plotNum.astype('int'), hist_data, top_data

def create_normalization_hist(in_dir, out_dir):
    
    list_dirs = os.walk(in_dir)
    heightHist = np.zeros((1728, 400))
    plotScanCount = np.zeros((1728))
    
    for root, dirs, files in list_dirs:
        for d in dirs:
            full_path = os.path.join(in_dir, d)
            if not os.path.isdir(full_path):
                continue
            
            plotNum, hist, top = get_height_result_for_roman(full_path)
            if len(plotNum) < 32:
                continue
            
            for j in range(0,plotNum.size):
                heightHist[plotNum[j]-1] = heightHist[plotNum[j]-1]+hist[j]
                plotScanCount[plotNum[j]-1] = plotScanCount[plotNum[j]-1] + 1
                
    for i in range(0, 1728):
        if plotScanCount[i] != 0:
            heightHist[i] = heightHist[i]/plotScanCount[i]
    
    histfile = os.path.join(in_dir, 'heightHist.npy')
    np.save(histfile, heightHist)
    
    hist_out_file = os.path.join(in_dir, 'hist.txt')
    np.savetxt(hist_out_file, np.array(heightHist), delimiter="\t")
    
    
    return

def full_day_array_to_xlsx_for_roman(in_dir):
    
    list_dirs = os.walk(in_dir)
    heightHist = np.zeros((1728, 400))
    topMat = np.zeros((1728))
    
    for sensor_d in ['e','w']:
        for root, dirs, files in list_dirs:
            for d in dirs:
                full_path = os.path.join(in_dir, d)
                if not os.path.isdir(full_path):
                    continue
                
                plotNum, hist, top = get_height_result_for_roman(full_path, sensor_d)
                if len(plotNum) < 32:
                    continue
                
                for j in range(0,plotNum.size):
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

def field_2_plot(x, y):
    plotRange = round(x/4)
    col = y
    if plotRange % 2 == 1:
        col = 16-col+1
    plotNum = plotRange*16+col
    return plotNum

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

def field_2_plot_for_roman(x_position, y_row):

    xRange = 0
    count = 0
        
    for (xmin, xmax) in terra_common._x_range_s4:
        count = count + 1
        if (x_position > xmin) and (x_position <= xmax):
            xRange = 55 - count
            
            plotNum = convt.fieldPartition_to_plotNum_32(xRange, y_row)
            
            return plotNum
    
    return 0

def find_result_files(in_dir, sensor_d):
    
    metadata_suffix = os.path.join(in_dir, '*_metadata.json')
    metas = glob(metadata_suffix)
    if len(metas) == 0:
        fail('No metadata file found in input directory.')
        return [], [], []

    hist_file = os.path.join(in_dir, 'hist_'+sensor_d+'.npy')
    top_file = os.path.join(in_dir, 'top_'+sensor_d+'.npy')
    if os.path.isfile(hist_file) == False | os.path.isfile(top_file) == False:
        fail('No hist file or top file in input directory')
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


def gen_height_histogram(plydata, scanDirection):
    
    yStart = 0
    yRange = 16
    yOffset = 1362
    zOffset = 10
    zRange = [-2000, 2000]
    hist = np.zeros((yRange, (zRange[1]-zRange[0])/zOffset))
    heightest = np.zeros((yRange, 1))
    data = plydata.elements[0].data
    if data.size == 0:
        return hist, heightest
    
    if scanDirection == 'False':
        data["y"] = 0 - data["y"]
    
    for i in range(0, yRange):
        ymin = yStart + i*yOffset
        ymax = ymin + yOffset
        specifiedIndex = np.where((data["y"]>ymin) & (data["y"]<ymax))
        target = data[specifiedIndex]
        
        zloop = 0
        for z in range(zRange[0],zRange[1], zOffset):       
            zmin = z
            zmax = (z+zOffset)
            zIndex = np.where((target["z"]>zmin) & (target["z"]<zmax));
            num = len(zIndex[0])
            hist[i][zloop] = num
            zloop = zloop + 1
    
        zTop = 0;
        if len(specifiedIndex[0])!=0:
            zTop = target["z"].max()
        
        heightest[i] = zTop
    
    
    return hist, heightest

def offset_choice(scanDirection, sensor_d):
    
    if sensor_d == 'w':
        if scanDirection == 'True':
            ret = -3.45#-3.08
        else:
            ret = -25.711#-25.18
            
    if sensor_d == 'e':
        if scanDirection == 'True':
            ret = -3.45#-3.08
        else:
            ret = -25.711#-25.18
    
    return ret

def gen_height_histogram_for_Roman(plydata, scanDirection, out_dir, sensor_d, center_position):
    
    gantry_z_offset = 0.35
    zGround = (3.445 - center_position[2] + gantry_z_offset)*1000
    yRange = 32
    yShift = offset_choice(scanDirection, sensor_d)
    zOffset = 10
    zRange = [-2000, 2000]
    scaleParam = 1000
    hist = np.zeros((yRange, (zRange[1]-zRange[0])/zOffset))
    heightest = np.zeros((yRange, 1))
    data = plydata.elements[0].data
    
    if data.size == 0:
        return hist, heightest

    # TODO: Replace with getting plot bounding box instead of yRange = 32
    for i in range(0, yRange):
        ymin = (terra_common._y_row_s4[i][0]+yShift) * scaleParam
        ymax = (terra_common._y_row_s4[i][1]+yShift) * scaleParam
        specifiedIndex = np.where((data["y"]>ymin) & (data["y"]<ymax))
        target = data[specifiedIndex]
        
        for j in range(0, 400):
            zmin = zGround + j * zOffset
            zmax = zGround + (j+1) * zOffset
            zIndex = np.where((target["z"]>zmin) & (target["z"]<zmax));
            num = len(zIndex[0])
            hist[i][j] = num
        
        '''
        zloop = 0
        for z in range(zRange[0],zRange[1], zOffset):       
            zmin = z
            zmax = (z+zOffset)
            zIndex = np.where((target["z"]>zmin) & (target["z"]<zmax));
            num = len(zIndex[0])
            hist[i][zloop] = num
            zloop = zloop + 1
        '''
    
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

def gen_height_histogram_for_season_two(plydata, scanDirection, out_dir, sensor_d):
    
    yRange = 16
    yShift = -3
    zOffset = 10
    zRange = [-2000, 2000]
    scaleParam = 1000
    hist = np.zeros((yRange, (zRange[1]-zRange[0])/zOffset))
    heightest = np.zeros((yRange, 1))
    data = plydata.elements[0].data
    if data.size == 0:
        return hist, heightest
    
    if scanDirection == 'False':
        yShift = -25.1
    
    for i in range(0, yRange):
        ymin = (terra_common._y_row_s2[i*2+1][0]+yShift) * scaleParam
        ymax = (terra_common._y_row_s2[i*2][1]+yShift) * scaleParam
        specifiedIndex = np.where((data["y"]>ymin) & (data["y"]<ymax))
        target = data[specifiedIndex]
        '''
        zloop = 0
        for z in range(zRange[0],zRange[1], zOffset):       
            zmin = z
            zmax = (z+zOffset)
            zIndex = np.where((target["z"]>zmin) & (target["z"]<zmax));
            num = len(zIndex[0])
            hist[i][zloop] = num
            zloop = zloop + 1
    
        zTop = 0;
        if len(specifiedIndex[0])!=0:
            zTop = target["z"].max()
        
        heightest[i] = zTop
        '''
        out_basename = str(i)+'.png'
        out_file = os.path.join(out_dir, out_basename)
        save_points(target, out_file, i)
        
    
    return hist, heightest

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

if __name__ == "__main__":

    main()
