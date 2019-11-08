'''
Created on Nov 20, 2018

@author: zli
'''
from laspy.file import File
import numpy as np

HIST_BIN_NUM = 500


def las_to_height_distribution(in_file, out_file):
    

    zRange = [0, 5000] # depth scope of laser scanner, unit in mm
    zOffset = 10  # bin width of histogram, 10 mm for each bin
    scaleParam = 1000 # min unit in las might be 0.001 mm
    
    height_hist = np.zeros((zRange[1]-zRange[0])/zOffset)
    
    las_handle = File(in_file)
    
    zData = las_handle.Z
    
    if (zData.size) == 0:
        return height_hist

    with open("hist.csv", 'w') as out:
        out.write("bin,range,pts\n")

        for i in range(0, HIST_BIN_NUM):
            zmin = i * zOffset * scaleParam
            zmax = (i+1) * zOffset * scaleParam
            if i == 0:
                zIndex = np.where(zData<zmax)
            elif i == HIST_BIN_NUM-1:
                zIndex = np.where(zData>zmin)
            else:
                zIndex = np.where((zData>zmin) & (zData<zmax))
            num = len(zIndex[0])
            height_hist[i] = num

            out.write("%s,%s,%s\n" % (i, "%s-%s" % (zmin, zmax), num))

    return height_hist

def las_to_height_distribution_optimize(in_file):
    
    
    zOffset = 10  # bin width of histogram, 10 mm for each bin
    scaleParam = 1000 # min unit in las might be 0.001 mm
    
    height_hist = np.zeros(HIST_BIN_NUM)
    
    las_handle = File(in_file)
    
    zData = las_handle.Z
    
    if (zData.size) == 0:
        return height_hist
    
    
    zVal = zData/(zOffset*scaleParam)
        
    height_hist = np.histogram(zVal, bins=range(-1, HIST_BIN_NUM), normed=False)[0]
    
    return height_hist

def export_to_csv(out_file_path, hist_data):
    
    np.savetxt(out_file_path, hist_data, delimiter=',')
    
    return
