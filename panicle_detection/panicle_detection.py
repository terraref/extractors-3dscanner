'''
Created on Aug 23, 2017

@author: zli
'''
import cv2
import sys, os, json
from glob import glob
import numpy as np
from PIL import Image
from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.utils.timer import Timer

from panicle_data_integration import full_day_output_integrate, offset_choice, crop_reflectance_image
from panicle_post_process import get_panicle_value, clustering_points, combine_close_boxes, mask_panicle_area, box_integrate
from plyfile import PlyData, PlyElement
import terra_common
    
convt = terra_common.CoordinateConverter()
SAVE_IMG = True

def main():
    
    in_dir = '/media/zli/data/Terra/sample_files/scanner3dTop/s2/'
    out_dir = '/media/zli/data/Terra/testing_output/panicle_output/s2_test_new'
    plot_dir = '/media/zli/data/Terra/testing_output/panicle_vis/s2_test'
    
    #plot_data_visualization(out_dir, plot_dir, 'cnt_mean')
    #for i in range(257, 288):
    #    split_images(out_dir, plot_dir, '/media/zli/data/Terra/testing_output/plot_img_test', i)
    
    
    list_dirs = os.listdir(in_dir)
    start_ind = 18
    ind = 0
    for d in list_dirs:
        i_path = os.path.join(in_dir, d)
        o_path = os.path.join(out_dir, d)
        ind += 1
        if ind < start_ind:
            continue
        full_day_gen_hist(i_path, i_path, o_path)
        if os.path.isdir(o_path):
            full_day_output_integrate(o_path, o_path)
    
    
    return


def full_day_gen_hist(ply_path, json_path, out_path):
    
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
        
    model_file_path = '/media/zli/data/VOC/models/saved_model_s2_3dPanicle/faster_rcnn_100000.h5'
    detector = load_model(model_file_path)
    
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
            
            try:
                process_one_directory(p_path, j_path, o_path, detector)
            except Exception as ex:
                terra_common.fail(p_path + str(ex))
    
    return

def process_one_directory(p_path, j_path, o_path, detector):
    
    if not os.path.isdir(o_path):
        os.mkdir(o_path)
        
    json_suffix = os.path.join(j_path, '*_metadata.json')    
    jsons = glob(json_suffix)
    if len(jsons) == 0:
        return
    
    g_img_suffix = os.path.join(j_path, '*west_0_g.png')    
    gimgs = glob(g_img_suffix)
    if len(gimgs) == 0:
        return
    
    p_img_suffix = os.path.join(j_path, '*west_0_p.png')    
    pimgs = glob(p_img_suffix)
    if len(pimgs) == 0:
        return
    
    ply_suffix = os.path.join(j_path, '*west_0.ply')    
    plys = glob(ply_suffix)
    if len(plys) == 0:
        return
    
    panicle_detection_from_laser(gimgs[0], plys[0], jsons[0], pimgs[0], o_path, detector)
    
    
    return


def panicle_detection_from_laser(g_img_path, ply_path, json_path, p_img_path, out_dir, detector):
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
        
    
    detected_img_dir = os.path.join(out_dir, 'mask')
    if not os.path.isdir(detected_img_dir):
        os.mkdir(detected_img_dir)
        
    plot_img_dir = os.path.join(out_dir, 'plotImg')
    if not os.path.isdir(plot_img_dir):
        os.mkdir(plot_img_dir)
        
    crop_img_dir = os.path.join(out_dir, 'cropImg')
    if not os.path.isdir(crop_img_dir):
        os.mkdir(crop_img_dir)
        
    histPlot_dir = os.path.join(out_dir, 'histPlot')
    if not os.path.isdir(histPlot_dir):
        os.mkdir(histPlot_dir)
    
    '''
    with open('/media/zli/data/Terra/testing_output/panicle_output/s2_new/2016-11-11/2016-11-11__03-20-53-757/plot_bound.txt', 'rb') as fp:
        plot_boundary_list = pickle.load(fp)
    
    
    points_out = os.path.join(out_dir, '3d_view')
    if not os.path.isdir(points_out):
        os.mkdir(points_out)
    ply_sub_dir = os.path.join(out_dir, 'subply')
    if not os.path.isdir(ply_sub_dir):
        os.mkdir(ply_sub_dir)
    '''
        
    metadata = terra_common.lower_keys(terra_common.load_json(json_path))
    dir_path = os.path.dirname(json_path)
    str_time = os.path.basename(dir_path)
    
    plotList = []
    countingList = []
    volumeList = []
    areaList = []
    boxList = []
    densityList = []
    xShift, yShift = offset_choice(metadata)
    
    g_img = cv2.imread(g_img_path)
    p_img = cv2.imread(p_img_path, -1)
    gIm = Image.open(g_img_path)
    
    ply_data = PlyData.read(ply_path)
    
    #src_data = PlyData.read(ply_path)
    
    x_inds, y_inds, img_vec = crop_reflectance_image(g_img)
    
    
    p_ind = 0
    
    saved_boxes = []
    for x_ind, y_ind, img in zip(x_inds, y_inds, img_vec):
        dets, scores, im2show = object_detection(detector, img)
        if len(dets)==0:
            continue
        cv2.imwrite(os.path.join(crop_img_dir, str_time + '_' + str(x_ind)+'_'+str(y_ind)+'.jpg'), im2show)
        new_boxes = remap_box_coordinates(x_ind, y_ind, dets)
        saved_boxes.append(new_boxes)
        
    # box integrate
    merged_boxes = box_integrate(saved_boxes)
    mask_vec = mask_panicle_area(p_img, merged_boxes, detected_img_dir, g_img, str_time)
    
    centerPoints = []
    maskedPoints = []
    for (box, mask_img) in zip(merged_boxes, mask_vec):
        localP = fetch_points_with_mask(gIm, ply_data, box, mask_img)
        if len(localP)<5:
            continue
        new_points, centerpoint = clustering_points(localP)
        centerPoints.append(centerpoint)
        maskedPoints.append(new_points)
        
    merged_boxes, centerPoints, maskedPoints = combine_close_boxes(merged_boxes, centerPoints, maskedPoints)
    #merged_boxes, centerPoints, maskedPoints, vec_scores, vec_nor_par = split_separated_panicles(merged_boxes, centerPoints, maskedPoints, histPlot_dir)
    #merged_boxes, centerPoints, maskedPoints = combine_close_boxes(merged_boxes, centerPoints, maskedPoints)
       
    for (box, new_points) in zip(merged_boxes, maskedPoints):
        # save feature data
        plotNum = points_2_plotNumber(new_points, xShift, yShift)
        if plotNum == -1:
            continue
            
        point_counting, volume_data, density, area_data = get_panicle_value(new_points)
        #area_data = abs(box[1]-box[3])*abs(box[0]-box[2])
        
        plotList.append(plotNum)
        countingList.append(point_counting)
        volumeList.append(volume_data)
        areaList.append(area_data)
        boxList.append(box)
        densityList.append(density)
            
        '''
        # data visulization
        out_png_file = os.path.join(points_out,str(p_ind)+'.png')
        save_sub_ply(new_points, src_data, os.path.join(ply_sub_dir,str(p_ind)+'.ply'))
        save_points(new_points, out_png_file, 5)
        '''
        p_ind += 1
            
    plot_boundary_list = sort_box_range(plotList, boxList)
    save_box_image(gIm, merged_boxes, out_dir, plot_img_dir, plot_boundary_list)
    save_data_to_file(plotList, volumeList, countingList, areaList, densityList, out_dir)
    
    return

def save_box_image(gImage, merged_boxes, out_dir, plot_img_dir, plot_boundary_list):
    
    im2show = np.copy(gImage)
    for box in merged_boxes :
        box = [int(i) for i in box]
        cv2.rectangle(im2show, (box[0],box[1]), (box[2],box[3]), (255, 205, 51), 3)
        #pt_ratio = (scores[0]/pars[0])/(scores[2]/pars[2])
        #cv2.putText(im2show, '%0.3f' % (pt_ratio), (box[0], box[1] + 20), cv2.FONT_HERSHEY_PLAIN,
        #                    2.0, (0, 0, 255), thickness=2)
        
    cv2.imwrite(os.path.join(out_dir, 'merged.jpg'), im2show)
    
    base_name = os.path.basename(out_dir)
    
    # save plot images
    plot_list = [i for i in range(257, 289)]
    ind = 0
    for plotNum in plot_list:
        y_range = plot_boundary_list[ind]
        ind += 1
        if y_range[0] == 0:
            continue
        
        crop_img = im2show[int(y_range[0]):int(y_range[1]), :]
        out_img_path = os.path.join(plot_img_dir, base_name+'_'+str(plotNum)+'.jpg')
        cv2.imwrite(out_img_path, crop_img)
    
    return

def sort_box_range(plotList, boxList):
    
    merge_box_list = [[] for i in range(32)]
    plot_boundary_list = []
    
    ind = 0
    for box in boxList:
        plotNum = plotList[ind]
        ind += 1
        
        list_ind = plotNum - 257
        merge_box_list[list_ind].append(box)
        
    for boxes in merge_box_list:
        ymin = 10000000
        ymax = 0
        for box in boxes:
            if ymin > box[1]:
                ymin = box[1]
            if ymax < box[3]:
                ymax = box[3]
        
        if ymin > ymax:
            plot_boundary_list.append([0,0])
        else:
            plot_boundary_list.append([ymin, ymax])
    
    return plot_boundary_list


def point_2_plotNum(point, xShift, yShift):
    
    x = point[0] + xShift
    y = point[1] + yShift
    
    count = 0
    
    plot_row = 0
    plot_col = 0
    for (ymin, ymax) in terra_common._y_row_s2:
        count = count + 1
        if y > ymin:
            plot_col = count
            break
        
    count = 0
    for (xmin, xmax) in terra_common._x_range_s2:
        count = count + 1
        if (x > xmin) and (x <= xmax):
            plot_row = 55 - count
            
            plotNum = convt.fieldPartition_to_plotNum_32(plot_row, plot_col)
            break
    
    return plotNum




def points_2_plotNumber(points, xShift, yShift):
    
    X = points["x"]
    Y = points["y"]
    point1 = (X.min()/1000, Y.min()/1000)
    point2 = (X.max()/1000, Y.max()/1000)
    
    plotNum1 = point_2_plotNum(point1, xShift, yShift)
    plotNum2 = point_2_plotNum(point2, xShift, yShift)
    
    if plotNum1 == plotNum2:
        return plotNum1
    else:
        return -1



def save_data_to_file(plotNum, volume_data, counting_data, areaList, densityList, out_dir):
    
    volumeFile = os.path.join(out_dir, 'volume.txt')
    volume_handle = open(volumeFile, 'w')
    for (plot, vol) in zip(plotNum, volume_data):
        volume_handle.write("%d,%f\n"%(plot, vol))
        
    volume_handle.close()
    
    countFile = os.path.join(out_dir, 'counting.txt')
    count_handle = open(countFile, 'w')
    for (plot, counts) in zip(plotNum, counting_data):
        count_handle.write("%d,%d\n"%(plot, counts))
        
    count_handle.close()
    
    areaFile = os.path.join(out_dir, 'area.txt')
    area_handle = open(areaFile, 'w')
    for (plot, areas) in zip(plotNum, areaList):
        area_handle.write("%d,%f\n"%(plot, areas))
        
    area_handle.close()
    
    densityFile = os.path.join(out_dir, 'density.txt')
    density_handle = open(densityFile, 'w')
    for (plot, densitys) in zip(plotNum, densityList):
        density_handle.write("%d,%f\n"%(plot, densitys))
        
    density_handle.close()
    
    return

def load_model(model_file_path):
    
    detector = FasterRCNN()
    network.load_net(model_file_path, detector)
    detector.cuda()
    detector.eval()
    print('load model successfully!')
    
    return detector

def object_detection(detector, image, score_threshold=0.9):
    
    dets, scores, classes = detector.detect(image, score_threshold)
        
    im2show = np.copy(image)
    for i, det in enumerate(dets):
        det = tuple(int(x) for x in det)
        cv2.rectangle(im2show, det[0:2], det[2:4], (255, 205, 51), 2)
        cv2.putText(im2show, '%s: %.3f-%d' % (classes[i], scores[i], i), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 255), thickness=2)
    
    return dets, scores, im2show

def remap_box_coordinates(fileIndX, fileIndY, boxes, out_img_size=(1024, 1024)):
    
    x_offset = fileIndX*out_img_size[0]
    y_offset = fileIndY*out_img_size[1]
    
    new_boxes = []
    
    for box in boxes:
        new_box = [box[0]+y_offset, box[1]+x_offset, box[2]+y_offset, box[3]+x_offset]
        new_boxes.append(new_box)
    
    return new_boxes


def fetch_points(gImg, ply_data, box):
    
    [gWid, gHei] = gImg.size
    
    pix = np.array(gImg).ravel()
    
    gIndex = (np.where(pix>32))
    nonZeroSize = gIndex[0].size
    
    pointSize = ply_data.elements[0].count
    
    if nonZeroSize != pointSize:
        return []
    
    gIndexImage = np.zeros(gWid*gHei)
    
    gIndexImage[gIndex[0]] = np.arange(1,pointSize+1)
    
    gIndexImage_ = np.reshape(gIndexImage, (-1, gWid))
    
    plyIndices = gIndexImage_[box[1]:box[3], box[0]:box[2]]
    plyIndices = plyIndices.ravel()
    plyIndices_ = np.where(plyIndices>0)
    localIndex = plyIndices[plyIndices_[0]].astype('int64')
    localP = ply_data.elements[0].data[localIndex-1]
    
    return localP

def fetch_points_with_mask(gImg, ply_data, box, mask_img):
    
    mask_img = mask_img > 1
    
    [gWid, gHei] = gImg.size
    
    pix = np.array(gImg).ravel()
    
    gIndex = (np.where(pix>32))
    nonZeroSize = gIndex[0].size
    
    pointSize = ply_data.elements[0].count
    
    if nonZeroSize != pointSize:
        return []
    
    gIndexImage = np.zeros(gWid*gHei)
    
    gIndexImage[gIndex[0]] = np.arange(1,pointSize+1)
    
    gIndexImage_ = np.reshape(gIndexImage, (-1, gWid))
    
    plyIndices = gIndexImage_[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
    plyIndices = plyIndices[mask_img]
    
    plyIndices = plyIndices.ravel()
    plyIndices_ = np.where(plyIndices>0)
    localIndex = plyIndices[plyIndices_[0]].astype('int64')
    localP = ply_data.elements[0].data[localIndex-1]
    
    return localP
        

if __name__ == "__main__":

    main()