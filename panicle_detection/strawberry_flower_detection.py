'''
Created on Sep 20, 2017

@author: zli
'''

import cv2
import sys, os, json, argparse
from glob import glob
import numpy as np
from PIL import Image
from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.utils.timer import Timer

from sklearn.cluster import AffinityPropagation, KMeans
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from datetime import date, timedelta

def options():
    
    parser = argparse.ArgumentParser(description='Phenode Strawberry flower detection',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-i", "--input_path", help="full path of input image")
    parser.add_argument("-o", "--out_dir", help="output directory")
    parser.add_argument("-m", "--module_path", help="R-CNN trained module path")

    args = parser.parse_args()

    return args


def main():
    
    args = options()
    
    detector = load_model(args.module_path)
    
    list_dirs = os.walk(args.input_path)
    for root, dirs, files in list_dirs:
        for f in files:
            file_path = os.path.join(args.input_path, f)
    
            flower_counting(file_path, args.out_dir, detector)
    
    return

def flower_counting(img_path, out_dir, detector):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    # load image, split into target resolution
    src_img = cv2.imread(img_path)
    x_inds, y_inds, img_vec = crop_image(src_img)
    
    # R-CNN detection
    saved_boxes = []
    for x_ind, y_ind, img in zip(x_inds, y_inds, img_vec):
        dets, scores, im2show = object_detection(detector, img, 0.85)
        print(x_ind, y_ind)
        if len(dets)==0:
            continue
        #cv2.imwrite(os.path.join(out_dir, str(x_ind)+'_'+str(y_ind)+'.jpg'), im2show)
        new_boxes = remap_box_coordinates(x_ind, y_ind, dets)
        for box in new_boxes:
            saved_boxes.append(box)
    
    saved_boxes = box_integrate(saved_boxes)
    saved_boxes = box_integrate(saved_boxes)
    
    base_name = os.path.basename(img_path)[:-4]
    # boxes post prosessing
    save_box_image(src_img, saved_boxes, out_dir, base_name)
    
    # counting and total area
    save_data_to_file(saved_boxes, out_dir, base_name)
    
    return

def box_integrate(all_boxes):
    
    new_boxes = []
    
    deled_ind = []
    
    for i in range(len(all_boxes)):
        if i in deled_ind:
            continue
        curr_box = all_boxes[i]
        for j in range(len(all_boxes)):
            if j in deled_ind:
                continue
            if i == j:
                continue
            
            pair_box = all_boxes[j]
            overlapping_ratio = compute_box_overlap(curr_box, pair_box)
            if overlapping_ratio > 0.3:
                deled_ind.append(j)
                add_box = combine_boxes(curr_box, pair_box)
                if max(add_box[2]-add_box[0], add_box[3]-add_box[1])>80:
                    continue
                else:
                    curr_box = add_box
                break
            
        new_boxes.append(curr_box)
    
    return new_boxes

def combine_boxes(box, pair_box):
    
    box[0] = min(box[0], pair_box[0])
    box[1] = min(box[1], pair_box[1])
    box[2] = max(box[2], pair_box[2])
    box[3] = max(box[3], pair_box[3])
    
    return box

def compute_box_overlap(box, box_pair):
    
    si = max(0, min(box[2], box_pair[2])-max(box[0], box_pair[0]))*max(0, min(box[3], box_pair[3])-max(box[1], box_pair[1]))
    
    sa = (box[2]-box[0])*(box[3]-box[1])
    sb = (box_pair[2]-box_pair[0])*(box_pair[3]-box_pair[1])
    if sa < sb:
        ret = si/sa
    else:
        ret = si/sb
    
    return max(0, ret)

def save_data_to_file(boxes, out_dir, base_name):
    
    counting = len(boxes)
    
    area = 0
    for box in boxes:
        area += (box[2]-box[0])*(box[3]-box[1])
        
    out_file_path = os.path.join(out_dir, base_name+'.csv')
    csv_handle = open(out_file_path, 'w')
    csv_handle.write("total flower:%d\n"%(counting))
    csv_handle.write("total flower area:%d\n"%(area))
    
    csv_handle.close()
    
    return

def save_box_image(gImage, merged_boxes, out_dir, base_name):
    
    im2show = np.copy(gImage)
    for box in merged_boxes:
        box = [int(i) for i in box]
        cv2.rectangle(im2show, (box[0],box[1]), (box[2],box[3]), (255, 205, 51), 3)
        
    cv2.imwrite(os.path.join(out_dir, 'labeled_'+base_name+'.jpg'), im2show)
    
    return

def crop_image(img, out_img_size=(600,600)):
    
    width, height, channels = img.shape
    
    i_wid_max = int(round(width/out_img_size[0]))+1
    i_hei_max = int(round(height/out_img_size[1]))+1
    
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

def object_detection(detector, image, score_threshold=0.7):
    
    dets, scores, classes = detector.detect(image, score_threshold)
        
    im2show = np.copy(image)
    for i, det in enumerate(dets):
        det = tuple(int(x) for x in det)
        cv2.rectangle(im2show, det[0:2], det[2:4], (255, 205, 51), 2)
        cv2.putText(im2show, '%s: %.3f-%d' % (classes[i], scores[i], i), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 255), thickness=1)
    
    return dets, scores, im2show

def remap_box_coordinates(fileIndX, fileIndY, boxes, out_img_size=(600, 600)):
    
    x_offset = fileIndX*out_img_size[0]
    y_offset = fileIndY*out_img_size[1]
    
    new_boxes = []
    
    for box in boxes:
        new_box = [box[0]+y_offset, box[1]+x_offset, box[2]+y_offset, box[3]+x_offset]
        new_boxes.append(new_box)
        
    #no_overlap_boxes = non_max_suppression_fast(new_boxes, 0.1)
    
    return new_boxes

def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float") 
        
    # initialize the list of picked indexes   
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
      
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
        np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

def load_model(model_file_path):
    
    detector = FasterRCNN()
    network.load_net(model_file_path, detector)
    detector.cuda()
    detector.eval()
    print('load model successfully!')
    
    return detector

if __name__ == '__main__':
    
    main()