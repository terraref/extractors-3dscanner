'''
Created on Mar 28, 2018

@author: zli
'''
import cv2
import os, math
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import sklearn.decomposition as deco
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, DBSCAN
from scipy.ndimage.filters import convolve

PARAM_PANICLE_RANGE = 100
ROUGHNESS_PARA = 120
SAVE_IMG = True

# object detection based on a cropped reflected image, we need to merge referencing box close to the edge
def box_integrate(all_boxes):
    
    total_image_number = len(all_boxes)
    middle_x_coord = 1024
    close_pix_to_bound = 2
    right_side_boundary = 10
    left_side_boundary = 10
    for i in range(total_image_number):
        # left side image
        if i % 2 == 0:
            target_box_list = all_boxes[i]
            for item in target_box_list:
                if item[0] < right_side_boundary:
                    target_box_list.remove(item)
                    
            right_list_index = i+1
            if right_list_index >= total_image_number:
                continue
            right_box_list = all_boxes[right_list_index]
            # check right boundary
            for box in target_box_list:
                if (middle_x_coord - box[2]) > close_pix_to_bound:
                    continue
                # find the pair box in right box list
                for j in range(len(right_box_list)):
                    if (right_box_list[j][0] - middle_x_coord) > close_pix_to_bound:
                        continue
                    if check_if_overlap(box[1], box[3], right_box_list[j][1], right_box_list[j][3]):
                        box[2] = right_box_list[j][2]
                        box[3] = max(box[3],right_box_list[j][3])
                        del right_box_list[j]
                        break
    
            
            if i+2 >= total_image_number:
                continue
            #print(len(all_boxes), i+2)
            bottom_box_list = all_boxes[i+2]
            # check bottom boundary
            for box in target_box_list:
                bottom_y_coord = (i/2+1)*1024
                if (bottom_y_coord - box[3]) > close_pix_to_bound:
                    continue
                # find the pair box in bottom box list
                for j in range(len(bottom_box_list)):
                    if (bottom_box_list[j][1] - bottom_y_coord) > close_pix_to_bound:
                        continue
                    
                    if check_if_overlap(box[0], box[2], bottom_box_list[j][0], bottom_box_list[j][2]):
                        box[2] = max(box[2],bottom_box_list[j][2])
                        box[3] = bottom_box_list[j][3]
                        del bottom_box_list[j]
                        break
            
        # right side image    
        else:
            target_box_list = all_boxes[i]
            for item in target_box_list:
                if (2048 - item[2]) < left_side_boundary:
                    target_box_list.remove(item)
                    
            if i+2 >= total_image_number:
                continue
            bottom_box_list = all_boxes[i+2]
            # check bottom boundary
            for box in target_box_list:
                bottom_y_coord = (i/2+1)*1024
                if (bottom_y_coord - box[3]) > close_pix_to_bound:
                    continue
                # find the pair box in bottom box list
                for j in range(len(bottom_box_list)):
                    if (bottom_box_list[j][1] - bottom_y_coord) > close_pix_to_bound:
                        continue
                    
                    if check_if_overlap(box[0], box[2], bottom_box_list[j][0], bottom_box_list[j][2]):
                        box[2] = max(box[2],bottom_box_list[j][2])
                        box[3] = bottom_box_list[j][3]
                        del bottom_box_list[j]
                        break
    
    new_boxes = []
    for box_list in all_boxes:
        for box in box_list:
            new_boxes.append(box)
    
    return new_boxes

# function of checking rectangle overlapping
def check_if_overlap(a1, a2, b1, b2):
    
    if max(a1, b1) < min(a2, b2):
        return True
    else:
        return False

# try to throw out points other than panicles, such stem, leaf or other stuff, using roughness information and depth information from pImage
def mask_panicle_area(pImage, merged_boxes, out_dir, g_img, str_time):
    
    if SAVE_IMG:
        save_dir = os.path.join(out_dir, 'im2show')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
    
    im2clip = np.copy(pImage)
    ind = 0
    mask_vec = []
    for box in merged_boxes:
        ind += 1
        box = [int(i) for i in box]
        depth_img = im2clip[box[1]:box[3], box[0]:box[2]]
        gray_img = g_img[box[1]:box[3], box[0]:box[2]]
        mask_t, mask_img = roughness_2_mask(depth_img, ind, out_dir)
        
        new_mask = panicle_mask_post_process(gray_img, depth_img, mask_img, ind, out_dir)
        
        mask_vec.append(new_mask)
        
        if SAVE_IMG:
            save_path = os.path.join(save_dir, str_time + '_' + str(ind)+'.png')
            cv2.imwrite(save_path, gray_img)
    
    return mask_vec

def panicle_mask_post_process(g_im, p_im, m_im, ind, mask_dir):
    
    GRAY_T = 0
    DEPTH_T = 0
    g_im = cv2.cvtColor(g_im, cv2.COLOR_BGR2GRAY)
    
    # grayscale threshold
    g_mask = cv2.bitwise_and(g_im,m_im)
    mean_gray_scale = np.mean(g_mask)
    mask1 = np.zeros_like(m_im)
    mask1_flag = g_mask > (mean_gray_scale - GRAY_T)
    mask1[mask1_flag] = 255
    
    #ret, mask1 = cv2.threshold(g_im, mean_gray_scale - GRAY_T, 255, cv2.THRESH_BINARY)
    #cv2.imshow('1', mask1)
    #cv2.waitKey()
    
    # depth threshold
    src_img = p_im.astype('int32')
    mean_depth_threshold = np.mean(src_img)
    mask2 = np.zeros_like(m_im)
    mask2_flag = src_img > (mean_depth_threshold - DEPTH_T)
    mask2[mask2_flag] = 255
    
    mask_img = cv2.bitwise_and(mask1,mask2)
    
    # remove small spot 
    
    im2, contours, hierarchy = cv2.findContours(mask_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            
    for i in range(len(contours)):
        if len(contours[i]) < 2:
            mask_img[contours[i][0][0][1], contours[i][0][0][0]] = 0
    
    #kernel = np.ones((3,3), np.uint8)
    #rel = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel)
    
    if SAVE_IMG:
        cv2.imwrite(os.path.join(mask_dir,str(ind)+'.png'), mask_img)
    
    return mask_img

def roughness_2_mask(src_img, ind, mask_dir, thre = 5, kernelSize = 3):
    
    src_img = src_img.astype('int32')
    
    fK = np.asarray(
        [[0, -1, 0],
         [-1, 4, -1],
         [0, -1, 0]]) / 4.0
         
    conv_img = convolve(src_img, fK)
    conv_img = np.absolute(conv_img)
    remain_mask = conv_img > ROUGHNESS_PARA
    conv_img[remain_mask] = 0
    
    conv_img = cv2.blur(conv_img,(kernelSize,kernelSize))
    dst_img = conv_img > thre
    
    mask_img = np.zeros_like(dst_img, dtype=np.uint8)
    mask_img[dst_img] = 255
    
    return dst_img, mask_img

# try to combine close panicle boxes
def combine_close_boxes(merged_boxes, centerPoints, maskedPoints):
    
    new_boxes = []
    new_centers = []
    new_points = []
    
    deled_ind = []
    
    for i in range(len(merged_boxes)):
        if i in deled_ind:
            continue
        curr_box = merged_boxes[i]
        curr_center = centerPoints[i]
        curr_points = maskedPoints[i]
        for j in range(len(merged_boxes)):
            if j in deled_ind:
                continue
            if i == j:
                continue
            
            pair_box = merged_boxes[j]
            pair_center = centerPoints[j]
            pair_points = maskedPoints[j]
            # if box overlapping, return 0, if two boxes too far away, return 2
            retFlag = compare_close_boxes(curr_box, pair_box)
            if retFlag == 0:
                # box overlapping test, if the overlapping ratio higher than 40%, combine two boxes
                overlapping_ratio = compute_box_overlap(curr_box, pair_box)
                if overlapping_ratio > 0.4:
                    deled_ind.append(j)
                    curr_box = combine_boxes(curr_box, pair_box)
                    curr_points = np.concatenate((curr_points, pair_points), axis=0)
                    curr_center = get_points_center(curr_points)
                    break
                
            if retFlag != 2:
                # check if 3d center is close
                if compare_close_center(curr_center, pair_center):
                    p1, d1 = nearest_distance(curr_points, pair_center)
                    p2, d2 = nearest_distance(pair_points, curr_center)
                    
                    dist = np.linalg.norm(p1-p2)
                    if dist < PARAM_PANICLE_RANGE / 3:
                        
                        add_box = combine_boxes(curr_box, pair_box)
                        wh_ratio = (add_box[2]-add_box[0])/(add_box[3]-add_box[1])
                        if wh_ratio>1.6:
                            continue
                        
                        added_w_ratio = ((add_box[2]-add_box[0])/float(max((curr_box[2]-curr_box[0]), (pair_box[2]-pair_box[0]))))
                        added_h_ratio = ((add_box[3]-add_box[1])/float(max((curr_box[3]-curr_box[1]), (pair_box[3]-pair_box[1]))))
                        if added_w_ratio/added_h_ratio > 10:
                            continue
                        else:
                            deled_ind.append(j)
                            curr_box = add_box
                            curr_points = np.concatenate((curr_points, pair_points), axis=0)
                            curr_center = get_points_center(curr_points)
                            break
        new_boxes.append(curr_box)
        new_centers.append(curr_center)
        new_points.append(curr_points)
                    
            
    return new_boxes, new_centers, new_points

def dist_from_point_to_line(point, p1, p2):
    
    ap = point-p1
    ab = p2-p1
    tarP = p1 + np.dot(ap,ab)/np.dot(ab,ab) * ab
    dist = np.linalg.norm(tarP-point)
    
    return dist**2

def compute_box_overlap(box, box_pair):
    
    si = max(0, min(box[2], box_pair[2])-max(box[0], box_pair[0]))*max(0, min(box[3], box_pair[3])-max(box[1], box_pair[1]))
    
    sa = (box[2]-box[0])*(box[3]-box[1])
    sb = (box_pair[2]-box_pair[0])*(box_pair[3]-box_pair[1])
    if sa < sb:
        ret = si/sa
    else:
        ret = si/sb
    
    return max(0, ret)

def nearest_distance(in_points, center_point):
    
    X = in_points["x"]
    Y = in_points["y"]
    Z = in_points["z"]
    points = np.concatenate((X,Y,Z),axis=0)
    points = np.reshape(points, (3,-1))
    points = points.T
    
    tree = KDTree(points)
    dist, ndx = tree.query(center_point, k=1)
    
    return points[ndx], dist

def get_points_center(points):
    
    x = points["x"].mean()
    y = points["y"].mean()
    z = points["z"].mean()
    
    return [x,y,z]

def combine_boxes(box, pair_box):
    
    newbox = [[] for i in range(4)]
    
    newbox[0] = min(box[0], pair_box[0])
    newbox[1] = min(box[1], pair_box[1])
    newbox[2] = max(box[2], pair_box[2])
    newbox[3] = max(box[3], pair_box[3])
    
    return newbox

def compare_close_boxes(box_a, box_b):
    
    center_a = [(box_a[0]+box_a[2])/2, (box_a[1]+box_a[3])/2]
    center_b = [(box_b[0]+box_b[2])/2, (box_b[1]+box_b[3])/2]
    
    x_length = (box_a[2]-box_a[0]+box_b[2]-box_b[0])/2
    y_length = (box_a[3]-box_a[1]+box_b[3]-box_b[1])/2
    
    minx = max(box_a[0], box_b[0])
    miny = max(box_a[1], box_b[1])
    maxx = min(box_a[2], box_b[2])
    maxy = min(box_a[3], box_b[3])
    
    if minx < maxx and miny < maxy:
        return 0
    
    dist_range = math.sqrt(x_length**2 + y_length**2)
    dist = math.sqrt(abs(center_a[0]-center_b[0])**2+abs(center_a[1]-center_b[1])**2)
    
    if dist < dist_range:
        return 1
    
    return 2

def compare_close_center(center_a, center_b):
    
    dist_range_3d = PARAM_PANICLE_RANGE
    
    dist = np.linalg.norm(center_a-center_b)
    
    if dist < dist_range_3d:
        return True
    
    return False

def clustering_points(in_points):
    
    X = in_points["x"]
    Y = in_points["y"]
    Z = in_points["z"]
    points = np.concatenate((X,Y,Z),axis=0)
    points = np.reshape(points, (3,-1))
    points = points.T
    
    
    km = KMeans(n_clusters=5).fit(points)
    labels = km.labels_
    
    # counting points number of each label
    ind_vec = []
    ind_count = np.zeros(5)
    for i in range(5):
        point_inds = np.where(labels==i)
        ind_vec.append(point_inds[0])
        ind_count[i] = (len(point_inds[0]))
    
    # distance from max points center
    max_ind = np.argmax(ind_count)
    max_center = km.cluster_centers_[max_ind]
    include_label = []
    
    # center threshold
    for i in range(5):
        if not calc_distance(max_center, km.cluster_centers_[i]):
            continue
        
        include_label.append(i)
        
    # new point set
    out_points = np.zeros(0)
    for i in range(len(include_label)):
        select_points = in_points[ind_vec[include_label[i]]]
        if i == 0:
            out_points = select_points
        else:
            out_points = np.concatenate((out_points, select_points), axis=0)
            
    return out_points, max_center

def calc_distance(point1, point2):
    
    dist = np.linalg.norm(point1-point2)
    
    if dist < PARAM_PANICLE_RANGE:
        return True
    else:
        return False

def convexHull(in_points):
    
    X = in_points["x"]
    Y = in_points["y"]
    Z = in_points["z"]
    points = np.concatenate((X,Y,Z),axis=0)
    points = np.reshape(points, (3,-1))
    points = points.T
    hull = ConvexHull(points)
    
    return hull

def get_panicle_value(points):
    
    point_counting = points.size
    '''
    X = points["x"]
    Y = points["y"]
    Z = points["z"]
    
    x_range = X.max() - X.min()
    y_range = Y.max() - Y.min()
    z_range = Z.max() - Z.min()
    
    volume_data = x_range*y_range*z_range
    '''
    hull = convexHull(points)
    
    density = point_counting / hull.volume
    
    return point_counting, hull.volume, density, hull.area

# panicle separating, on going...
def split_separated_panicles(merged_boxes, centerPoints, maskedPoints, out_dir):
    
    new_boxes = []
    new_centers = []
    new_points = []
    
    deleted_boxes = []
    
    vec_scores = []
    vec_nor_par = []
    
    for i in range(len(merged_boxes)):
        curr_box = merged_boxes[i]
        curr_center = centerPoints[i]
        curr_points = maskedPoints[i]
        
        hist, bin_edges = dist_histogram(curr_points, curr_center)
        
        plot_histogram(hist, i, out_dir)
        
        dist_list, max_dist_list = moment_of_inertia_pca_analysis(curr_points)
        vec_scores.append(dist_list)
        vec_nor_par.append(max_dist_list)
        '''
            deleted_boxes.append(curr_box)
        else:
            new_boxes.append(curr_box)
            new_centers.append(curr_center)
            new_points.append(curr_points)
        '''
    
    return merged_boxes, centerPoints, maskedPoints, vec_scores, vec_nor_par

def dist_histogram(in_points, centerPoints):
    
    X = in_points["x"]
    Y = in_points["y"]
    Z = in_points["z"]
    points = np.concatenate((X,Y,Z),axis=0)
    points = np.reshape(points, (3,-1))
    points = points.T
    
    np_test = np.empty((1,3))
    np_test[:,:] = centerPoints
    np_dist = cdist(points, np_test)
    
    hist, bin_edges = np.histogram(np_dist, bins=100)
    
    return hist, bin_edges

def points_pca_analysis(in_points):
    
    X = in_points["x"]
    Y = in_points["y"]
    Z = in_points["z"]
    points = np.concatenate((X,Y,Z),axis=0)
    points = np.reshape(points, (3,-1))
    points = points.T
    
    
    km = KMeans(n_clusters=2).fit(points)
    labels = km.labels_
    
    point_inds = np.where(labels==0)
    points1 = in_points[point_inds[0]]
    point_inds = np.where(labels==1)
    points2 = in_points[point_inds[0]]
    
    pc1 = get_pca_result(points1)
    pc2 = get_pca_result(points2)
    
    plot_pca_direction(pc1, pc2, points1, points2)
    
    return

def plot_pca_direction(pc1, pc2, points1, points2):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points1["x"], points1["y"], points1["z"], s=1, color='blue')
    
    ax.scatter(points2["x"], points2["y"], points2["z"], s=1, color='red')
    
    ax.quiver(0,0,0, pc1.components_[0], pc1.components_[1], pc1.components_[2], length=20)
    
    ax.quiver(0,0,0, pc2.components_[0], pc2.components_[1], pc2.components_[2], length=20)
    
    plt.draw()
    plt.close()
    
    return

def get_pca_result(in_points):
    
    X = in_points["x"]
    Y = in_points["y"]
    Z = in_points["z"]
    points = np.concatenate((X,Y,Z),axis=0)
    points = np.reshape(points, (3,-1))
    points = points.T
    
    x = (points - np.mean(points, 0)) / np.std(points, 0)
    pca = deco.PCA(3)
    pca.fit(x).transform(x)
    
    return pca

def moment_of_inertia_pca_analysis(in_points):
    
    pc = get_pca_result(in_points)
    
    dist_list = []
    max_dist_list = []
    
    for i in range(3):
        normal_vec = pc.components_[i]
        
        dist, max_dist = moment_of_vector(in_points, normal_vec)
        
        dist_list.append(dist)
        max_dist_list.append(max_dist)
    
    
    return dist_list, max_dist_list

def moment_of_vector(in_points, normal_vec):
    
    X = in_points["x"]
    Y = in_points["y"]
    Z = in_points["z"]
    points = np.concatenate((X,Y,Z),axis=0)
    points = np.reshape(points, (3,-1))
    points = points.T
    
    a = np.mean(points, axis=0)
    b = a+normal_vec
    
    ap = points-a
    ab = b-a
    c = np.dot(ap,ab)/np.dot(ab,ab)
    c = np.concatenate((c, c, c), axis=0)
    c = np.reshape(c, (3,-1))
    c = c.T
    d = c*ab
    tarP = a + d
    
    dist = 0
    max_dist = 0
    for i in range(tarP.shape[0]):
        dist_s = np.linalg.norm(tarP[i]-points[i]) ** 2
        dist += dist_s
        if dist_s > max_dist:
            max_dist = dist_s
        
    dist /= tarP.shape[0]
    
    return dist, max_dist

def moment_of_inertia_analysis(in_points):
    
    X = in_points["x"]
    Y = in_points["y"]
    Z = in_points["z"]
    points = np.concatenate((X,Y,Z),axis=0)
    points = np.reshape(points, (3,-1))
    points = points.T
    
    U, s, V = np.linalg.svd(points, full_matrices=False)
    
    normals = V[2,:]
    normals = np.sign(normals[2])*normals
    
    a = np.mean(points, axis=0)
    b = a+normals
    
    ap = points-a
    ab = b-a
    c = np.dot(ap,ab)/np.dot(ab,ab)
    c = np.concatenate((c, c, c), axis=0)
    c = np.reshape(c, (3,-1))
    c = c.T
    d = c*ab
    tarP = a + d
    
    dist = 0
    for i in range(tarP.shape[0]):
        dist += np.linalg.norm(tarP[i]-points[i]) ** 2
        
    dist /= tarP.shape[0]
    
    return dist

def plot_histogram(hist, ind, out_dir):
    
    fig, ax = plt.subplots()
    x = np.arange(hist.size)
    
    for i in range(hist.size):
        plt.bar(x[i], hist[i], width=0.8)
        
    out_file = os.path.join(out_dir, '%d.png'%(ind))
    plt.savefig(out_file)
    plt.close()
    return

def points_clustering(points_a, points_b):
    
    in_points = np.concatenate((points_a, points_b), axis=0)
    
    X = in_points["x"]
    Y = in_points["y"]
    Z = in_points["z"]
    points = np.concatenate((X,Y,Z),axis=0)
    points = np.reshape(points, (3,-1))
    points = points.T
    
    sub_points = points[1::10]
    db = DBSCAN(eps=40, min_samples=10).fit(sub_points)
    
    if db.labels_.max() > 0:
        return False
    
    return True




