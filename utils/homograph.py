import cv2
import json
import os
import numpy as np

data_path = "/Users/Qianyu/Documents/Projects/data/Unleash"


def rotate_label(points, w_t, h_t):
    ''' Rotate the target view 90 counter-clockwise for better alignment'''
    new_ps = []
    for p in points:
        new_x = p[1]
        new_y = w_t - p[0]
        new_x *= w_t / h_t
        new_y *= h_t / w_t
        new_ps.append([new_x, new_y])
    return new_ps


def cal_homograph(data_path, mode='near'):
    src_labels = json.load(open(os.path.join(data_path, 'views', 'camera-view-point.json'), 'r'))
    tgt_labels = json.load(open(os.path.join(data_path, 'views', 'google-earth-point.json'), 'r'))
    s_img_path = os.path.join(data_path, 'pics', 'label_camera.png')
    t_img_path = os.path.join(data_path, 'pics', 'label_earth.png')
    
    s_img = cv2.imread(s_img_path)
    t_img = cv2.imread(t_img_path)
    h_s, w_s, _ = s_img.shape
    h_t, w_t, _ = t_img.shape
    r_h, r_w = h_t/h_s, w_t/w_s
    point_group = {'near': [0,1,2,3],
                    'mid': [2,3,4,5],
                    'far': [4,5,6,7]}
    s_points = []
    t_points = []
    for i in point_group[mode]:
        s_points.append(src_labels['shapes'][i]['points'][0])
        t_points.append(tgt_labels['shapes'][i]['points'][0])
    
    s_points = np.array(s_points)
    s_points[:, 0] *= r_w
    s_points[:, 1] *= r_h
    t_points = rotate_label(t_points, w_t, h_t)
    
    H, _ = cv2.findHomography(s_points, np.array(t_points))
    return H


def transform(bbox, Hs):

    # point to transform
    x0, y0, x1, y1 = bbox[0],bbox[1],bbox[2],bbox[3]
    x, y = (x0+x1)/2, y0*3/4+y1/4

    # select homograph
    anchor = [1120, 420]
    if x > anchor[0]:
        mode = 'near'
    elif y > anchor[1]:
        mode = 'mid'
    else:
        mode = 'far'

    H = Hs[mode]

    # transform
    p = np.array([x, y]).reshape(-1, 1, 2)
    p_t = cv2.perspectiveTransform(p, H)
    return p_t[0][0]
