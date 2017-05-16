#/usr/bin/python
# --------------------------------------------------------
# floor_recog
# Written by Sai Prabhakar
# CMU-RI Masters
# --------------------------------------------------------

import sys
import os
os.environ['GLOG_minloglevel'] = '3'

import argparse
import numpy as np
import cv2
import datetime

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

import pickle
from ipdb import set_trace as debug
import time as t
import modifiedSiamese.helpers2 as h2
import copy
'''
Script to generate explanation for scene detection
given object detection and important region heat map.
'''


def load_image_name(fileName, class_adju):
    '''
    load file names
    '''
    with open(fileName) as f:
        lines = [line.rstrip('\n') for line in f]
    imlist = []
    imageDict = {}
    for i in lines:
        temp = i.split(' ')
        imageDict[temp[0]] = int(temp[1]) - class_adju
        imlist.append(temp[0])

    return imlist, imageDict


def load_image_dets(filename):
    '''
    load detections with format <class_name conf x1 x2 y1 y2>
    '''
    with open(filename) as f:
        lines = [line.rstrip('\n') for line in f]
    dets = []
    for i in lines:
        temp = i.split(' ')
        conf, x1, x2, y1, y2 = [float(x) for x in temp[-5:]]
        class_name = ' '.join(temp[:-5])
        dets.append([class_name, conf, [x1, x2, y1, y2]])
    return dets


def resize_dets(dets, im_target_size, initial_image_size):
    '''
    resize dets from initial size to target size and round them off
    to image size
    '''
    xscale = im_target_size / float(initial_image_size[1])
    yscale = im_target_size / float(initial_image_size[0])
    for i in range(len(dets)):
        dets[i][2][0] = int(dets[i][2][0] * xscale)
        dets[i][2][1] = int(dets[i][2][1] * xscale)
        dets[i][2][2] = int(dets[i][2][2] * yscale)
        dets[i][2][3] = int(dets[i][2][3] * yscale)

        dets[i][2][0] = min(max(dets[i][2][0], 0.0), im_target_size)
        dets[i][2][1] = min(max(dets[i][2][1], 0.0), im_target_size)
        dets[i][2][2] = min(max(dets[i][2][2], 0.0), im_target_size)
        dets[i][2][3] = min(max(dets[i][2][3], 0.0), im_target_size)

    return dets


net = 'floor'

if net == 'floor':
    fileName_test_visu = 'imagelist_all_test.txt'
    class_size = 6
    class_adju = 2
    im_target_size = 227
    initial_image_size = (768, 1024)  #rows, cols
    im_obj_det_suf = '_dets.txt'
    visu_file_suf = '--M-nSize-1000-tstamp---visualizations' + '.pickle'

#img directory
img_data_dir = 'data/data_' + net + '/'
img_data_list = img_data_dir + fileName_test_visu
#object detec direc
img_obj_dets_dir = 'data/data_yolo_' + net + '/'
#importance heat map direc
img_imp_dir = 'visu/' + net + '_NetResults_visu_n_/'

imlist, imageDict = load_image_name(img_data_list, class_adju)

for im1 in range(len(imlist)):
    print 'explaning -', imageDict[imlist[im1]], imlist[im1]

    im = h2._load_image(
        img_name=img_data_dir + imlist[im1], im_target_size=im_target_size)

    #get heat map
    visu_file = img_imp_dir + imlist[im1].split('.')[0] + visu_file_suf
    #print "visu file", visu_file
    with open(visu_file) as f:
        im_name, class_index, tech_s, size_patch_s, outputBlobName, outputLayerName, dilate_iteration_s, heat_map_raw_occ_s, heat_map_raw_grad_s, heat_map_raw_exci_s = pickle.load(
            f)

    #get object detections
    det_file = img_obj_dets_dir + imlist[im1].split('.')[0] + im_obj_det_suf
    #print "det file", det_file
    dets = load_image_dets(det_file)
    dets = resize_dets(copy.deepcopy(dets), im_target_size, initial_image_size)

    #TODO find relevent dets
    #TODO convert to sentence
