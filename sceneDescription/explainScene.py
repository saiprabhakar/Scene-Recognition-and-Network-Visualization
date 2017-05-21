#/usr/bin/python
# --------------------------------------------------------
# floor_recog
# Written by Sai Prabhakar
# CMU-RI Masters
# --------------------------------------------------------

import sys
import os
os.environ['GLOG_minloglevel'] = '3'

import numpy as np
import cv2
import datetime

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

import pickle
from ipdb import set_trace as debug
import modifiedSiamese.helpers2 as h2
import copy
import argparse
'''
Script to generate explanation for scene detection
given object detection and important region heat map.
'''


def load_image_name(fileName, class_adju):
    '''
    Load file names
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
    Load detections with format <class_name conf x1 x2 y1 y2>
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
    Resize dets from initial size to target size and round them off
    to image size
    '''
    xscale = im_target_size / float(initial_image_size[1])
    yscale = im_target_size / float(initial_image_size[0])
    for i in range(len(dets)):
        dets[i][2][0] = int(dets[i][2][0] * xscale)
        dets[i][2][1] = int(dets[i][2][1] * xscale)
        dets[i][2][2] = int(dets[i][2][2] * yscale)
        dets[i][2][3] = int(dets[i][2][3] * yscale)

        dets[i][2][0] = min(max(dets[i][2][0], 0.0), im_target_size - 1)
        dets[i][2][1] = min(max(dets[i][2][1], 0.0), im_target_size - 1)
        dets[i][2][2] = min(max(dets[i][2][2], 0.0), im_target_size - 1)
        dets[i][2][3] = min(max(dets[i][2][3], 0.0), im_target_size - 1)
    return dets


def find_imp_overlap(dets, heat_mask):
    '''
  Find the overlap with the heat_mask
  '''
    overlaps = np.zeros(len(dets))
    for i in range(len(dets)):
        x1, x2, y1, y2 = dets[i][2]
        det_imp_map = heat_mask[x1:x2, y1:y2]
        overlap = det_imp_map.sum() / float(det_imp_map.size)
        overlaps[i] = overlap
    return overlaps


def find_relevant_dets(dets, overlaps, thres_overlap, thres_conf):
    '''
  Finds relevant detections.
  For each detection have confidence greater than thres_conf
  and if the overlap is greater than thres_overlap it is a
  relevant dets.
  '''
    rel_dets = []
    for i in range(len(dets)):
        if overlaps[i] > thres_overlap:
            if dets[i][1] > thres_conf:
                rel_dets.append(dets[i])
    return rel_dets


def describe_scene(rel_dets, class_id):
    '''
    Describe the scene based on the labels of the rel_dets and
    location of the labels
    '''
    description = 'In ' + class_id + ' because I can see '
    for i in range(len(rel_dets)):
        description += rel_dets[i][0] + " "

    return description


def describe_dataset(dataset, img_data_list, img_data_dir, img_obj_dets_dir,
                     img_imp_dir, dilate_iterations, importance_ratio,
                     thres_overlap, thres_conf):
    net = dataset  #'floor'
    kernel = np.ones((3, 3), np.uint8)
    #dilate_iterations = 2
    #importance_ratio = 0.25
    #thres_overlap = 0.3
    #thres_conf = 0.0
    visu_file_suf = '--M-nSize-1000-tstamp---visualizations' + '.pickle'
    im_obj_det_suf = '_dets.txt'

    if net == "places":
        #fileName_test_visu = 'images_all.txt'
        class_size = 365
        class_adju = 0
        im_target_size = 227
        initial_image_size = (256, 256)  #rows, cols
        class_ids = [''] * 6  #TODO get actual label from file

    else:
        #fileName_test_visu = 'imagelist_all.txt'
        class_size = 6
        class_adju = 2
        im_target_size = 227
        initial_image_size = (768, 1024)  #rows, cols
        class_ids = [''] * 6  #TODO get actual label from file

    #Img directory
    #img_data_dir = 'data/data_' + net + '/'
    #img_data_list = img_data_dir + fileName_test_visu
    #Object detec direc
    #img_obj_dets_dir = 'data/data_yolo_' + net + '/'
    #Importance heat map direc
    #img_imp_dir = 'visu/' + net + '_NetResults_visu_n_/'

    imlist, imageDict = load_image_name(img_data_list, class_adju)

    for im1 in range(len(imlist)):
        print 'explaning -', imageDict[imlist[im1]], imlist[im1]

        im = h2._load_image(
            img_name=img_data_dir + imlist[im1], im_target_size=im_target_size)

        #Get heat map
        visu_file = img_imp_dir + imlist[im1].split('.')[0] + visu_file_suf
        with open(visu_file) as f:
            im_name, class_index, tech_s, size_patch_s, outputBlobName, outputLayerName, dilate_iteration_s, heat_map_raw_occ_s, heat_map_raw_grad_s, heat_map_raw_exci_s = pickle.load(
                f)
        heat_map_raw = heat_map_raw_grad_s[0]

        #Get object detections
        det_file = img_obj_dets_dir + imlist[im1].split('.')[
            0] + im_obj_det_suf
        dets = load_image_dets(det_file)
        dets = resize_dets(
            copy.deepcopy(dets), im_target_size, initial_image_size)

        #Dilate heat map (good for gradient based visualizations)
        if dilate_iterations > 0:
            heat_map_raw = cv2.dilate(
                heat_map_raw, kernel, iterations=dilate_iterations)
        #Find heat_mask from heat_map
        threshold = h2._find_threshold(heat_map_raw, ratio=importance_ratio)
        heat_mask = np.zeros(heat_map_raw.shape)
        heat_mask[heat_map_raw < threshold] = 0
        heat_mask[heat_map_raw >= threshold] = 1

        #Find relevent dets
        overlaps = find_imp_overlap(dets, heat_mask)
        rel_dets = find_relevant_dets(dets, overlaps, thres_overlap,
                                      thres_conf)

        #Convert to sentence
        #TODO find the predicted class id
        description = describe_scene(rel_dets, class_ids[0])
        print description
        #debug()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='foo help')
    args = parser.parse_args()
    net = args.dataset
    fileName_test_visu = 'imagelist_all.txt'
    img_data_dir = 'data/data_' + net + '/'
    img_obj_dets_dir = 'data/data_' + net + '_yolo_dets/'
    img_imp_dir = 'visu/' + net + '_NetResults_visu_n_/'
    img_data_list = img_data_dir + fileName_test_visu
    dilate_iterations = 2
    importance_ratio = 0.5
    thres_overlap = 0.01
    thres_conf = 0.0

    describe_dataset(net, img_data_list, img_data_dir, img_obj_dets_dir,
                     img_imp_dir, dilate_iterations, importance_ratio,
                     thres_overlap, thres_conf)

    #describe_dataset(dataset=args.dataset)
