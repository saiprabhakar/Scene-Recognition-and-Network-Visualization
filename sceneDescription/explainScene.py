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
from dataset_processor import *
'''
Script to generate explanation for scene detection
given object detection and important region heat map.
'''


def load_image_dets(filename):
    '''
    Load detections with format <object_name conf x1 x2 y1 y2>
    '''
    with open(filename) as f:
        lines = [line.rstrip('\n') for line in f]
    dets = []
    for i in lines:
        temp = i.split(' ')
        conf, x1, x2, y1, y2 = [float(x) for x in temp[-5:]]
        object_name = ' '.join(temp[:-5])
        dets.append([object_name, conf, [x1, x2, y1, y2]])
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
    #TODO replace these by semantics
    rel_dets = []
    ignore_id = ['living thing', 'person']
    #ignore_id = ['home appliance', 'living thing', 'container', 'artifact',
    #             'person', 'conveyance', 'bottle', 'instrumentality', 'whole']
    group_id = [['chair', 'seat'], ['furnishing', 'furniture']]

    group_dict = {}
    for i in range(len(group_id)):
        for j in range(len(group_id[i])):
            group_dict[group_id[i][j]] = group_id[i][0]

    for i in range(len(dets)):
        if dets[i][0] not in ignore_id:
            if overlaps[i] > thres_overlap:
                if dets[i][1] > thres_conf:
                    t_det = dets[i]
                    if len(t_det) > 0:
                        t_det[0] = group_dict[t_det[0]] if (
                            t_det[0] in group_dict) else t_det[0]
                    rel_dets.append(t_det)
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


def get_rel_dets_dataset(dataset, img_data_list, img_data_dir,
                         img_obj_dets_dir, img_imp_dir, dilate_iterations,
                         importance_ratio, thres_overlap, thres_conf,
                         is_sub_scene, class_size, class_adju, im_target_size,
                         initial_image_size, class_names):
    '''
    Get relevant detection for entire dataset
    '''

    imlist, imageDict = load_image_name(img_data_list, class_adju)

    rel_dets_all = get_features_dataset(
        dataset, img_data_list, img_data_dir, img_obj_dets_dir, img_imp_dir,
        dilate_iterations, importance_ratio, thres_overlap, thres_conf,
        class_names, imlist, imageDict, im_target_size, initial_image_size)

    return rel_dets_all, imlist, imageDict


def classwise_features(rel_det_all, imlist, imageDict, class_name):
    '''
    Donot use this function
    Only for reference
    '''
    if is_sub_scene:
        obj_dict, feats_all = get_number_features(rel_dets_all)
        use_d_tree = 1
        if use_d_tree == 0:
            #Get class features
            class_feat_all = get_class_features(obj_dict, feats_all, imlist,
                                                imageDict)
            #TODO find difference between classes
            class_uni_feat_all = get_unique_class_features(class_feat_all)
            debug()
        else:
            #Get complete features
            feats_comp_all, feats_y, feat_list = create_complete_feat(
                feats_all, imlist, imageDict)
            #Build tree
            d_tree = create_tree(feats_comp_all,
                                 feats_y)  #rel_dets_all, imlist, imageDict)
            print_tree(d_tree, feat_list, obj_dict, class_names)
    else:
        for im1 in range(len(imlist)):
            print 'explaning -', imageDict[imlist[im1]], imlist[im1]
            #debug()
            #Convert to sentence
            #TODO find the predicted class name
            description = describe_scene(rel_dets_all[im1], class_names[0])
            print description
            #debug()


def get_features_dataset(
        dataset, img_data_list, img_data_dir, img_obj_dets_dir, img_imp_dir,
        dilate_iterations, importance_ratio, thres_overlap, thres_conf,
        class_names, imlist, imageDict, im_target_size, initial_image_size):
    net = dataset  #'floor'
    kernel = np.ones((3, 3), np.uint8)
    #dilate_iterations = 2
    #importance_ratio = 0.25
    #thres_overlap = 0.3
    #thres_conf = 0.0
    visu_file_suf = '--M-nSize-1000-tstamp---visualizations' + '.pickle'
    im_obj_det_suf = '_dets.txt'
    rel_dets_all = []

    #Img directory
    #img_data_dir = 'data/data_' + net + '/'
    #img_data_list = img_data_dir + fileName_test_visu
    #Object detec direc
    #img_obj_dets_dir = 'data/data_yolo_' + net + '/'
    #Importance heat map direc
    #img_imp_dir = 'visu/' + net + '_NetResults_visu_n_/'

    for im1 in range(len(imlist)):

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
        rel_dets_all.append(rel_dets)
    return rel_dets_all


def get_number_features(rel_dets_all,
                        no_regions,
                        im_target_size,
                        obj_next=0,
                        obj_dict={}):
    '''
    Get trainable features of the form [obj_id, x, y]
    '''
    feats_all = []
    for i in range(len(rel_dets_all)):
        feats = []
        for dets in rel_dets_all[i]:
            if dets[0] not in obj_dict:
                obj_dict[dets[0]] = obj_next
                obj_next += 1
            obj_id = obj_dict[dets[0]]
            obj_x = (dets[2][0] + dets[2][1]) * 0.5
            obj_y = (dets[2][2] + dets[2][3]) * 0.5

            obj_x_id = int(obj_x / (im_target_size / no_regions))
            obj_y_id = int(obj_y / (im_target_size / no_regions))
            feats.append([obj_id, obj_x_id, obj_y_id])
        feats = find_unique(feats)
        feats_all.append(feats)

    return obj_dict, obj_next, feats_all


def find_unique(feats):
    '''
    find unique elements in list of list
    '''
    f = [tuple(i) for i in feats]
    f = [list(i) for i in set(f)]
    return f


def find_intersection(feat_1, feat_2):
    '''
    Find intersection between 2 list of list feats
    '''
    feat_1_t = [tuple(i) for i in feat_1]
    feat_2_t = [tuple(i) for i in feat_2]
    feat_f = set(feat_1_t).intersection(feat_2_t)
    feat_f = [list(i) for i in set(feat_f)]
    return feat_f


def find_union(feat_1, feat_2):
    '''
    Find union between 2 list of list feats
    '''
    feat_1_t = [tuple(i) for i in feat_1]
    feat_2_t = [tuple(i) for i in feat_2]
    feat_f = find_unique(feat_1_t + feat_2_t)
    return feat_f


def get_class_features(obj_dict, feats_all, imlist, imageDict):
    '''
    Get class features
    '''
    class_feat_all = {}  #[[]]*len(set(obj_dict.values()))
    for i in range(len(imlist)):
        class_id = imageDict[imlist[i]]
        if class_id in class_feat_all:
            class_feat_t = class_feat_all[class_id]
            #TODO instead of union try others like 1. intersection, 2. find most frequent ones
            class_feat = find_union(class_feat_t, feats_all[i])
        else:
            #debug()
            class_feat = feats_all[i]
        class_feat_all[class_id] = class_feat

    return class_feat_all


def create_complete_feat(feats_all, imlist, imageDict):
    '''
    Create complete features and secondary feature dictionary
    '''
    feats = [f for feat in feats_all for f in feat]
    feat_list = list(set([tuple(f) for f in feats]))
    feat_list = [list(f) for f in feat_list]
    feats_comp_all = []
    feats_y = []
    for im, feat in zip(imlist, feats_all):
        feat_t = np.zeros(len(feat_list))
        for f in feat:
            id_ = feat_list.index(f)
            feat_t[id_] = 1
        feats_comp_all.append(feat_t)
        feats_y.append(imageDict[im])
    return feats_comp_all, feats_y, feat_list


def create_tree(feats_comp_all, feats_y):
    '''
    Find features and build tree
    '''
    #TODO
    from sklearn import tree
    d_t = tree.DecisionTreeClassifier()
    d_t = d_t.fit(feats_comp_all, feats_y)

    #for feats in zip(feats_comp_all, feats_y):
    #    d_t.predict(
    feats_y_p = d_t.predict(feats_comp_all)
    print "actual and predicted labels"
    print feats_y
    print feats_y_p
    return d_t


def print_feat_list_list(feat_1, obj_dict):
    '''
    create feature names
    '''
    feat_names = []
    for f in feat_1:
        obj_name = obj_dict.keys()[obj_dict.values().index(f[0])]
        s_ = str(f[1]) + "-" + str(f[2]) + ":" + obj_name
        feat_names.append(s_)
    return feat_names


def print_tree(d_t, feat_list, obj_dict, class_names):
    '''
    Visualize decision tree, features are of the form [obj_id, x, y]
    '''
    from sklearn import tree
    import pydotplus
    from cStringIO import StringIO
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    #create feature names
    feat_names = print_feat_list_list(feat_list, obj_dict)

    #build graph
    dot_data = tree.export_graphviz(
        d_t,
        out_file=None,
        feature_names=feat_names,
        class_names=class_names,
        filled=True,
        rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)

    #vizualize graph
    png_str = graph.create_png(prog='dot')
    sio = StringIO()
    sio.write(png_str)
    sio.seek(0)
    img = mpimg.imread(sio)
    fig = plt.gcf()
    imgplot = plt.imshow(img, aspect='equal')
    fig.set_size_inches(185, 105)
    plt.savefig('decision-tree.png', dpi=100)
    #plt.show()
    #plt.show(block=False)


def get_unique_class_features(class_feat_all, print_=0):
    '''
    Get unique features for each class
    '''
    class_uni_feat_all = {}
    for id1 in range(len(class_feat_all)):
        if print_:
            print "---------------- ", id1
            print class_feat_all[id1]
        feat_u = class_feat_all[id1]
        for id2 in range(len(class_feat_all)):
            if id1 != id2:
                feat_u = find_diff(feat_u, class_feat_all[id2])
                if print_:
                    print "*", id2
                    print feat_u
        class_uni_feat_all[id1] = feat_u
    return class_uni_feat_all


def find_diff(f1, f2):
    '''
    Finds f1 - f2
    '''
    f1_t = [tuple(i) for i in f1]
    f2_t = [tuple(i) for i in f2]
    diff12 = [list(i) for i in set(f1_t) - set(f2_t)]
    return diff12


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
