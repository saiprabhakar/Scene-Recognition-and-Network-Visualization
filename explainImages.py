# --------------------------------------------------------
# floor_recog
# Written by Sai Prabhakar
# CMU-RI Masters
# --------------------------------------------------------

import sys
import os
os.environ['GLOG_minloglevel'] = '3'
from ipdb import set_trace as debug

from sceneDescription.explainScene import *
from visuScene import generate_visualizations


def create_yolo_filelist(fileName, img_data_dir, newFileName):
    '''
    Creates new filelist in the format required by yolo
    '''
    prefix = os.getcwd() + '/' + img_data_dir
    with open(fileName) as f:
        lines = [line.rstrip('\n') for line in f]
    with open(newFileName, 'w') as f:
        for i in lines:
            f.write(prefix + '/' + i.split(' ')[0] + '\n')


def prepare_dataset(dataset, fileName, data_file, yolo_thresh,
                    yolo_hier_thresh, viz_tech, dilate_iterations,
                    importance_ratio, thres_overlap, thres_conf, do_yolo,
                    do_vis):
    img_data_dir = 'data/data_' + dataset + '/'
    fileName_visu = img_data_dir + fileName  #'imagelist_all.txt'
    img_data_dir = 'data/data_' + dataset + '/' + data_file
    recogdir = os.getcwd() + '/'

    yolodir = '../darknet/'
    yolo_image_list = img_data_dir + 'imagelist_yolo_all.txt'
    yolo_out_dir = 'data/data_' + dataset + '_yolo_dets'

    #object detection with yolo
    if do_yolo:
        create_yolo_filelist(fileName_visu, img_data_dir, yolo_image_list)
        if os.path.isdir(yolo_out_dir) == False:
            os.system('mkdir ' + yolo_out_dir)
        os.chdir(yolodir)
        cmd_yolo_detection = './darknet detector test_file ' + 'cfg/combine9k.data ' + 'cfg/yolo9000.cfg ' + 'data/yolo9000.weights ' + recogdir + yolo_image_list + ' -thresh ' + str(
            yolo_thresh) + ' -outdir ' + recogdir + yolo_out_dir + ' -hier ' + str(
                yolo_hier_thresh)
        print "excuting yolo detection cmd: ", cmd_yolo_detection
        try:
            os.system(cmd_yolo_detection)
            os.system('pwd')
        finally:
            print "comming back"
            os.chdir(recogdir)
            os.system('rm ' + yolo_image_list)

    #importance region from scene recognition
    img_imp_dir = 'visu/' + dataset + '_NetResults_visu_n_/'
    if do_vis:
        generate_visualizations(
            dataset,
            viz_tech,
            fileName_visu,
            data_folder=img_data_dir,
            visu_all_save_dir=img_imp_dir)
    return fileName_visu, yolo_out_dir, img_data_dir, img_imp_dir


def describe_all_images(dataset, fileName_test, fileName_train, data_file,
                        yolo_thresh, yolo_hier_thresh, viz_tech,
                        dilate_iterations, importance_ratio, thres_overlap,
                        thres_conf, do_yolo, do_vis, is_sub_scene):
    '''
    Complete pipeline for generating explanations.
    1. Creates object dets output using yolo9000.
    2. Creates visualization heat maps for the dataset.
    3. Generates explantions using the two
    '''
    use_spatial = 1

    im_target_size = 227

    #Get test feature
    fileName_visu_test, yolo_out_dir_test, img_data_dir_test, img_imp_dir_test = prepare_dataset(
        dataset, fileName_test, data_file, yolo_thresh, yolo_hier_thresh,
        viz_tech, dilate_iterations, importance_ratio, thres_overlap,
        thres_conf, do_yolo, do_vis)
    rel_det_all_test, imlist_test, imageDict_test, class_name_test = get_rel_dets_dataset(
        dataset, fileName_visu_test, img_data_dir_test,
        yolo_out_dir_test + '/', img_imp_dir_test, dilate_iterations,
        importance_ratio, thres_overlap, thres_conf, is_sub_scene)
    if fileName_train != None:
        #Get train features
        fileName_visu_train, yolo_out_dir_train, img_data_dir_train, img_imp_dir_train = prepare_dataset(
            dataset, fileName_train, data_file, yolo_thresh, yolo_hier_thresh,
            viz_tech, dilate_iterations, importance_ratio, thres_overlap,
            thres_conf, do_yolo, do_vis)
        rel_det_all_train, imlist_train, imageDict_train, class_name_train = get_rel_dets_dataset(
            dataset, fileName_visu_train, img_data_dir_train,
            yolo_out_dir_train + '/', img_imp_dir_train, dilate_iterations,
            importance_ratio, thres_overlap, thres_conf, is_sub_scene)
    assert class_name_train == class_name_test
    class_names = class_name_test

    obj_next = 0
    obj_dict = {}
    no_regions = 1
    if use_spatial == 1:
        no_regions = 5

    obj_dict, obj_next, feats_all_test = get_number_features(
        rel_det_all_test,
        no_regions,
        im_target_size,
        obj_next=obj_next,
        obj_dict=obj_dict)

    if fileName_train != None:
        obj_dict, obj_next, feats_all_train = get_number_features(
            rel_det_all_train,
            no_regions,
            im_target_size,
            obj_next=obj_next,
            obj_dict=obj_dict)
        # class_all_feat and feats_all has [obj, x, y] features
        class_feat_all_train = get_class_features(
            obj_dict, feats_all_train, imlist_train, imageDict_train)
        ##TODO improve the difference between classes
        class_uni_feat_all_train = get_unique_class_features(
            class_feat_all_train, print_=0)

        print "\nclass feat:"
        for key_ in class_feat_all_train.keys():
            print key_, ":", print_feat_list_list(class_feat_all_train[key_],
                                                  obj_dict)
        print "class uni feat:"
        for key_ in class_uni_feat_all_train.keys():
            print key_, ":", print_feat_list_list(
                class_uni_feat_all_train[key_], obj_dict)

        for method in [1, 2, 3]:
            print "\nmethod:", method, "--------------"
            for i in range(len(imlist_test)):
                if method == 1:
                    #intersection btw test[im] and train[class[im]]
                    class_feat_t = class_feat_all_train[imageDict_test[
                        imlist_test[i]]]
                    feat_f = find_intersection(feats_all_test[i], class_feat_t)
                    pass
                elif method == 2:
                    #intersection btw test[im] and train_uni[class[im]]
                    class_feat_t = class_uni_feat_all_train[imageDict_test[
                        imlist_test[i]]]
                    feat_f = find_intersection(feats_all_test[i], class_feat_t)
                    pass
                elif method == 3:
                    #train[class[im]]
                    feat_f = class_feat_all_train[imageDict_test[imlist_test[
                        i]]]
                    pass
                print "class name ", class_names[imageDict_test[imlist_test[
                    i]]]
                print print_feat_list_list(feat_f, obj_dict)
        print "\n"


if __name__ == '__main__':
    #data config
    dataset = 'floor'
    datafile = ''
    fileName_test = 'imagelist_all_test.txt'
    fileName_train = 'imagelist_all.txt'
    do_yolo = 0
    do_vis = 0
    is_sub_scene = 1

    #yolo config
    yolo_thresh = 0.1
    yolo_hier_thresh = 0.3  #0.7

    #visu config
    dilate_iterations = 2
    importance_ratio = 0.25
    viz_tech = ['grad']

    #description config
    thres_overlap = 0.3
    thres_conf = 0.0

    if dataset == 'places':
        datafile = 'val_265/'

    describe_all_images(dataset, fileName_test, fileName_train, datafile,
                        yolo_thresh, yolo_hier_thresh, viz_tech,
                        dilate_iterations, importance_ratio, thres_overlap,
                        thres_conf, do_yolo, do_vis, is_sub_scene)
