import sys
import os
os.environ['GLOG_minloglevel'] = '3'
from ipdb import set_trace as debug

from explainScene import describe_dataset
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


def describe_all_images(dataset, fileName, data_file, yolo_thresh,
                        yolo_hier_thresh, viz_tech, dilate_iterations,
                        importance_ratio, thres_overlap, thres_conf):
    '''
    Complete pipeline for generating explanations.
    1. Creates object dets output using yolo9000.
    2. Creates visualization heat maps for the dataset.
    3. Generates explantions using the two
    '''

    img_data_dir = 'data/data_' + dataset + '/'
    fileName_test_visu = img_data_dir + fileName  #'imagelist_all.txt'
    img_data_dir = 'data/data_' + dataset + '/' + data_file
    recogdir = os.getcwd() + '/'

    yolodir = '../darknet/'
    yolo_image_list = img_data_dir + 'imagelist_yolo_all.txt'
    yolo_out_dir = 'data/data_' + dataset + '_yolo_dets'
    create_yolo_filelist(fileName_test_visu, img_data_dir, yolo_image_list)
    if os.path.isdir(yolo_out_dir) == False:
        os.system('mkdir ' + yolo_out_dir)

    #object detection with yolo
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
    generate_visualizations(
        dataset,
        viz_tech,
        fileName_test_visu,
        data_folder=img_data_dir,
        visu_all_save_dir=img_imp_dir)

    #Generate explanations
    describe_dataset(dataset, fileName_test_visu, img_data_dir,
                     yolo_out_dir + '/', img_imp_dir, dilate_iterations,
                     importance_ratio, thres_overlap, thres_conf)


if __name__ == '__main__':
    #data config
    dataset = 'places'
    datafile = 'val_256/'
    fileName = 'imagelist_all1.txt'

    #yolo config
    yolo_thresh = 0.1
    yolo_hier_thresh = 0.3

    #visu config
    dilate_iterations = 2
    importance_ratio = 0.25
    viz_tech = ['grad']

    #description config
    thres_overlap = 0.3
    thres_conf = 0.0

    describe_all_images(dataset, fileName, datafile, yolo_thresh,
                        yolo_hier_thresh, viz_tech, dilate_iterations,
                        importance_ratio, thres_overlap, thres_conf)
