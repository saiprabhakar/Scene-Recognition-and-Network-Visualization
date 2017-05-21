# --------------------------------------------------------
# floor_recog
# Written by Sai Prabhakar
# CMU-RI Masters
# --------------------------------------------------------

from modifiedSiamese.SiameseTrainer import *
import os
import argparse
from ipdb import set_trace as debug
'''
Script to generate visualization using a specif tech (this is a compressed version of visuModel.py)
'''


def generate_visualizations(dataset, viz_tech, fileName_test_visu, data_folder,
                            visu_all_save_dir):

    v = 1
    #net = "floor"
    #net = "places"
    #net = "rand"

    net = dataset
    save_data = 1
    save_img = 0

    visu = 1
    heat_mask_ratio = 0.05

    visu_all_pos = True  #False
    analyse_all_visualizations = 0
    test_prototxt0 = None
    test_prototxt1 = None
    #Note use the grad prototxt file shouldnt have any softmax

    if net == "floor":
        netSize = 1000
        if 'occ' in viz_tech or 'exci' in viz_tech:
            test_prototxt0 = 'modifiedSiameseModels/extracted_siamesePlaces_' + str(
                netSize) + '_test.prototxt'
        if 'grad' in viz_tech:
            test_prototxt1 = 'modifiedSiameseModels/grad_visu_extracted_siamesePlaces_' + str(
                netSize) + '_test.prototxt'
        meanfile = 'placesOriginalModel/places205CNN_mean.binaryproto'
        trainedModel = 'modifiedNetResults/Modified-netsize-1000-epoch-18-tstamp--Timestamp-2017-01-22-20:02:03-net.caffemodel'
        class_size = 6
        class_adju = 2
        im_target_size = 227
        final_layer = 'fc9_f'  #final_layer
        outputLayerName = 'pool2'
        outputBlobName = 'pool2'
        #outputLayerName = 'conv2'
        #outputBlobName = 'conv2'
        topBlobName = 'fc9_f'
        topLayerName = 'fc9_f'
        secondTopLayerName = 'fc8_s'
        secondTopBlobName = 'fc8_s_r'

    elif net == "places":
        netSize = 1000
        if 'occ' in viz_tech or 'exci' in viz_tech:
            test_prototxt0 = 'placesOriginalModel/deploy_alexnet_places365.prototxt'
        if 'grad' in viz_tech:
            test_prototxt1 = 'placesOriginalModel/grad_visu_deploy_alexnet_places365.prototxt'
        meanfile = 'placesOriginalModel/places365CNN_mean.binaryproto'
        trainedModel = 'placesOriginalModel/alexnet_places365.caffemodel'  #None
        class_size = 365
        class_adju = 0
        im_target_size = 227
        final_layer = 'fc8'  #final_layer
        outputLayerName = 'pool2'
        outputBlobName = 'pool2'
        topBlobName = 'fc8'
        topLayerName = 'fc8'
        secondTopLayerName = 'fc7'
        secondTopBlobName = 'fc7'

    else:  # net == "rand":
        netSize = 1000
        if 'occ' in viz_tech or 'exci' in viz_tech:
            test_prototxt0 = 'modifiedSiameseModels/extracted_siamesePlaces_' + str(
                netSize) + '_test.prototxt'
        if 'grad' in viz_tech:
            test_prototxt1 = 'modifiedSiameseModels/grad_visu_extracted_siamesePlaces_' + str(
                netSize) + '_test.prototxt'

        meanfile = 'placesOriginalModel/places205CNN_mean.binaryproto'
        trainedModel = 'modifiedNetResults/Modified-netsize-1000-epoch-18-tstamp--Timestamp-2017-01-22-20:02:03-net.caffemodel'
        class_size = 6
        class_adju = 2
        im_target_size = 227
        final_layer = 'fc9_f'  #final_layer
        outputLayerName = 'pool2'
        outputBlobName = 'pool2'
        #outputLayerName = 'conv2'
        #outputBlobName = 'conv2'
        topBlobName = 'fc9_f'
        topLayerName = 'fc9_f'
        secondTopLayerName = 'fc8_s'
        secondTopBlobName = 'fc8_s_r'

    print heat_mask_ratio
    print fileName_test_visu
    #####################################################

    pretrained_model_proto = None  #'placesOriginalModel/places_processed.prototxt'
    pretrained_model = None  #'placesOriginalModel/places205CNN_iter_300000_upgraded.caffemodel'
    siameseSolver = None  #'modifiedSiameseModels/siamesePlaces_' + str(netSize) + '_solver.prototxt'
    train = 0

    # mkdir <net>_NetResults_visu_grad/occ
    #save dir is used by cam too
    #TODO create folders if doesnt exist
    if save_data and os.path.isdir(visu_all_save_dir) == False:
        os.system('mkdir ' + visu_all_save_dir)
    if save_img:
        for t in viz_tech:
            if os.path.isdir('mkdir ' + visu_all_save_dir + '_' + t) == False:
                os.system('mkdir ' + visu_all_save_dir + '_' + t)

    testProto1 = None
    compare = 1
    size_patch_s = []
    dilate_iteration_s = []

    #load appropriate model while testing
    pretrainedSiameseModel = trainedModel
    size_patch_s = [10]  #, 50, 100]
    dilate_iteration_s = [0]

    siameseTrainer(
        siameseSolver=siameseSolver,
        pretrainedSiameseModel=pretrainedSiameseModel,
        fileName_test_visu=fileName_test_visu,
        pretrained_model=pretrained_model,
        pretrained_model_proto=pretrained_model_proto,
        testProto=test_prototxt0,
        testProto1=test_prototxt1,
        compare=compare,
        im_target_size=im_target_size,
        train=train,
        visu=visu,
        visu_all=visu_all_pos,
        heat_mask_ratio=heat_mask_ratio,
        visu_all_save_dir=visu_all_save_dir,
        viz_tech=viz_tech,
        meanfile=meanfile,
        net=net,
        final_layer=final_layer,
        data_folder=data_folder,
        class_size=class_size,
        class_adju=class_adju,
        save=save_data,
        save_img=save_img,
        netSize=netSize,
        outputLayerName=outputLayerName,
        outputBlobName=outputBlobName,
        topLayerName=topLayerName,
        topBlobName=topBlobName,
        secondTopLayerName=secondTopLayerName,
        secondTopBlobName=secondTopBlobName,
        dilate_iteration_s=dilate_iteration_s,
        size_patch_s=size_patch_s)


if __name__ == '__main__':
    #TODO arg parse
    #For now only works for grad
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='foo help')
    args = parser.parse_args()
    fileName_test_visu = 'data/data_floor/imagelist_all.txt'
    data_folder = 'data/data_floor/'
    visu_all_save_dir = "visu/" + args.dataset + '_NetResults_visu_n_'
    generate_visualizations(args.dataset, ['grad'], fileName_test_visu,
                            data_folder, visu_all_save_dir)
