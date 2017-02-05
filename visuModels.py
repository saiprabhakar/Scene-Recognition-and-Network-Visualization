# --------------------------------------------------------
# floor_recog
# Written by Sai Prabhakar
# CMU-RI Masters
# --------------------------------------------------------

from modifiedSiamese.SiameseTrainer import *
from modifiedSiamese.analyse_vis import *

#to visualize toggle train and visu
#to test toggle only train
#to save all the possible masks of visualization use visu_all_pos and visu
v = 1
#net = "floor"
net = "places"
tech = 'both'
save_data = 1
save_img = 0

if v == 1:
    visu = 1
    visu_all_pos = True  #False
    analyse_all_visualizations = 0
else:
    visu = 0
    visu_all_pos = False
    analyse_all_visualizations = 1

heat_mask_ratio = 0.5

if net == "floor":
    netSize = 1000
    test_prototxt0 = 'modifiedSiameseModels/extracted_siamesePlaces_' + str(
        netSize) + '_test.prototxt'
    test_prototxt1 = 'modifiedSiameseModels/grad_visu_extracted_siamesePlaces_' + str(
        netSize) + '_test.prototxt'
    meanfile = 'placesOriginalModel/places205CNN_mean.binaryproto'
    trainedModel = 'modifiedNetResults/Modified-netsize-1000-epoch-18-tstamp--Timestamp-2017-01-22-20:02:03-net.caffemodel'
    fileName_test_visu = 'data/data_floor/imagelist_all.txt'
    class_size = 6
    class_adju = 2
    data_folder = 'data/data_floor/'
    im_target_size = 227
    final_layer = 'fc9_f'  #final_layer
    data_index = ''

elif net == "places":
    netSize = 1000
    test_prototxt0 = 'placesOriginalModel/deploy_alexnet_places365.prototxt'
    test_prototxt1 = 'placesOriginalModel/grad_visu_deploy_alexnet_places365.prototxt'
    meanfile = 'placesOriginalModel/places365CNN_mean.binaryproto'
    trainedModel = 'placesOriginalModel/alexnet_places365.caffemodel'  #None
    data_index = '1'
    fileName_test_visu = 'data/data_places/images_all' + data_index + '.txt'
    class_size = 365
    class_adju = 0
    data_folder = 'data/data_places/val_256/'
    im_target_size = 227
    final_layer = 'fc8'  #final_layer
print heat_mask_ratio
#####################################################

pretrained_model_proto = None  #'placesOriginalModel/places_processed.prototxt'
pretrained_model = None  #'placesOriginalModel/places205CNN_iter_300000_upgraded.caffemodel'
siameseSolver = None  #'modifiedSiameseModels/siamesePlaces_' + str(netSize) + '_solver.prototxt'
train = 0

# mkdir <net>_NetResults_visu_grad/occ
visu_all_save_dir = "visu/" + net + '_NetResults_visu' + data_index
visu_all_analyse_dir = 'visu/' + net + '_NetResults_visu' + data_index + '/'
#visu_all_analyse_dir = 'visu/' + net + '_NetResults_visu_back/'

testProto1 = None
compare = 1
if visu == 1:
    #load appropriate model while testing
    pretrainedSiameseModel = trainedModel
    if tech == 'both':
        testProto = test_prototxt0
        testProto1 = test_prototxt1
        compare = 1
elif analyse_all_visualizations == 1:
    # loading test prototype for analysis
    pretrainedSiameseModel = trainedModel
    testProto = test_prototxt0
_run = []

if analyse_all_visualizations == 1:
    analyseNet(
        pretrainedSiameseModel=pretrainedSiameseModel,
        testProto=testProto,
        analyse_all_visualizations=analyse_all_visualizations,
        visu_all_analyse_dir=visu_all_analyse_dir,
        fileName_test_visu=fileName_test_visu,
        viz_tech=tech,
        meanfile=meanfile,
        netSize=netSize,
        class_size=class_size,
        class_adju=class_adju,
        heat_mask_ratio=heat_mask_ratio,
        im_target_size=im_target_size,
        final_layer=final_layer,
        net=net,
        save_img=save_img,
        save_data=save_data,
        data_folder=data_folder)
else:
    siameseTrainer(
        siameseSolver=siameseSolver,
        pretrainedSiameseModel=pretrainedSiameseModel,
        fileName_test_visu=fileName_test_visu,
        pretrained_model=pretrained_model,
        pretrained_model_proto=pretrained_model_proto,
        testProto=testProto,
        testProto1=testProto1,
        compare=compare,
        im_target_size=im_target_size,
        train=train,
        visu=visu,
        visu_all=visu_all_pos,
        heat_mask_ratio=heat_mask_ratio,
        visu_all_save_dir=visu_all_save_dir,
        viz_tech=tech,
        meanfile=meanfile,
        net=net,
        final_layer=final_layer,
        data_folder=data_folder,
        class_size=class_size,
        class_adju=class_adju,
        save=save_data,
        save_img=save_img,
        netSize=netSize)
