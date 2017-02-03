# --------------------------------------------------------
# floor_recog
# Written by Sai Prabhakar
# CMU-RI Masters
# --------------------------------------------------------

from modifiedSiamese.SiameseTrainer import *
from modifiedSiamese.analyse_vis import *
from sacred import Experiment

ex = Experiment('visuModels')
###change the fc8 layer size here
###make sure you have created the appropriate prototxt files first


@ex.config
def default_config():
    #to visualize toggle train and visu
    #to test toggle only train
    #to save all the possible masks of visualization use visu_all_pos and visu
    v = 0
    net = "floor"
    tech = 'both'

    if v == 1:
        visu = 1
        visu_all_pos = True  #False
        analyse_all_visualizations = 0
    else:
        visu = 0
        visu_all_pos = False
        analyse_all_visualizations = 1
        heat_map_ratio = 0.25

    if net == "floor":
        netSize = 1000
        test_prototxt0 = 'modifiedSiameseModels/extracted_siamesePlaces_' + str(
            netSize) + '_test.prototxt'
        test_prototxt1 = 'modifiedSiameseModels/grad_visu_extracted_siamesePlaces_' + str(
            netSize) + '_test.prototxt'
        meanfile = 'placesOriginalModel/places205CNN_mean.binaryproto'
        trainedModel = 'modifiedNetResults/Modified-netsize-1000-epoch-18-tstamp--Timestamp-2017-01-22-20:02:03-net.caffemodel'
        fileName_test_visu = 'data/imagelist_all.txt'
        class_size = 6
        class_adju = 2
        data_folder = 'data/'
        im_target_size = 227
        save = 1

    elif net == "places":
        test_prototxt0 = None
        test_prototxt1 = None
        meanfile = ''
        trainedModel = None

#####################################################

    pretrained_model_proto = None  #'placesOriginalModel/places_processed.prototxt'
    pretrained_model = None  #'placesOriginalModel/places205CNN_iter_300000_upgraded.caffemodel'
    siameseSolver = None  #'modifiedSiameseModels/siamesePlaces_' + str(netSize) + '_solver.prototxt'
    train = 0

    # mkdir <net>_NetResults_visu_grad/occ
    visu_all_save_dir = net + '_NetResults_visu/'
    visu_all_analyse_dir = net + '_NetResults_visu_back/'

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


@ex.automain
def execute(siameseSolver, im_target_size, pretrainedSiameseModel,
            fileName_test_visu, pretrained_model, pretrained_model_proto,
            testProto, testProto1, compare, train, visu, visu_all_pos, tech,
            visu_all_save_dir, meanfile, analyse_all_visualizations,
            visu_all_analyse_dir, class_size, class_adju, net, data_folder,
            netSize, heat_map_ratio, _run):

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
            heat_mask_ratio=heat_map_ratio,
            im_target_size=im_target_size,
            data_folder=data_folder,
            _run=_run)
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
            train=train,
            visu=visu,
            visu_all=visu_all_pos,
            visu_all_save_dir=visu_all_save_dir,
            viz_tech=tech,
            meanfile=meanfile,
            net=net,
            netSize=netSize)
