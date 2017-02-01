# --------------------------------------------------------
# floor_recog
# Written by Sai Prabhakar
# CMU-RI Masters
# --------------------------------------------------------

from modifiedSiamese.SiameseTrainer import *
from modifiedSiamese.analyse_vis import *

###change the fc8 layer size here
###make sure you have created the appropriate prototxt files first
netSize = 1000

pretrained_model_proto = 'placesOriginalModel/places_processed.prototxt'
pretrained_model = 'placesOriginalModel/places205CNN_iter_300000_upgraded.caffemodel'
siameseSolver = 'modifiedSiameseModels/siamesePlaces_' + str(
    netSize) + '_solver.prototxt'
fileName_test_visu = 'data/imagelist_all.txt'
meanfile = 'placesOriginalModel/places205CNN_mean.binaryproto'
#to visualize toggle train and visu
#to test toggle only train
#to save all the possible masks of visualization use visu_all_pos and visu
train = 0
visu = 0
visu_all_pos = False
analyse_all_visualizations = 1

visu_all_save_dir = 'modifiedNetResults_visu/'
visu_all_analyse_dir = 'modifiedNetResults_visu/'

#specify the technique used to visualize the network
#tech = 'grad'
tech = 'both'
#tech = 'grad'

testProto1 = None
compare = 1
if visu == 1:
    #load appropriate model while testing
    #pretrainedSiameseModel = 'modifiedNetResults/Modified-netsize-1000-epoch-4-tstamp--Timestamp-2016-12-23-05:23:20-net-final.caffemodel'
    pretrainedSiameseModel = 'modifiedNetResults/Modified-netsize-1000-epoch-18-tstamp--Timestamp-2017-01-22-20:02:03-net.caffemodel'
    if tech == 'both':
        testProto = 'modifiedSiameseModels/extracted_siamesePlaces_' + str(
            netSize) + '_test.prototxt'
        testProto1 = 'modifiedSiameseModels/grad_visu_extracted_siamesePlaces_' + str(
            netSize) + '_test.prototxt'
        compare = 1
    elif tech == 'occ':
        testProto = 'modifiedSiameseModels/extracted_siamesePlaces_' + str(
            netSize) + '_test.prototxt'
    elif tech == 'grad':
        #needs force gradient line in the prototxt file
        testProto = 'modifiedSiameseModels/grad_visu_extracted_siamesePlaces_' + str(
            netSize) + '_test.prototxt'
elif analyse_all_visualizations == 1:
    # loading test prototype for analysis
    pretrainedSiameseModel = 'modifiedNetResults/Modified-netsize-1000-epoch-18-tstamp--Timestamp-2017-01-22-20:02:03-net.caffemodel'
    testProto = 'modifiedSiameseModels/extracted_siamesePlaces_' + str(
        netSize) + '_test.prototxt'
else:
    tech = None
    testProto = None
    pretrainedSiameseModel = None

if analyse_all_visualizations == 1:
    analyseNet(
        pretrainedSiameseModel=pretrainedSiameseModel,
        testProto=testProto,
        analyse_all_visualizations=analyse_all_visualizations,
        visu_all_analyse_dir=visu_all_analyse_dir,
        fileName_test_visu=fileName_test_visu,
        viz_tech=tech,
        netSize=netSize)
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
        visu_all_analyse_dir=visu_all_analyse_dir,
        viz_tech=tech,
        netSize=netSize)
