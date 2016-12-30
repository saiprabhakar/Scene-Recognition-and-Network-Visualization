# --------------------------------------------------------
# floor_recog
# Written by Sai Prabhakar
# CMU-RI Masters
# --------------------------------------------------------

from modifiedSiamese.SiameseTrainer import *

###change the fc8 layer size here
###make sure you have created the appropriate prototxt files first
netSize = 1000

pretrained_model_proto = 'placesOriginalModel/places_processed.prototxt'
pretrained_model = 'placesOriginalModel/places205CNN_iter_300000_upgraded.caffemodel'
siameseSolver = 'modifiedSiameseModels/siamesePlaces_' + str(
    netSize) + '_solver.prototxt'
fileName = 'data/imagelist.txt'

#to visualize toggle train and visu
#to test toggle only train
train = 0
visu = 1

#specify the technique used to visualize the network
#tech = 'grad'
tech = 'occ'

if visu == 1:
    #load appropriate model while testing
    pretrainedSiameseModel = 'modifiedNetResults/Modified-netsize-1000-epoch-4-tstamp--Timestamp-2016-12-23-05:23:20-net-final.caffemodel'
    if tech == 'occ':
        testProto = 'modifiedSiameseModels/extracted_siamesePlaces_' + str(
            netSize) + '_test.prototxt'
    elif tech == 'grad':
        #needs force gradient line in the prototxt file
        testProto = 'modifiedSiameseModels/grad_visu_extracted_siamesePlaces_' + str(
            netSize) + '_test.prototxt'
else:
    tech = None
    testProto = None
    pretrainedSiameseModel = None

siameseTrainer(
    siameseSolver=siameseSolver,
    pretrainedSiameseModel=pretrainedSiameseModel,
    fileName=fileName,
    pretrained_model=pretrained_model,
    pretrained_model_proto=pretrained_model_proto,
    testProto=testProto,
    train=train,
    visu=visu,
    viz_tech=tech,
    netSize=netSize)
