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
import caffe
from pythonlayers.helpers import *
import modifiedSiamese.helpers2 as h2
import matplotlib.pyplot as plt
import datetime

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

import pickle
from os import listdir
from os.path import isfile, join


class AnalyseVisualizations(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self,
                 pretrainedSiameseModel=None,
                 testProto=None,
                 netSize=1000,
                 analysis=0,
                 tech=None,
                 class_size=6,
                 class_adju=2):
        """Initialize the SolverWrapper."""
        caffe.set_device(0)
        caffe.set_mode_gpu()
        self.netSize = netSize
        self.class_size = class_size
        self.class_adju = class_adju
        self.data_folder = 'data/'
        self.im_target_size = 227
        self.meanarr = h2._load_mean_binaryproto(
            fileName='placesOriginalModel/places205CNN_mean.binaryproto',
            im_target_size=self.im_target_size)

        print 'anaysis net initializing'
        assert testProto != None
        assert pretrainedSiameseModel != None
        self.siameseTestNet = caffe.Net(testProto, pretrainedSiameseModel,
                                        caffe.TEST)

    def analyse_visualizations(self, database_fileName, visu_all_analyse_dir):
        """Analysing visualizations saved in the files
        """
        tStamp = '-Timestamp-{:%Y-%m-%d-%H:%M:%S}'.format(
            datetime.datetime.now())
        f = open(database_fileName)
        lines = [line.rstrip('\n') for line in f]
        imageDict = {}
        imlist = []
        for i in lines:
            temp = i.split(' ')
            imageDict[temp[0]] = int(temp[1]) - self.class_adju
            imlist.append(temp[0])

        visu_file_s = [f for f in listdir(visu_all_analyse_dir)
                       if (isfile(join(visu_all_analyse_dir, f)) and
                           os.path.splitext(f)[1] == '.pickle')]

        def get_prediction_prob(image_blob, class_index):
            self.siameseTestNet.forward(data=image_blob.astype(
                np.float32, copy=True))
            #import IPython
            #IPython.embed()
            p = self.siameseTestNet.blobs['fc9_f'].data[0].copy()
            p = p - p.min()
            p = p / p.sum()
            prob1 = p[class_index]
            return prob1, p

        for i in range(len(visu_file_s)):
            visu_file = visu_all_analyse_dir + visu_file_s[i]
            with open(visu_file) as f:
                im_name, tech_s, size_patch_s, dilate_iteration_s, heat_map_occ_s, heat_map_raw_occ_s, heat_map_grad_s, heat_map_raw_grad_s = pickle.load(
                    f)

            img = h2._load_image(
                img_name=self.data_folder + im_name,
                im_target_size=self.im_target_size)
            original_blob = h2._get_image_blob_from_image(img, self.meanarr,
                                                          self.im_target_size)
            orig_prob, all_prob = get_prediction_prob(original_blob,
                                                      imageDict[im_name])
            print "original prob", im_name, imageDict[
                im_name], orig_prob, all_prob

            #for i in range(7):
            #    size = 17 + 2*i
            #    print size
            #    kernel = np.ones((size,size),np.float32)/(size*size)
            #    dst = cv2.filter2D(img,-1,kernel)
            #
            #    blurred_blob = h2._get_image_blob_from_image(dst, self.meanarr, self.im_target_size)
            #    blur_prob, all_prob = get_prediction_prob(blurred_blob, imageDict[im_name])
            #    print blur_prob, all_prob
            #
            #    plt.subplot(121),plt.imshow(img[:,:,::-1].astype(np.uint8)),plt.title('Original')
            #    plt.xticks([]), plt.yticks([])
            #    plt.subplot(122),plt.imshow(dst[:,:,::-1].astype(np.uint8)),plt.title('Averaging')
            #    plt.xticks([]), plt.yticks([])
            #    plt.show()

            size = 17
            kernel = np.ones((size, size), np.float32) / (size * size)
            img_blur = cv2.filter2D(img, -1, kernel)
            blurred_blob = h2._get_image_blob_from_image(
                img_blur, self.meanarr, self.im_target_size)
            blur_prob, all_prob = get_prediction_prob(blurred_blob,
                                                      imageDict[im_name])
            print blur_prob, all_prob

            for i in range(len(size_patch_s)):
                heat_mask_o = heat_map_occ_s[i]
                c_img = h2._combine_images(img_blur, heat_mask_o, img)
                c_blob = h2._get_image_blob_from_image(c_img, self.meanarr,
                                                       self.im_target_size)
                c_prob, c_all_prob = get_prediction_prob(c_blob,
                                                         imageDict[im_name])
                print size_patch_s[i], c_prob, c_all_prob

                plt.subplot(131), plt.imshow(img[:, :, ::-1].astype(
                    np.uint8)), plt.title('Original')
                plt.xticks([]), plt.yticks([])
                plt.subplot(132), plt.imshow(img_blur[:, :, ::-1].astype(
                    np.uint8)), plt.title('Averaging')
                plt.xticks([]), plt.yticks([])
                plt.subplot(133), plt.imshow(c_img[:, :, ::-1].astype(
                    np.uint8)), plt.title('final')
                plt.xticks([]), plt.yticks([])
                plt.show()

            import IPython
            IPython.embed()


def analyseNet(pretrainedSiameseModel,
               testProto,
               fileName_test_visu,
               viz_tech=None,
               analyse_all_visualizations=0,
               visu_all_analyse_dir=None,
               netSize=1000):

    sw = AnalyseVisualizations(
        pretrainedSiameseModel=pretrainedSiameseModel,
        testProto=testProto,
        analysis=analyse_all_visualizations,
        tech=viz_tech,
        netSize=netSize)

    if analyse_all_visualizations == 1:
        print 'analysing all visualization'
        sw.analyse_visualizations(
            database_fileName=fileName_test_visu,
            visu_all_analyse_dir=visu_all_analyse_dir)
    else:  #if visu == 0:
        print 'testing not implemented'
        #print "testing with ", pretrainedSiameseModel
        #sw.test(fileName_test_visu)
