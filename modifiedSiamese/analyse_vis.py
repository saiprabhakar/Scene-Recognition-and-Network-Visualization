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
                 meanfile='',
                 class_size=6,
                 class_adju=2):
        """Initialize the SolverWrapper."""
        caffe.set_device(0)
        caffe.set_mode_gpu()
        self.netSize = netSize
        self.class_size = class_size
        self.heat_map_ratio = 0.25  #####
        self.class_adju = class_adju
        self.data_folder = 'data/'  #####
        self.im_target_size = 227  #####
        self.max_grow_iter = 300
        self.meanarr = h2._load_mean_binaryproto(
            fileName=meanfile, im_target_size=self.im_target_size)

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

        # load list of all pickel files
        visu_file_s = [f for f in listdir(visu_all_analyse_dir)
                       if (isfile(join(visu_all_analyse_dir, f)) and
                           os.path.splitext(f)[1] == '.pickle')]

        # load sample file
        visu_file = visu_all_analyse_dir + visu_file_s[0]
        with open(visu_file) as f:
            im_name, class_index, tech_s, size_patch_s, dilate_iteration_s, heat_map_occ_s, heat_map_raw_occ_s, heat_map_grad_s, heat_map_raw_grad_s = pickle.load(
                f)

        # different ways to combine the images
        combine_tech_s = ["blur", "black"]
        for p in range(len(combine_tech_s)):
            combine_tech = combine_tech_s[p]
            o_rel_inc = np.zeros((len(visu_file_s), len(size_patch_s)))
            g_rel_inc = np.zeros((len(visu_file_s), len(dilate_iteration_s)))
            less_per = np.zeros((len(visu_file_s), 1))
            o_req_mask_percent = np.zeros(
                (len(visu_file_s), len(size_patch_s)))
            g_req_mask_percent = np.zeros(
                (len(visu_file_s), len(dilate_iteration_s)))
            o_req_dilate_iter = np.zeros(
                (len(visu_file_s), len(size_patch_s))) - 2
            g_req_dilate_iter = np.zeros(
                (len(visu_file_s), len(dilate_iteration_s))) - 2
            images = []
            # for all files
            for i in range(len(visu_file_s)):
                visu_file = visu_all_analyse_dir + visu_file_s[i]
                # load files
                with open(visu_file) as f:
                    im_name, class_index, tech_s, size_patch_s, dilate_iteration_s, heat_map_occ_s_25, heat_map_raw_occ_s, heat_map_grad_s_25, heat_map_raw_grad_s = pickle.load(
                        f)

                #import IPython
                #IPython.embed()

                # load image and find prob
                images.append(im_name)
                img = h2._load_image(
                    img_name=self.data_folder + im_name,
                    im_target_size=self.im_target_size)
                original_blob = h2._get_image_blob_from_image(
                    img, self.meanarr, self.im_target_size)
                orig_prob, all_prob = self.get_prediction_prob(original_blob,
                                                               class_index)
                #print "original prob", im_name, class_index, orig_prob#, all_prob

                # modify image according to the technique and find prediction
                if combine_tech == "blur":
                    size = 17
                    kernel = np.ones((size, size), np.float32) / (size * size)
                    mod_img = cv2.filter2D(img, -1, kernel)
                    mod_blob = h2._get_image_blob_from_image(
                        mod_img, self.meanarr, self.im_target_size)
                    mod_prob, all_prob = self.get_prediction_prob(mod_blob,
                                                                  class_index)
                elif combine_tech == "black":
                    mod_img = np.zeros(img.shape)
                    mod_prob = 0.0
                    all_prob = np.array([0, 0])
                else:
                    print "not implemented"
                    assert 1 == 2
                print combine_tech, orig_prob, class_index, mod_prob, all_prob.argmax(
                ), class_index == all_prob.argmax()

                # combine modified and original image-- ignore is the difference is
                # less than 5 percent
                # TODO is this necessary
                if orig_prob - mod_prob > 0.0:
                    # using occlusion visualization
                    for k in range(len(size_patch_s)):
                        heat_map_o = h2._get_mask_from_raw_map(
                            heat_map_raw_occ_s[k], self.heat_map_ratio)
                        pred_class_index, c_prob, c_all_prob = self.combine_and_predict(
                            mod_img, heat_mask_o, img, class_index)
                        # find relative increase in network confidence
                        rel_inc = 100 * (c_prob - mod_prob) / (
                            orig_prob - mod_prob + 0.001)
                        #print "occ", size_patch_s[k], c_prob, c_all_prob, rel_inc
                        o_rel_inc[i, k] = rel_inc
                        # put this inside a function too?
                        #TODO save final modified image and mask,
                        if pred_class_index == class_index:
                            req_dilate_iter = 0
                            req_percent = h2._find_percentage_mask(heat_mask_o)
                        else:
                            req_dilate_iter, req_heat_mask_o = self.grow_till_confident(
                                mod_img, heat_mask_o, img, class_index)
                            req_percent = h2._find_percentage_mask(
                                req_heat_mask_o)
                        o_req_mask_percent[i, k] = req_percent
                        o_req_dilate_iter[i, k] = req_dilate_iter

                        #h2.plot_images(img, mod_img, c_img)

                    # using gradient visualization
                    for k in range(len(dilate_iteration_s)):
                        heat_map_g = h2._get_mask_from_raw_map(
                            heat_map_raw_grad_s[k], self.heat_map_ratio)
                        pred_class_index, c_prob, c_all_prob = self.combine_and_predict(
                            mod_img, heat_mask_g, img, class_index)

                        rel_inc = 100 * (c_prob - mod_prob) / (
                            orig_prob - mod_prob + 0.001)
                        #print "grad", dilate_iteration_s[k], c_prob, c_all_prob, rel_inc
                        g_rel_inc[i, k] = rel_inc
                        # put this inside a function too?
                        if pred_class_index == class_index:
                            req_dilate_iter = 0
                            req_percent = h2._find_percentage_mask(heat_mask_g)
                        else:
                            req_dilate_iter, req_heat_mask_g = self.grow_till_confident(
                                mod_img, heat_mask_g, img, class_index)
                            req_percent = h2._find_percentage_mask(
                                req_heat_mask_g)
                        g_req_mask_percent[i, k] = req_percent
                        g_req_dilate_iter[i, k] = req_dilate_iter
                else:
                    less_per[i] = 1

            # ignore the if using visualization decreased the
            # network confidence
            #print o_rel_inc.astype(int)
            #print g_rel_inc.astype(int)
            #print less_per
            o_rel_inc[o_rel_inc < 0] = 0
            g_rel_inc[g_rel_inc < 0] = 0
            o_fre = np.zeros(o_rel_inc.shape)
            o_fre[o_rel_inc > 0] = 1
            g_fre = np.zeros(g_rel_inc.shape)
            g_fre[g_rel_inc > 0] = 1
            o_rel_inc_avg = o_rel_inc.sum(axis=0) / o_fre.sum(axis=0)
            g_rel_inc_avg = g_rel_inc.sum(axis=0) / g_fre.sum(axis=0)

            # finding best performing configs
            print combine_tech
            print "occ sum", o_rel_inc_avg, "best perfomance", size_patch_s[
                o_rel_inc_avg.argmax()]
            print "grd_sum", g_rel_inc_avg, "best perfomance", dilate_iteration_s[
                g_rel_inc_avg.argmax()]
            import IPython
            IPython.embed()

    def combine_and_predict(self, mod_img, heat_mask_o, img, class_index):
        c_img = h2._combine_images(mod_img, heat_mask_o, img)
        c_blob = h2._get_image_blob_from_image(c_img, self.meanarr,
                                               self.im_target_size)
        c_prob, c_all_prob = self.get_prediction_prob(c_blob, class_index)
        pred_class_index = c_all_prob.argmax()
        return pred_class_index, c_prob, c_all_prob

    def grow_till_confident(self, m_img, ori_mask, img_o, class_index):
        kernel = np.ones((3, 3), np.uint8)
        mask_now = ori_mask
        _iter = 0
        while (1):
            _iter += 1
            mask_now = cv2.dilate(mask_now, kernel, iterations=1)
            pred_class_index, c_prob, c_all_prob = self.combine_and_predict(
                m_img, mask_now, img_o, class_index)
            if pred_class_index == class_index:
                req_iter = _iter
                break
            if _iter > self.max_grow_iter:
                req_iter = -1
                break
        return req_iter, mask_now

    def get_prediction_prob(self, image_blob, class_index_n):
        # finds the network confidence
        self.siameseTestNet.forward(data=image_blob.astype(
            np.float32, copy=True))
        #p = self.siameseTestNet.blobs['prob'].data[0].copy()
        #prob1 = p[class_index_n]
        p = self.siameseTestNet.blobs['fc9_f'].data[0].copy()
        prob1 = h2._get_prob(p, class_index_n)
        return prob1, p


def analyseNet(pretrainedSiameseModel,
               testProto,
               fileName_test_visu,
               viz_tech=None,
               analyse_all_visualizations=0,
               visu_all_analyse_dir=None,
               meanfile='',
               netSize=1000):

    sw = AnalyseVisualizations(
        pretrainedSiameseModel=pretrainedSiameseModel,
        testProto=testProto,
        analysis=analyse_all_visualizations,
        tech=viz_tech,
        meanfile=meanfile,
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
