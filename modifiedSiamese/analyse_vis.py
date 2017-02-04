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
import ipdb
from ipdb import set_trace as debug


class AnalyseVisualizations(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, pretrainedSiameseModel, testProto, netSize, analysis,
                 tech, meanfile, im_target_size, data_folder, class_size,
                 heat_mask_ratio, class_adju, save_img, save_data, net,
                 final_layer):
        """Initialize the SolverWrapper."""
        caffe.set_device(0)
        caffe.set_mode_gpu()

        self.save_img = save_img
        self.save_data = save_data
        self.net = net
        self.final_layer = final_layer
        self.netSize = netSize
        self.class_size = class_size
        self.heat_mask_ratio = heat_mask_ratio  #####
        self.class_adju = class_adju
        self.data_folder = data_folder  #'data/'  #####
        self.im_target_size = im_target_size  #227  #####
        self.max_grow_iter = 300
        self.meanarr = h2._load_mean_binaryproto(
            fileName=meanfile, im_target_size=self.im_target_size)
        self.meanfile = meanfile

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
        save = 1

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

            o_rel_inc_fin = np.zeros((len(visu_file_s), len(size_patch_s)))
            g_rel_inc_fin = np.zeros(
                (len(visu_file_s), len(dilate_iteration_s)))
            o_rel_inc = np.zeros((len(visu_file_s), len(size_patch_s)))
            g_rel_inc = np.zeros((len(visu_file_s), len(dilate_iteration_s)))
            mod_prob_s = np.zeros((len(visu_file_s), 1))
            orig_prob_s = np.zeros((len(visu_file_s), 1))
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
            class_index_s = []
            preName_occ_visu_init = []
            preName_occ_visu_fin = []
            preName_grad_visu_init = []
            preName_grad_visu_fin = []
            preName_occ_init = []
            preName_occ_fin = []
            preName_grad_init = []
            preName_grad_fin = []
            part_name = '_' + self.net + '_' + "thres" + str(
                int(100 * self.heat_mask_ratio)) + '-' + str(
                    combine_tech) + "-" + tStamp

            # for all files
            for i in range(5):  #len(visu_file_s)):
                visu_file = visu_all_analyse_dir + visu_file_s[i]
                # load files
                with open(visu_file) as f:
                    im_name, class_index, tech_s, size_patch_s, dilate_iteration_s, heat_map_occ_s_25, heat_map_raw_occ_s, heat_map_grad_s_25, heat_map_raw_grad_s = pickle.load(
                        f)

                #import IPython
                #IPython.embed()

                # load image and find prob
                images.append(im_name)
                class_index_s.append(class_index)
                img = h2._load_image(
                    img_name=self.data_folder + im_name,
                    im_target_size=self.im_target_size)
                original_blob = h2._get_image_blob_from_image(
                    img, self.meanarr, self.im_target_size)
                orig_prob, all_prob = self.get_prediction_prob(original_blob,
                                                               class_index)

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
                #print combine_tech, orig_prob, class_index, mod_prob, all_prob.argmax(
                #), class_index == all_prob.argmax()

                # combine modified and original image-- ignore is the difference is
                # less than 5 percent
                # TODO is this necessary
                mod_prob_s[i] = mod_prob
                orig_prob_s[i] = orig_prob
                if orig_prob - mod_prob > 0.0:

                    # using occlusion visualization
                    preName1, preName2, preName3, preName4, rel_inc_1, rel_inc_2, req_percent, req_dilate_iter = self.find_mask_confidence_analysis(
                        config=size_patch_s,
                        raw_map=heat_map_raw_occ_s,
                        mask_ratio=self.heat_mask_ratio,
                        img=img,
                        mod_img=mod_img,
                        im_name=im_name,
                        class_index=class_index,
                        orig_prob=orig_prob,
                        mod_prob=mod_prob,
                        part_name=part_name,
                        tech_name='occ')

                    preName_occ_visu_init += preName1
                    preName_occ_visu_fin += preName2
                    preName_occ_init += preName3
                    preName_occ_fin += preName4
                    o_rel_inc[i, :] = rel_inc_1
                    o_rel_inc_fin[i, :] = rel_inc_2
                    o_req_mask_percent[i, :] = req_percent
                    o_req_dilate_iter[i, :] = req_dilate_iter

                    # using gradient visualization
                    preName1, preName2, preName3, preName4, rel_inc_1, rel_inc_2, req_percent, req_dilate_iter = self.find_mask_confidence_analysis(
                        config=dilate_iteration_s,
                        raw_map=heat_map_raw_grad_s,
                        mask_ratio=self.heat_mask_ratio,
                        img=img,
                        mod_img=mod_img,
                        im_name=im_name,
                        class_index=class_index,
                        orig_prob=orig_prob,
                        mod_prob=mod_prob,
                        part_name=part_name,
                        tech_name='grad')

                    preName_grad_visu_init += preName1
                    preName_grad_visu_fin += preName2
                    preName_grad_init += preName3
                    preName_grad_fin += preName4
                    g_rel_inc[i, :] = rel_inc_1
                    g_rel_inc_fin[i, :] = rel_inc_2
                    g_req_mask_percent[i, :] = req_percent
                    g_req_dilate_iter[i, :] = req_dilate_iter

                    #TODO
                    #intersection of occlusion and gradients

                    #neg occlusion with gradient

                    #neg gradient with occlusion

                else:
                    less_per[i] = 1

            # ignore the if using visualization decreased the
            # network confidence
            o_rel_inc[o_rel_inc < 0] = 0
            g_rel_inc[g_rel_inc < 0] = 0
            o_fre = np.zeros(o_rel_inc.shape)
            o_fre[o_rel_inc > 0] = 1
            g_fre = np.zeros(g_rel_inc.shape)
            g_fre[g_rel_inc > 0] = 1
            o_rel_inc_avg = o_rel_inc.sum(axis=0) / o_fre.sum(axis=0)
            g_rel_inc_avg = g_rel_inc.sum(axis=0) / g_fre.sum(axis=0)
            best_patch_size_init = size_patch_s[o_rel_inc_avg.argmax()]
            best_dilate_iter_init = dilate_iteration_s[g_rel_inc_avg.argmax()]

            # finding best performing configs
            print combine_tech
            print "occ sum", o_rel_inc_avg, "best perfomance", best_patch_size_init
            print "grd_sum", g_rel_inc_avg, "best perfomance", best_dilate_iter_init

            data = {
                'tStamp': tStamp,
                'images': images,
                'class_index_s': class_index_s,
                'heat_mask_ratio': self.heat_mask_ratio,
                'combine_tech': combine_tech,
                'best_patch_size_init': best_patch_size_init,
                'best_dilate_iter_init': best_dilate_iter_init,
                'o_rel_inc': o_rel_inc,
                'o_rel_inc_fin': o_rel_inc_fin,
                'o_req_mask_percent': o_req_mask_percent,
                'o_req_dilate_iter': o_req_dilate_iter,
                'preName_occ_init': preName_occ_init,
                'preName_occ_fin': preName_occ_fin,
                'preName_occ_visu_init': preName_occ_visu_init,
                'preName_occ_visu_fin': preName_occ_visu_fin,
                'g_rel_inc': g_rel_inc,
                'g_req_dilate_iter': g_req_dilate_iter,
                'g_req_mask_percent': g_req_mask_percent,
                'g_rel_inc_fin': g_rel_inc_fin,
                'preName_grad_init': preName_grad_init,
                'preName_grad_fin': preName_grad_fin,
                'preName_grad_visu_init': preName_grad_visu_init,
                'preName_grad_visu_fin': preName_grad_visu_fin
            }

            image_list = [
                preName_grad_init, preName_grad_fin, preName_grad_visu_init,
                preName_grad_visu_fin
            ]
            #for i in image_list:
            #    for j in i:
            #        self.run.add_artifact(j)

            #self.run.info[combine_tech] = data
            if self.save_data == 1:
                with open('analysis_results/analysis_results_' + part_name +
                          '.pickle', 'w') as f:
                    pickle.dump([data], f)

        #import IPython
        #IPython.embed()
        print tStamp

    def combine_and_predict(self, mod_img, heat_mask_o, img, class_index):
        c_img = h2._combine_images(mod_img, heat_mask_o, img)
        c_blob = h2._get_image_blob_from_image(c_img, self.meanarr,
                                               self.im_target_size)
        c_prob, c_all_prob = self.get_prediction_prob(c_blob, class_index)
        pred_class_index = c_all_prob.argmax()
        return pred_class_index, c_prob, c_all_prob, c_img

    def grow_till_confident(self, m_img, ori_mask, img_o, class_index):
        kernel = np.ones((3, 3), np.uint8)
        mask_now = ori_mask
        _iter = 0
        while (1):
            _iter += 1
            mask_now = cv2.dilate(mask_now, kernel, iterations=1)
            pred_class_index, c_prob, c_all_prob, c_img = self.combine_and_predict(
                m_img, mask_now, img_o, class_index)
            if pred_class_index == class_index:
                req_iter = _iter
                break
            if _iter > self.max_grow_iter:
                req_iter = -1
                break
        return req_iter, mask_now, c_img, c_prob, c_all_prob

    def get_prediction_prob(self, image_blob, class_index_n):
        # finds the network confidence
        self.siameseTestNet.forward(data=image_blob.astype(
            np.float32, copy=True))
        #p = self.siameseTestNet.blobs['prob'].data[0].copy()
        #prob1 = p[class_index_n]
        p = self.siameseTestNet.blobs[self.final_layer].data[0].copy()
        prob1 = h2._get_prob(p, class_index_n)
        return prob1, p

    def find_mask_confidence_analysis(self, config, raw_map, mask_ratio, img,
                                      mod_img, im_name, class_index, orig_prob,
                                      mod_prob, part_name, tech_name):
        preName1 = []
        preName2 = []
        preName3 = []
        preName4 = []
        rel_inc_1 = np.zeros(len(config))
        rel_inc_2 = np.zeros(len(config))
        req_percent = np.zeros(len(config))
        req_dilate_iter = np.zeros(len(config))

        for k in range(len(config)):
            heat_mask_o = h2._get_mask_from_raw_map(raw_map[k], mask_ratio)
            pred_class_index_init, c_prob_init, c_all_prob_init, c_img_init = self.combine_and_predict(
                mod_img, heat_mask_o, img, class_index)
            # find relative increase in network confidence
            rel_inc_1[k] = 100 * (c_prob_init - mod_prob) / (
                orig_prob - mod_prob + 0.001)
            # put this inside a function too?
            #TODO initial and final mask as pickle
            if pred_class_index_init == class_index:
                req_dilate_iter[k] = 0
                req_percent[k] = -1  #h2._find_percentage_mask(heat_mask_o)
                c_prob_fin = -1  #orig_prob
                req_heat_mask_o = np.zeros(heat_mask_o.shape)
                c_img_fin = np.zeros(c_img_init.shape)
            else:
                req_dilate_iter[
                    k], req_heat_mask_o, c_img_fin, c_prob_fin, c_all_prob_fin = self.grow_till_confident(
                        mod_img, heat_mask_o, img, class_index)
                req_percent[k] = h2._find_percentage_mask(req_heat_mask_o)
            rel_inc_2[k] = 100 * (c_prob_fin - mod_prob) / (
                orig_prob - mod_prob + 0.001)

            #debug()
            if self.save_img == 1:
                c_img_visu_init_o = h2._visu_heat_map(img.copy(), heat_mask_o)
                c_img_visu_fin_o = h2._visu_heat_map(img.copy(),
                                                     req_heat_mask_o)
                preName1.append('analysis_visu_' + tech_name + '/' +
                                im_name[:-4] + '-itera-' + str(config[
                                    k]) + '-M-nSize-' + str(self.netSize) +
                                "-initial-heat_map" + part_name + ".png")
                cv2.imwrite(preName1[-1], c_img_visu_init_o)
                preName2.append('analysis_visu_' + tech_name + '/' +
                                im_name[:-4] + '-itera-' + str(config[
                                    k]) + '-M-nSize-' + str(self.netSize) +
                                "-final-heat_map" + part_name + ".png")
                cv2.imwrite(preName2[-1], c_img_visu_fin_o)
                preName3.append('analysis_visu_' + tech_name + '/' +
                                im_name[:-4] + '-itera-' + str(config[
                                    k]) + '-M-nSize-' + str(self.netSize) +
                                "-initial" + part_name + ".png")
                cv2.imwrite(preName3[-1], c_img_init)
                preName4.append('analysis_visu_' + tech_name + '/' +
                                im_name[:-4] + '-itera-' + str(config[
                                    k]) + '-M-nSize-' + str(self.netSize) +
                                "-final" + part_name + ".png")
                cv2.imwrite(preName4[-1], c_img_fin)
        return preName1, preName2, preName3, preName4, rel_inc_1, rel_inc_2, req_percent, req_dilate_iter


def analyseNet(pretrainedSiameseModel,
               testProto,
               fileName_test_visu,
               viz_tech,
               analyse_all_visualizations,
               visu_all_analyse_dir,
               meanfile,
               heat_mask_ratio,  #heat_mask_ratio,#
               data_folder,  #data_folder,#
               im_target_size,  #im_target_size,#
               class_size,  #class_size,#
               class_adju,  #class_adju,#
               netSize,
               net,
               final_layer,  #final_layer,
               save_img,
               save_data, ):

    sw = AnalyseVisualizations(
        pretrainedSiameseModel=pretrainedSiameseModel,#
        testProto=testProto,#
        analysis=analyse_all_visualizations,
        tech=viz_tech,
        heat_mask_ratio=heat_mask_ratio,#
        data_folder=data_folder,#
        im_target_size=im_target_size,#
        meanfile=meanfile,#
        class_size=class_size,#
        class_adju=class_adju,#
        final_layer=final_layer,
        net=net,
        netSize=netSize,
        save_img=save_img,save_data=save_data
        )

    if analyse_all_visualizations == 1:
        print 'analysing all visualization'
        sw.analyse_visualizations(
            database_fileName=fileName_test_visu,
            visu_all_analyse_dir=visu_all_analyse_dir)
    else:  #if visu == 0:
        print 'testing not implemented'
        #print "testing with ", pretrainedSiameseModel
        #sw.test(fileName_test_visu)
