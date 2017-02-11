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
from ipdb import set_trace as debug


class SiameseTrainWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, pretrainedSiameseModel,
                 pretrained_model, pretrained_model_proto, testProto, train,
                 testProto1, netSize, visu, tech, netName, heat_mask_ratio,
                 final_layer, meanfile, im_target_size, data_folder,
                 class_size, class_adju):
        """Initialize the SolverWrapper."""
        caffe.set_device(0)
        caffe.set_mode_gpu()

        self.im_target_size = im_target_size  #227  #####
        self.heat_mask_ratio = heat_mask_ratio  #####
        self.final_layer = final_layer
        self.train = train
        self.netSize = netSize
        self.netName = netName
        self.class_size = class_size
        self.class_adju = class_adju
        self.data_folder = data_folder  #'data/'
        self.im_target_size = im_target_size
        self.viz_tech = tech
        self.meanarr = h2._load_mean_binaryproto(
            fileName=meanfile, im_target_size=self.im_target_size)

        if self.train == 1:

            self.solver = caffe.SGDSolver(solver_prototxt)
            if pretrainedSiameseModel is not None:
                print('Loading pretrained model '
                      'weights from {:s}').format(pretrainedSiameseModel)
                self.solver.net.copy_from(pretrainedSiameseModel)

            elif pretrained_model is not None:
                self.solver.net.copy_from(pretrained_model)

            else:
                print('Initializing completely from scratch .... really ?')

            self.solver.test_nets[0].share_with(self.solver.net)
        elif visu == 1:
            assert testProto != None
            assert pretrainedSiameseModel != None
            if self.viz_tech == 'occ' or self.viz_tech == 'both':
                self.siameseTestNet = caffe.Net(
                    testProto, pretrainedSiameseModel, caffe.TEST)
            if self.viz_tech == 'grad':
                self.siameseTestNet_grad = caffe.Net(
                    testProto, pretrainedSiameseModel, caffe.TEST)
            if self.viz_tech == 'both':
                assert testProto1 != None
                self.siameseTestNet_grad = caffe.Net(
                    testProto1, pretrainedSiameseModel, caffe.TEST)
        else:
            assert testProto != None
            assert pretrainedSiameseModel != None
            self.siameseTestNet = caffe.Net(testProto, pretrainedSiameseModel,
                                            caffe.TEST)
            print "testing Not implemented"

    def trainTest(self):
        num_data_epoch_train = 540
        num_data_epoch_verification = 60
        num_data_epoch_test = 30
        cont_weight = 0.5
        soft_weight = 0.25
        test_skip = 1
        val_skip = 1

        plot_data_d_v = np.zeros((0, 2))
        plot_data_s_v = np.zeros((0, 2))
        plot_data_id_l1_v = np.zeros((0, 2))
        plot_data_all_v = np.zeros((0, 2))
        plot_data_d = np.zeros((0, 2))
        plot_data_s = np.zeros((0, 2))
        plot_data_id_l1 = np.zeros((0, 2))
        plot_data_all = np.zeros((0, 2))
        plot_acc1 = np.zeros((0, 2))
        plot_acc2 = np.zeros((0, 2))
        tStamp = '-Timestamp-{:%Y-%m-%d-%H:%M:%S}'.format(
            datetime.datetime.now())
        print tStamp
        plt.ion()

        def get_scaled_losses(cont_loss, soft_loss1, soft_loss2):
            return cont_weight * cont_loss, soft_weight * soft_loss1, soft_weight * soft_loss2

        print "training source", self.solver.net.layers[0].source_file
        print "verification source", self.solver.test_nets[0].layers[
            0].source_file
        print "testing source", self.solver.test_nets[1].layers[0].source_file
        save = 1
        try:
            for k in range(150):
                disLoss = 0
                simLoss = 0
                simC = 0
                disC = 0
                lossId1s = 0
                #lossId2s = 0
                lossAll = 0
                lossCos = 0
                for i in range(num_data_epoch_train):
                    self.solver.step(1)
                    lossCo, lossId1, lossId2 = get_scaled_losses(
                        self.solver.net.blobs['cont_loss'].data,
                        self.solver.net.blobs['softmax_loss_1'].data,
                        self.solver.net.blobs['softmax_loss_2'].data)
                    if self.solver.net.blobs['sim'].data == 1:
                        simC += 1
                        simLoss += lossCo
                        plot_data_s = np.vstack(
                            (plot_data_s, [k + 0.5, lossCo]))
                    else:
                        disC += 1
                        disLoss += lossCo
                        plot_data_d = np.vstack((plot_data_d, [k, lossCo]))
                    lossId1s += lossId1 + lossId2
                    lossAll += lossId1 + lossId2 + lossCo
                    lossCos += lossCo
                    #lossId2s += lossId2

                plot_data_id_l1 = np.vstack(
                    (plot_data_id_l1, [k, lossId1s / num_data_epoch_train]))
                plot_data_all = np.vstack(
                    (plot_data_all, [k, lossAll / num_data_epoch_train]))
                simLoss = simLoss / (simC + 0.1)
                disLoss = disLoss / (disC + 0.1)
                print k, "cont net loss", simLoss, disLoss, "cont loss ", lossCos / num_data_epoch_train, "Id net loss", lossId1s / num_data_epoch_train, lossAll / num_data_epoch_train
                #print simC, disC

                if k % val_skip == 0:
                    acc1 = np.zeros(self.class_size)
                    acc2 = np.zeros(self.class_size)
                    fre1 = np.zeros(self.class_size)
                    fre2 = np.zeros(self.class_size)
                    confusion_dis = np.zeros(
                        (self.class_size, self.class_size))
                    lossId1s = 0
                    #lossId2s = 0
                    lossAll = 0
                    simC = 0
                    disC = 0
                    simLoss_v = 0
                    disLoss_v = 0
                    for i in range(num_data_epoch_verification):
                        #print self.solver.test_nets[0].layers[0].source_file
                        loss1 = self.solver.test_nets[0].forward()
                        #print i, loss1, loss1['sim'], loss1['euc_dist']
                        lossCo, lossId1, lossId2 = get_scaled_losses(
                            self.solver.test_nets[0].blobs['cont_loss'].data,
                            self.solver.test_nets[0].blobs[
                                'softmax_loss_1'].data, self.solver.test_nets[
                                    0].blobs['softmax_loss_2'].data)
                        similarity = self.solver.test_nets[0].blobs[
                            'sim'].data[0]
                        if similarity == 1:
                            simC += 1
                            simLoss_v += loss1['euc_dist']
                            plot_data_s_v = np.vstack(
                                (plot_data_s_v, [k + 0.5, loss1['euc_dist']]))
                        else:
                            disC += 1
                            disLoss_v += loss1['euc_dist']
                            plot_data_d_v = np.vstack(
                                (plot_data_d_v, [k, loss1['euc_dist']]))
                        lossId1s += lossId1 + lossId2
                        lossAll += lossId1 + lossId2 + lossCo
                        #lossId2s += lossId2

                        id1 = self.solver.test_nets[0].layers[0].m_batch_1[0][
                            1] - self.class_adju
                        id2 = self.solver.test_nets[0].layers[0].m_batch_2[0][
                            1] - self.class_adju
                        confusion_dis[id1, id2] += loss1['euc_dist']
                        acc1[int(self.solver.test_nets[0].blobs['label1'].data[
                            0])] += self.solver.test_nets[0].blobs[
                                'accuracy1'].data
                        fre1[int(self.solver.test_nets[0].blobs['label1'].data[
                            0])] += 1
                        acc2[int(self.solver.test_nets[0].blobs['label2'].data[
                            0])] += self.solver.test_nets[0].blobs[
                                'accuracy2'].data
                        fre2[int(self.solver.test_nets[0].blobs['label2'].data[
                            0])] += 1
                    netacc = (acc1.sum() + acc2.sum()) / (
                        fre1.sum() + fre2.sum())
                    netacc1 = (acc1.sum()) / (fre1.sum())
                    netacc2 = (acc2.sum()) / (fre2.sum())
                    acc1 = acc1 / (fre1 + 0.00000001)
                    acc2 = acc2 / (fre2 + 0.00000001)

                    plot_data_id_l1_v = np.vstack(
                        (plot_data_id_l1_v,
                         [k, lossId1s / num_data_epoch_train]))
                    plot_data_all_v = np.vstack(
                        (plot_data_all_v, [k, lossAll / num_data_epoch_train]))
                    plot_acc1 = np.vstack((plot_acc1, [k, netacc1]))
                    plot_acc2 = np.vstack((plot_acc2, [k, netacc2]))
                    print "** verification net loss ", simLoss_v / (
                        simC + 0.1
                    ), disLoss_v / (
                        disC + 0.1
                    ), "id loss", lossId1s, "net loss", lossAll, netacc * 100
                    #print simC, disC
                    #print confusion_dis
                    #print acc1, acc2

                if k % test_skip == 0:
                    acc1_t = np.zeros(self.class_size)
                    fre1_t = np.zeros(self.class_size)
                    confusion_dis_t = np.zeros(
                        (self.class_size, self.class_size))
                    for i in range(num_data_epoch_test):
                        loss1 = self.solver.test_nets[1].forward()
                        id1 = self.solver.test_nets[1].layers[0].m_batch_1[0][
                            1] - self.class_adju
                        id2 = self.solver.test_nets[1].layers[0].m_batch_2[0][
                            1] - self.class_adju
                        confusion_dis_t[id1, id2] += loss1['euc_dist']
                        # we worry only about net's first bracnch because all the
                        # labels of input in the second branch will be same
                        # as designed for convinience. This can be noticed in
                        # confusion matrix
                        acc1_t[int(self.solver.test_nets[1].blobs[
                            'label1'].data[0])] += self.solver.test_nets[
                                1].blobs['accuracy1'].data
                        #if self.solver.test_nets[1].blobs['accuracy1'].data == 0:
                        #    import IPython
                        #    IPython.embed()
                        fre1_t[int(self.solver.test_nets[1].blobs[
                            'label1'].data[0])] += 1
                        #print i, int(self.solver.test_nets[1].blobs['label1'].data[
                        #    0])
                        #acc1_t[int(self.solver.test_nets[1].blobs['label2'].data[
                        #    0])] += self.solver.test_nets[1].blobs[
                        #        'accuracy2'].data
                        #fre1_t[int(self.solver.test_nets[1].blobs['label2'].data[
                        #    0])] += 1
                    netacc_t = (acc1_t.sum()) / (fre1_t.sum())
                    acc1_t = acc1_t / (fre1_t + 0.00000001)

                    print "**** testing net ", netacc_t * 100
                    #print confusion_dis_t
                    #print acc1_t, fre1_t
                #import IPython
                #IPython.embed()

                if k % 2 == 0 and save == 1:
                    preName = 'modifiedNetResults/' + 'Modified-netsize-' + str(
                        self.netSize) + '-epoch-' + str(
                            k) + '-tstamp-' + tStamp
                    self.solver.net.save(preName + '-net.caffemodel')

                plt.figure(1)
                plt.xlim(-0.5, 150)
                plt.title(str(self.netSize) + " train errors")
                plt.plot(plot_data_s[:, 0], plot_data_s[:, 1], 'r.')
                plt.plot(plot_data_d[:, 0], plot_data_d[:, 1], 'b.')
                plt.pause(0.05)

                plt.figure(2)
                plt.clf()
                plt.xlim(-0.5, 150)
                plt.title(str(self.netSize) + " train and validation errors")
                plt.plot(
                    plot_data_id_l1[:, 0],
                    plot_data_id_l1[:, 1],
                    'r',
                    label="train id")
                plt.plot(
                    plot_data_all[:, 0],
                    plot_data_all[:, 1],
                    'b',
                    label="train all")
                plt.plot(
                    plot_data_id_l1_v[:, 0],
                    plot_data_id_l1_v[:, 1],
                    'g',
                    label="test id")
                plt.plot(
                    plot_data_all_v[:, 0],
                    plot_data_all_v[:, 1],
                    'k',
                    label="test all")
                plt.legend(bbox_to_anchor=(.80, .8), loc=2, borderaxespad=0.)
                plt.pause(0.05)

                plt.figure(3)
                plt.xlim(-0.5, 150)
                plt.title(str(self.netSize) + " verification accuracy")
                plt.plot(k, netacc1, 'r.')
                plt.plot(k, netacc2, 'b.')
                plt.pause(0.05)

                plt.figure(4)
                plt.xlim(-0.5, 150)
                plt.title(str(self.netSize) + " train cont losses")
                plt.plot(k, simLoss, 'r.')
                plt.plot(k, disLoss, 'b.')
                plt.pause(0.05)

        except KeyboardInterrupt:
            pass

        preName = 'modifiedNetResults/' + 'Modified-netsize-' + str(
            self.netSize) + '-epoch-' + str(k) + '-tstamp-' + tStamp
        plt.ioff()

        if save == 1:
            plt.figure(1).savefig(preName + '-train-d-error.png')
            plt.figure(2).savefig(preName + '-train-valid-error.png')
            #plt.figure(3).savefig(preName + '-test-acc.png')
            plt.figure(4).savefig(preName + '-train-cont.png')
            self.solver.net.save(preName + '-net-final.caffemodel')
        plt.close('all')

        import IPython
        IPython.embed()

    def visualize_all(self, fileName, tech, compare, visu_all_save_dir, save,
                      outputLayerName, outputBlobName, topLayerName,
                      topBlobName, secondTopLayerName, secondTopBlobName,
                      save_img):
        ''' Visualizing all and saving
        '''
        #tStamp = '-Timestamp-{:%Y-%m-%d-%H:%M:%S}'.format(
        #    datetime.datetime.now())
        tStamp = ''
        f = open(fileName)
        lines = [line.rstrip('\n') for line in f]
        imageDict = {}
        imlist = []

        stride = 10
        highlighted_ratio = self.heat_mask_ratio
        heat_map_occ_s = {}
        heat_map_raw_occ_s = {}
        heat_map_exci_s = {}
        heat_map_raw_exci_s = {}
        heat_map_grad_s = {}
        heat_map_raw_grad_s = {}

        size_patch_s = [10, 50, 100]
        dilate_iteration_s = [0, 2, 5]
        #tech_s = ['occ', 'grad']
        tech_s = ['occ', 'grad', 'exci']
        #tech_s = ['exci']

        for i in lines:
            temp = i.split(' ')
            imageDict[temp[0]] = int(temp[1]) - self.class_adju
            imlist.append(temp[0])

        for i in range(len(imlist)):
            im1 = i
            print 'generating visu-', imageDict[imlist[im1]], imlist[
                im1], 'using occ patch', size_patch_s, ' grad with dilate iter', dilate_iteration_s

            if 'occ' in tech_s:
                #occlude im1
                for i in range(len(size_patch_s)):
                    size_patch = size_patch_s[i]

                    im_gen_occ, heat_map_occ, heat_map_raw_occ = self.generate_heat_map_softmax(
                        imageDict,
                        imlist,
                        im1,
                        size_patch,
                        stride,
                        ratio=highlighted_ratio)
                    heat_map_occ_s[i] = heat_map_occ
                    heat_map_raw_occ_s[i] = heat_map_raw_occ
                    preName = visu_all_save_dir + '_occ/' + imlist[
                        im1][:-4] + '-' + str(size_patch) + '-' + str(
                            stride) + '-' + '-M-nSize-' + str(
                                self.netSize) + '-tstamp-' + tStamp
                    if save_img == 1:
                        cv2.imwrite(preName + '.png', im_gen_occ)
            else:
                heat_map_occ_s[i] = heat_map_raw_occ_s[i] = None

            if 'grad' in tech_s:
                # image specific class saliency map
                for i in range(len(dilate_iteration_s)):
                    #the loop can be removed
                    dilate_iteration = dilate_iteration_s[i]

                    im_gen_grad, heat_map_grad, heat_map_raw_grad = self.generate_heat_map_gradients(
                        imageDict,
                        imlist,
                        im1,
                        ratio=highlighted_ratio,
                        dilate_iterations=dilate_iteration)
                    heat_map_grad_s[i] = heat_map_grad
                    heat_map_raw_grad_s[i] = heat_map_raw_grad
                    preName = visu_all_save_dir + '_grad/' + imlist[
                        im1][:-4] + '-itera-' + str(
                            dilate_iteration) + '-M-nSize-' + str(
                                self.netSize) + '-tstamp-' + tStamp
                    if save_img == 1:
                        cv2.imwrite(preName + '.png', im_gen_grad)
            else:
                heat_map_grad_s[i] = heat_map_raw_grad_s[i] = None

            if 'exci' in tech_s:
                # use excitation BP
                for i in range(1):

                    im_gen_exci, heat_map_exci, heat_map_raw_exci = self.generate_heat_map_excitation(
                        imageDict,
                        imlist,
                        im1,
                        outputLayerName,
                        outputBlobName,
                        topLayerName,
                        topBlobName,
                        secondTopLayerName,
                        secondTopBlobName,
                        ratio=highlighted_ratio)

                    heat_map_exci_s[i] = heat_map_exci
                    heat_map_raw_exci_s[i] = heat_map_raw_exci
                    preName = visu_all_save_dir + '_exci/' + imlist[
                        im1][:-4] + '-M-nSize-' + str(
                            self.netSize) + '-tstamp-' + tStamp
                    if save_img == 1:
                        cv2.imwrite(preName + '.png', im_gen_exci)
            else:
                heat_map_exci_s[i] = heat_map_raw_exci_s[i] = None

            preName = visu_all_save_dir + '/' + imlist[
                im1][:-4] + '--M-nSize-' + str(
                    self.netSize) + '-tstamp-' + tStamp + '--visualizations'

            if save == 1:
                with open(preName + '.pickle', 'w') as f:
                    pickle.dump([imlist[im1],
                                 imageDict[imlist[im1]],
                                 tech_s,
                                 size_patch_s,
                                 dilate_iteration_s,
                                 heat_map_raw_occ_s,
                                 heat_map_raw_grad_s,
                                 heat_map_raw_exci_s, ], f)

    def visualize(self, fileName, tech, compare):
        ''' Visualizing using gray occlusion patches or gradients of input image
        '''
        #tStamp = '-Timestamp-{:%Y-%m-%d-%H:%M:%S}'.format(
        #    datetime.datetime.now())
        tStamp = ''
        f = open(fileName)
        lines = [line.rstrip('\n') for line in f]
        imageDict = {}
        imlist = []
        size_patch = 1
        stride = 1
        highlighted_ratio = 0.25
        gradient_dilate_iterations = 10

        for i in lines:
            temp = i.split(' ')
            imageDict[temp[0]] = int(temp[1]) - self.class_adju
            imlist.append(temp[0])
        if tech == 'both':
            self.ov_lap_heat_map = np.zeros(len(imlist))
            self.class_ov_lap_heat_map = np.zeros(self.class_size)
            self.class_fre = np.zeros(self.class_size)
        if tech == 'both' or tech == 'grad':
            dilate_iteration = gradient_dilate_iterations
        for i in range(len(imlist)):
            im1 = i
            save = 1
            if tech == 'occ':
                #occlude im1 and get map on im1
                print 'generating heat map for ', imageDict[imlist[
                    im1]], imlist[im1], 'using occluding patch'

                im_gen_occ, heat_map_occ, heat_map_raw_occ = self.generate_heat_map_softmax(
                    imageDict,
                    imlist,
                    im1,
                    size_patch,
                    stride,
                    ratio=highlighted_ratio)
                preName = 'modifiedNetResults_visu_occ/' + imlist[
                    im1][:-4] + '-' + str(size_patch) + '-' + str(
                        stride) + '-' + '-M-nSize-' + str(
                            self.netSize) + '-tstamp-' + tStamp
                cv2.imwrite(preName + '.png', im_gen_occ)
            elif tech == 'grad':
                #using graidents wrt input to visualize
                print 'generating heat map for ', imageDict[imlist[
                    im1]], imlist[im1], 'using gradients wrt image'

                im_gen_grad, heat_map_grad, heat_map_raw_grad = self.generate_heat_map_gradients(
                    imageDict,
                    imlist,
                    im1,
                    ratio=highlighted_ratio,
                    dilate_iterations=dilate_iteration)
                preName = 'modifiedNetResults_visu_grad/' + imlist[
                    im1][:-4] + '-itera-' + str(
                        self.iterations) + '-M-nSize-' + str(
                            self.netSize) + '-tstamp-' + tStamp
                cv2.imwrite(preName + '.png', im_gen_grad)
            elif tech == 'both':
                #use both gradient and occlusion path to visualize
                print 'generating heat map for ', imageDict[imlist[im1]], imlist[
                    im1], 'using both gradients wrt image and occlusion patches'

                im_gen_occ, heat_map_occ, heat_map_raw_occ = self.generate_heat_map_softmax(
                    imageDict,
                    imlist,
                    im1,
                    size_patch,
                    stride,
                    ratio=highlighted_ratio)
                preName_occ = 'modifiedNetResults_visu_occ/' + imlist[
                    im1][:-4] + '-' + str(size_patch) + '-' + str(
                        stride) + '-' + '-M-nSize-' + str(
                            self.netSize) + '-tstamp-' + tStamp

                im_gen_grad, heat_map_grad, heat_map_raw_grad = self.generate_heat_map_gradients(
                    imageDict,
                    imlist,
                    im1,
                    ratio=highlighted_ratio,
                    dilate_iterations=dilate_iteration)
                preName_grad = 'modifiedNetResults_visu_grad/' + imlist[
                    im1][:-4] + '-itera-' + str(
                        self.iterations) + '-M-nSize-' + str(
                            self.netSize) + '-tstamp-' + tStamp

                save = 0
                if save == 1:
                    cv2.imwrite(preName_grad + '.png', im_gen_grad)
                save = 1
                if save == 1:
                    cv2.imwrite(preName_occ + '.png', im_gen_occ)
                #code for comparision
                if compare == 1:
                    self.ov_lap_heat_map[i] = h2._analyse_heat_maps(
                        heat_map_occ, heat_map_grad)
                    print "overlap percentage", self.ov_lap_heat_map[i]
                    self.class_fre[imageDict[imlist[im1]]] += 1
                    self.class_ov_lap_heat_map[imageDict[imlist[
                        im1]]] += self.ov_lap_heat_map[i]

        if compare == 1:
            print self.ov_lap_heat_map
            print self.class_ov_lap_heat_map / (self.class_fre + 0.000001)
            #import IPython
            #IPython.embed()

    def generate_heat_map_excitation(
            self, imageDict, imlist, im1, outputLayerName, outputBlobName,
            topLayerName, topBlobName, secondTopLayerName, secondTopBlobName,
            ratio):
        im = h2._load_image(
            img_name=self.data_folder + imlist[im1],
            im_target_size=self.im_target_size)
        blobs = {'data': None}
        blobs['data'] = h2._get_image_blob_from_image(
            im=im, meanarr=self.meanarr, im_target_size=self.im_target_size)
        heat_map = np.zeros((self.im_target_size, self.im_target_size))
        label_index = imageDict[imlist[im1]]
        tagID = label_index
        #TODO Read paper and decide on what topblob to use

        pred = self.siameseTestNet.forward(
            data=blobs['data'].astype(
                np.float32, copy=True), end=topLayerName)

        #do excitation BP
        caffe.set_mode_eb_gpu()
        self.siameseTestNet.blobs[topBlobName].diff[0][...] = 0
        #TODO verify this
        #self.siameseTestNet.blobs[topBlobName].diff[0][tagID] = np.exp(self.siameseTestNet.blobs[topBlobName].data[0][tagID].copy())
        #self.siameseTestNet.blobs[topBlobName].diff[0][tagID] /= self.siameseTestNet.blobs[topBlobName].diff[0][tagID].sum()
        self.siameseTestNet.blobs[topBlobName].diff[0][tagID] = 1
        # invert the top layer weights
        self.siameseTestNet.params[topLayerName][0].data[...] *= -1
        out = self.siameseTestNet.backward(
            start=topLayerName, end=secondTopLayerName)
        buff = self.siameseTestNet.blobs[secondTopBlobName].diff.copy()
        # invert back
        self.siameseTestNet.params[topLayerName][0].data[...] *= -1
        out = self.siameseTestNet.backward(
            start=topLayerName, end=secondTopLayerName)
        # compute the contrastive signal
        self.siameseTestNet.blobs[secondTopBlobName].diff[...] -= buff
        # generate heat map
        out = self.siameseTestNet.backward(
            start=secondTopLayerName, end=outputLayerName)
        attMap = np.maximum(
            self.siameseTestNet.blobs[outputBlobName].diff[0].sum(0), 0)

        caffe.set_mode_gpu()

        print "heat map size", attMap.shape
        heat_map = h2._get_exci_final_map(attMap.copy(), self.im_target_size)
        heat_map_raw = (heat_map - heat_map.min()) / (
            heat_map.max() - heat_map.min())
        heat_map = heat_map_raw.copy()
        threshold = h2._find_threshold(heat_map, ratio=ratio)
        heat_map[heat_map < threshold] = 0
        heat_map[heat_map >= threshold] = 1

        temp = np.sum(im, axis=2) / 3.0
        im[:, :, 2] = temp
        im[:, :, 1] = temp
        im[:, :, 0] = temp
        im[:, :, 2] += heat_map * 100

        im = h2._round_image(im)
        return im.astype(np.uint8), heat_map.astype(np.uint8), heat_map_raw

    def generate_heat_map_softmax(self, imageDict, imlist, im1, size_patch,
                                  stride, ratio):
        im = h2._load_image(
            img_name=self.data_folder + imlist[im1],
            im_target_size=self.im_target_size)
        blobs = {'data': None}
        heat_map = np.zeros((self.im_target_size, self.im_target_size))

        im_size1 = im.shape[0]
        im_size2 = im.shape[1]
        cR = -size_patch + 1
        i = 0
        while im_size1 - 1 >= cR - 1:
            cC = -size_patch + 1
            while im_size2 - 1 > cC - 1:
                i = i + 1
                l_blobs_im1, l_occ_map = h2._get_occluded_image_blob(
                    im=im,
                    size_patch=size_patch,
                    cR=cR,
                    cC=cC,
                    meanarr=self.meanarr,
                    im_target_size=self.im_target_size)
                blobs['data'] = l_blobs_im1
                blobs_out1 = self.siameseTestNet.forward(
                    data=blobs['data'].astype(
                        np.float32, copy=True))

                p = self.siameseTestNet.blobs[self.final_layer].data[0].copy()
                prob1 = h2._get_prob(p, imageDict[imlist[im1]])
                heat_map += l_occ_map * prob1

                cC += stride
            cR += stride

        heat_map_o = (heat_map - heat_map.min()) / (
            heat_map.max() - heat_map.min())
        #invert the heat map
        heat_map_o = 1.0 - heat_map_o
        heat_map = heat_map_o.copy()

        threshold = h2._find_threshold(heat_map, ratio)
        heat_map[heat_map < threshold] = 0
        heat_map[heat_map >= threshold] = 1
        heat_map *= 1

        temp = np.sum(im, axis=2) / 3.0
        im[:, :, 2] = temp
        im[:, :, 1] = temp
        im[:, :, 0] = temp
        im[:, :, 2] += heat_map * 100

        im = h2._round_image(im)
        return im.astype(np.uint8), heat_map.astype(np.uint8), heat_map_o

    def generate_heat_map_gradients(self, imageDict, imlist, im1, ratio,
                                    dilate_iterations):
        im = h2._load_image(
            img_name=self.data_folder + imlist[im1],
            im_target_size=self.im_target_size)
        blobs = {'data': None}
        blobs['data'] = h2._get_image_blob_from_image(
            im=im, meanarr=self.meanarr, im_target_size=self.im_target_size)
        label_index = imageDict[imlist[im1]]
        caffeLabel = np.zeros((1, self.class_size))
        caffeLabel[0, label_index] = 1

        pred = self.siameseTestNet_grad.forward(data=blobs['data'].astype(
            np.float32, copy=True))
        bw = self.siameseTestNet_grad.backward(
            **{self.siameseTestNet_grad.outputs[0]: caffeLabel})  #
        diff = bw['data'].copy()

        # Find the saliency map as described in the paper. Normalize the map and assign it to variabe "saliency"
        diff = np.abs(diff)
        diff -= diff.min()
        diff /= diff.max()
        diff_sq = np.squeeze(diff)
        saliency = np.amax(diff_sq, axis=0)

        heat_map_raw = saliency.copy()
        kernel = np.ones((3, 3), np.uint8)
        if dilate_iterations > 0:
            heat_map_raw = cv2.dilate(
                heat_map_raw, kernel, iterations=dilate_iterations)
        #else:
        heat_map = heat_map_raw.copy()

        threshold = h2._find_threshold(heat_map, ratio=ratio)
        heat_map[heat_map < threshold] = 0
        heat_map[heat_map >= threshold] = 1

        temp = np.sum(im, axis=2) / 3.0
        im[:, :, 2] = temp
        im[:, :, 1] = temp
        im[:, :, 0] = temp
        im[:, :, 2] += heat_map * 100

        im = h2._round_image(im)
        return im.astype(np.uint8), heat_map.astype(np.uint8), heat_map_raw


def siameseTrainer(siameseSolver,
                   fileName_test_visu,
                   pretrained_model,
                   pretrainedSiameseModel,
                   testProto,
                   pretrained_model_proto,
                   train,
                   visu,
                   testProto1,
                   viz_tech,
                   visu_all,
                   visu_all_save_dir,
                   compare,
                   data_folder,
                   heat_mask_ratio,
                   final_layer,
                   net,
                   im_target_size,
                   class_size,
                   class_adju,
                   meanfile,
                   netSize,
                   save,
                   save_img,
                   outputLayerName,
                   outputBlobName,
                   topLayerName,
                   topBlobName,
                   secondTopLayerName,
                   secondTopBlobName, ):
    sw = SiameseTrainWrapper(
        solver_prototxt=siameseSolver,
        pretrainedSiameseModel=pretrainedSiameseModel,
        pretrained_model=pretrained_model,
        pretrained_model_proto=pretrained_model_proto,
        testProto=testProto,
        testProto1=testProto1,
        train=train,
        visu=visu,
        final_layer=final_layer,
        heat_mask_ratio=heat_mask_ratio,
        netName=net,
        meanfile=meanfile,
        im_target_size=im_target_size,
        data_folder=data_folder,
        class_size=class_size,
        class_adju=class_adju,
        tech=viz_tech,
        netSize=netSize)
    if train == 1:
        print "training"
        sw.trainTest()
    elif visu == 1:
        if visu_all == False:
            print 'visalizing with ', pretrainedSiameseModel
            sw.visualize(fileName_test_visu, tech=viz_tech, compare=compare)
        else:
            print 'visalizing all possible ', pretrainedSiameseModel
            sw.visualize_all(
                fileName_test_visu,
                tech=viz_tech,
                compare=compare,
                visu_all_save_dir=visu_all_save_dir,
                save=save,
                outputLayerName=outputLayerName,
                outputBlobName=outputBlobName,
                topLayerName=topLayerName,
                topBlobName=topBlobName,
                secondTopLayerName=secondTopLayerName,
                secondTopBlobName=secondTopBlobName,
                save_img=save_img)
    elif visu == 0:
        print 'testing not implemented'
        #print "testing with ", pretrainedSiameseModel
        #sw.test(fileName_test_visu)
