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


class SiameseTrainWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self,
                 solver_prototxt,
                 pretrainedSiameseModel=None,
                 pretrained_model=None,
                 pretrained_model_proto=None,
                 testProto=None,
                 train=1,
                 testProto1=None,
                 netSize=1000,
                 visu=0,
                 tech=None,
                 class_size=6,
                 class_adju=2):
        """Initialize the SolverWrapper."""
        caffe.set_device(0)
        caffe.set_mode_gpu()
        #caffe.set_mode_cpu()
        self.train = train
        self.netSize = netSize
        self.class_size = class_size
        self.class_adju = class_adju
        self.data_folder = 'data/'
        self.im_target_size = 227
        self.viz_tech = tech
        self.meanarr = h2._load_mean_binaryproto(
            fileName='placesOriginalModel/places205CNN_mean.binaryproto',
            im_target_size=self.im_target_size)

        if self.train == 1:
            import IPython
            IPython.embed()

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
        #self.solver.test_nets[0].forward()
        #self.solver.net.forward()
        #self.solver.test_nets[0].blobs['conv1'].data[0,0,1,1:5]
        #self.solver.net.blobs['conv1'].data[0,0,1,1:5]
        #print self.solver.net.params['conv1'][0].data[1,1,1:5,1]
        #print self.solver.test_nets[0].params['conv1'][0].data[1,1,1:5,1]
        num_data_epoch_train = 540
        num_data_epoch_test = 240
        tStamp = '-Timestamp-{:%Y-%m-%d-%H:%M:%S}'.format(
            datetime.datetime.now())
        plt.ion()
        try:
            for k in range(100):
                disLoss = 0
                simLoss = 0
                simC = 0
                disC = 0
                plot_data_d = np.zeros((0, 2))
                plot_data_s = np.zeros((0, 2))
                plot_data_id_l1 = np.zeros((0, 2))
                plot_data_id_l2 = np.zeros((0, 2))
                lossId1s = 0
                lossId2s = 0
                for i in range(num_data_epoch_train):
                    self.solver.step(1)

                    lossCo = 0.2 * self.solver.net.blobs['cont_loss'].data
                    lossId1 = 0.4 * self.solver.net.blobs[
                        'softmax_loss_1'].data
                    lossId2 = 0.4 * self.solver.net.blobs[
                        'softmax_loss_2'].data
                    if self.solver.net.blobs['sim'].data == 1:
                        simC += 1
                        simLoss += lossCo
                        plot_data_s = np.vstack(
                            (plot_data_s, [k + 0.5, lossCo]))
                    else:
                        disC += 1
                        disLoss += lossCo
                        plot_data_d = np.vstack((plot_data_d, [k, lossCo]))
                    lossId1s += lossId1
                    lossId2s += lossId2

                    #print "sim", self.solver.net.blobs[
                    #    'sim'].data, "cont loss", lossCo, "id1", lossId1, "id2", lossId2
                plot_data_id_l1 = np.vstack(
                    (plot_data_id_l1, [k, lossId1s / num_data_epoch_train]))
                plot_data_id_l2 = np.vstack(
                    (plot_data_id_l2, [k, lossId2s / num_data_epoch_train]))
                print k, "cont net loss", simLoss / (simC + 0.1), disLoss / (
                    disC + 0.1), simC, disC, "Id net loss", lossId1s, lossId2s
                plt.figure(1)
                plt.xlim(-0.5, 100)
                plt.title(str(self.netSize) + "train errors")
                plt.plot(plot_data_s[:, 0], plot_data_s[:, 1], 'r.')
                plt.plot(plot_data_d[:, 0], plot_data_d[:, 1], 'b.')
                plt.pause(0.05)

                plt.figure(2)
                plt.xlim(-0.5, 100)
                plt.title(str(self.netSize) + "train softmax loss")
                plt.plot(plot_data_id_l1[:, 0], plot_data_id_l1[:, 1], 'r.')
                plt.plot(plot_data_id_l2[:, 0], plot_data_id_l2[:, 1], 'b.')
                plt.pause(0.05)

                if k % 5 == 0:
                    acc1 = np.zeros(self.class_size)
                    acc2 = np.zeros(self.class_size)
                    fre1 = np.zeros(self.class_size)
                    fre2 = np.zeros(self.class_size)
                    plot_acc1 = np.zeros((0, 2))
                    plot_acc2 = np.zeros((0, 2))
                    confusion_dis = np.zeros(
                        (self.class_size, self.class_size))
                    for i in range(num_data_epoch_test):
                        loss1 = self.solver.test_nets[0].forward()
                        #print i, loss1, loss1['sim'], loss1['euc_dist']

                        if loss1['sim'] == 1:
                            simC += 1
                            simLoss += loss1['euc_dist']
                            plot_data_s = np.vstack(
                                (plot_data_s, [k + 0.5, loss1['euc_dist']]))
                        else:
                            disC += 1
                            disLoss += loss1['euc_dist']
                            plot_data_d = np.vstack(
                                (plot_data_d, [k, loss1['euc_dist']]))
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
                    acc1 = acc1 / (fre1 + 0.1)
                    acc2 = acc2 / (fre2 + 0.1)
                    plot_acc1 = np.vstack((plot_acc1, [k, netacc1]))
                    plot_acc2 = np.vstack((plot_acc2, [k, netacc2]))
                    print "testing**** net loss", simLoss / (
                        simC + 0.1), disLoss / (disC + 0.1), simC, disC
                    #print confusion_dis
                    print acc1, acc2
                    print netacc
                    plt.figure(3)
                    plt.xlim(-0.5, 100)
                    plt.title(str(self.netSize) + "test accuracy")
                    plt.plot(k, netacc1, 'r.')
                    plt.plot(k, netacc2, 'b.')
                    plt.pause(0.05)

                    plt.figure(4)
                    plt.xlim(-0.5, 100)
                    plt.title(str(self.netSize) + "test distance")
                    plt.plot(plot_data_s[:, 0], plot_data_s[:, 1], 'r.')
                    plt.plot(plot_data_d[:, 0], plot_data_d[:, 1], 'b.')
                    plt.pause(0.05)

                if k % 1 == 0:
                    preName = 'modifiedNetResults/' + 'Modified-netsize-' + str(
                        self.netSize) + '-epoch-' + str(
                            k) + '-tstamp-' + tStamp
                    self.solver.net.save(preName + '-net.caffemodel')

        except KeyboardInterrupt:
            pass

        preName = 'modifiedNetResults/' + 'Modified-netsize-' + str(
            self.netSize) + '-epoch-' + str(k) + '-tstamp-' + tStamp
        plt.ioff()

        plt.figure(1).savefig(preName + '-train-d-error.png')
        plt.figure(1).savefig(preName + '-train-s-error.png')
        plt.figure(3).savefig(preName + '-test-acc.png')
        plt.figure(4).savefig(preName + '-test-dist.png')
        self.solver.net.save(preName + '-net-final.caffemodel')
        plt.close('all')

    def visualize(self, fileName, tech, compare):
        ''' Visualizing using gray occlusion patches or gradients of input image
        '''
        tStamp = '-Timestamp-{:%Y-%m-%d-%H:%M:%S}'.format(
            datetime.datetime.now())
        f = open(fileName)
        lines = [line.rstrip('\n') for line in f]
        imageDict = {}
        imlist = []
        size_patch = 100
        stride = 10
        highlighted_ratio = 0.25

        for i in lines:
            temp = i.split(' ')
            imageDict[temp[0]] = int(temp[1]) - self.class_adju
            imlist.append(temp[0])
        if tech == 'both':
            self.ov_lap_heat_map = np.zeros(len(imlist))
        if tech == 'both' or tech == 'grad':
            self.iterations = 5
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
                    im1] + '-' + str(size_patch) + '-' + str(
                        stride) + '-' + '-M-nSize-' + str(
                            self.netSize) + '-tstamp-' + tStamp
                cv2.imwrite(preName + '.png', im_gen_occ)
            elif tech == 'grad':
                #using graidents wrt input to visualize
                print 'generating heat map for ', imageDict[imlist[
                    im1]], imlist[im1], 'using gradients wrt image'

                im_gen_grad, heat_map_grad, heat_map_raw_grad = self.generate_heat_map_gradients(
                    imageDict, imlist, im1, ratio=highlighted_ratio)
                preName = 'modifiedNetResults_visu_grad/' + imlist[
                    im1] + '-itera-' + str(
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
                    im1] + '-' + str(size_patch) + '-' + str(
                        stride) + '-' + '-M-nSize-' + str(
                            self.netSize) + '-tstamp-' + tStamp

                im_gen_grad, heat_map_grad, heat_map_raw_grad = self.generate_heat_map_gradients(
                    imageDict, imlist, im1, ratio=highlighted_ratio)
                preName_grad = 'modifiedNetResults_visu_grad/' + imlist[
                    im1] + '-itera-' + str(
                        self.iterations) + '-M-nSize-' + str(
                            self.netSize) + '-tstamp-' + tStamp

                save = 1
                if save == 1:
                    cv2.imwrite(preName_grad + '.png', im_gen_grad)
                    cv2.imwrite(preName_occ + '.png', im_gen_occ)
                #code for comparision
                if compare == 1:
                    self.ov_lap_heat_map[i] = h2._analyse_heat_maps(
                        heat_map_occ, heat_map_grad)
                    print "overlap percentage", self.ov_lap_heat_map[i]
        if compare == 1:
            print self.ov_lap_heat_map
            import IPython
            IPython.embed()

    def generate_heat_map_softmax(self,
                                  imageDict,
                                  imlist,
                                  im1,
                                  size_patch,
                                  stride,
                                  ratio=0.25):
        offset = 228
        l_blobs_im1, l_occ_map = h2._get_occluded_image_blobs(
            img_name=self.data_folder + imlist[im1],
            size_patch=size_patch,
            stride=stride,
            meanarr=self.meanarr,
            im_target_size=self.im_target_size)
        #print "no of occluded maps ", len(l_blobs_im1)
        blobs = {'data': None}
        heat_map = np.zeros((self.im_target_size, self.im_target_size))

        for i in range(len(l_blobs_im1)):
            blobs['data'] = l_blobs_im1[i]
            blobs_out1 = self.siameseTestNet.forward(data=blobs['data'].astype(
                np.float32, copy=True))

            p = self.siameseTestNet.blobs['fc9_f'].data[0]
            p = p - p.min()
            p = p / p.sum()
            prob1 = p[imageDict[imlist[im1]]]
            heat_map += l_occ_map[i] * prob1
        heat_map = heat_map[:offset, :offset]
        heat_map_o = (heat_map - heat_map.min()) / (
            heat_map.max() - heat_map.min())
        img1 = h2._load_image(
            self.data_folder + imlist[im1], im_target_size=self.im_target_size)
        heat_map = heat_map_o.copy()

        #invert the heat map
        heat_map = 1.0 - heat_map
        threshold = h2._find_threshold(heat_map, ratio)

        heat_map[heat_map < threshold] = 0
        heat_map[heat_map >= threshold] = 1
        heat_map *= 1
        #plt.matshow(heat_map)
        #plt.colorbar()
        #plt.show()

        img1_o = img1.copy()
        temp = np.sum(img1, axis=2) / 3.0
        img1[:, :, 2] = temp
        img1[:, :, 1] = temp
        img1[:, :, 0] = temp
        img1[:, :, 2] += heat_map * 100

        img1 = h2._round_image(img1)
        #cv2.imshow('image', img1.astype(np.uint8))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return img1.astype(np.uint8), heat_map.astype(np.uint8), heat_map_o

    def generate_heat_map_gradients(self, imageDict, imlist, im1, ratio=0.25):
        blobs = {'data': None}
        blobs['data'] = h2._get_image_blob(
            img_name=self.data_folder + imlist[im1],
            meanarr=self.meanarr,
            im_target_size=self.im_target_size)

        label_index = imageDict[imlist[im1]] - self.class_adju
        caffe_data = np.random.random(
            (1, 3, self.im_target_size, self.im_target_size))
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
        #iterations = 5
        if self.iterations > 0:
            heat_map = cv2.dilate(
                heat_map_raw, kernel, iterations=self.iterations)
        else:
            heat_map = heat_map_raw.copy()

        threshold = h2._find_threshold(heat_map, ratio=ratio)
        heat_map[heat_map < threshold] = 0
        heat_map[heat_map >= threshold] = 1
        #plt.matshow(heat_map)
        #plt.colorbar()
        #plt.show()

        img1 = h2._load_image(
            self.data_folder + imlist[im1], im_target_size=self.im_target_size)
        temp = np.sum(img1, axis=2) / 3.0
        img1[:, :, 2] = temp
        img1[:, :, 1] = temp
        img1[:, :, 0] = temp
        img1[:, :, 2] += heat_map * 100

        img1 = h2._round_image(img1)
        #plt.figure()
        #plt.imshow(img1.astype(np.uint8))
        #plt.show()
        return img1.astype(np.uint8), heat_map.astype(np.uint8), saliency


def siameseTrainer(siameseSolver,
                   fileName,
                   pretrained_model,
                   pretrainedSiameseModel,
                   testProto,
                   pretrained_model_proto,
                   train=1,
                   visu=0,
                   testProto1=None,
                   viz_tech=None,
                   compare=0,
                   netSize=1000):
    sw = SiameseTrainWrapper(
        siameseSolver,
        pretrainedSiameseModel=pretrainedSiameseModel,
        pretrained_model=pretrained_model,
        pretrained_model_proto=pretrained_model_proto,
        testProto=testProto,
        testProto1=testProto1,
        train=train,
        visu=visu,
        tech=viz_tech,
        netSize=netSize)
    if train == 1:
        print "training"
        sw.trainTest()
    elif visu == 0:
        print "testing with ", pretrainedSiameseModel
        sw.test(fileName)
    else:
        print 'visalizing with ', pretrainedSiameseModel
        sw.visualize(fileName, tech=viz_tech, compare=compare)
