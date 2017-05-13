#/usr/bin/python
# --------------------------------------------------------
# floor_recog
# Written by Sai Prabhakar
# CMU-RI Masters
# --------------------------------------------------------
import numpy as np
import cv2
from pythonlayers.helpers import *
#import modifiedSiamese.helpers2 as h2
import helpers2 as h2
import matplotlib.pyplot as plt

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

import pickle
from ipdb import set_trace as debug
'''
Script to generate images for paper
'''


def run():
    visu_file = 'visu/floor_NetResults_visu/IMG_20161114_202219--M-nSize-1000-tstamp---visualizations.pickle'
    combine_tech_s = ["blur", "black"]
    combine_tech = "blur"
    #visu_all_save_dir = "visu/" + net + '_NetResults_visu' + data_index
    data_folder = 'data/data_floor/'
    save_place = 'paper_imgs/I1_'

    with open(visu_file) as f:
        im_name, class_index, tech_s, size_patch_s, outputBlobName, outputLayerName, dilate_iteration_s, heat_map_raw_occ_s, heat_map_raw_grad_s, heat_raw_exci_s = pickle.load(
            f)
    print data_folder + im_name
    img = h2._load_image(data_folder + im_name, 227)
    size = 17
    kernel = np.ones((size, size), np.float32) / (size * size)
    mod_img = cv2.filter2D(img, -1, kernel)

    cv2.imwrite(save_place + 'blurr.png', mod_img)
    cv2.imwrite(save_place + 'orig.png', img)

    #heat_map_raw = heat_map_raw_occ_s[0]
    heat_map_raw = heat_raw_exci_s[0]
    heat_map = heat_map_raw.copy()
    ratio = 0.05

    threshold = h2._find_threshold(heat_map, ratio=ratio)
    heat_map[heat_map < threshold] = 0
    heat_map[heat_map >= threshold] = 1

    heat_map1 = heat_map
    kernel = np.ones((3, 3), np.uint8)
    heat_map2 = cv2.dilate(heat_map1, kernel, iterations=30)

    im_m1 = img.copy()
    temp = np.sum(im_m1, axis=2) / 3.0
    im_m1[:, :, 2] = temp
    im_m1[:, :, 1] = temp
    im_m1[:, :, 0] = temp
    im_m1[:, :, 2] += heat_map1 * 100
    im_m1 = h2._round_image(im_m1)

    im_m2 = img.copy()
    temp = np.sum(im_m2, axis=2) / 3.0
    im_m2[:, :, 2] = temp
    im_m2[:, :, 1] = temp
    im_m2[:, :, 0] = temp
    im_m2[:, :, 2] += heat_map2 * 100
    im_m2 = h2._round_image(im_m2)

    cv2.imwrite(save_place + 'mask1.png', im_m1)
    cv2.imwrite(save_place + 'mask2.png', im_m2)

    c_img1 = h2._combine_images(mod_img, heat_map1, img)
    c_img2 = h2._combine_images(mod_img, heat_map2, img)
    c_img3 = h2._combine_images(np.zeros(mod_img.shape), heat_map1, img)
    cv2.imwrite(save_place + 'hyb1.png', c_img1)
    cv2.imwrite(save_place + 'hyb2.png', c_img2)
    cv2.imwrite(save_place + 'hyb3.png', c_img3)

    plt.tick_params(
        bottom='off',
        top='off',
        left='off',
        right='off',
        labelbottom='off',
        labeltop='off',
        labelleft='off',
        labelright='off')
    plt.imshow(img.astype(np.uint8))
    plt.imshow(heat_map_raw, cmap="jet", interpolation="nearest", alpha=0.4)
    plt.savefig(save_place + 'heat_map.png', bbox_inches='tight', pad_inches=0)


def get_masked_image(img, heat_map, ratio):
    threshold = h2._find_threshold(heat_map, ratio=ratio)
    heat_map[heat_map < threshold] = 0
    heat_map[heat_map >= threshold] = 1

    im_m1 = img.copy()
    temp = np.sum(im_m1, axis=2) / 3.0
    im_m1[:, :, 2] = temp
    im_m1[:, :, 1] = temp
    im_m1[:, :, 0] = temp
    im_m1[:, :, 2] += heat_map * 100
    im_m1 = h2._round_image(im_m1)
    return im_m1


def get_masked_image_from_mask(img, heat_map, ratio):

    im_m1 = img.copy()
    temp = np.sum(im_m1, axis=2) / 3.0
    im_m1[:, :, 2] = temp
    im_m1[:, :, 1] = temp
    im_m1[:, :, 0] = temp
    im_m1[:, :, 2] += heat_map * 100
    im_m1 = h2._round_image(im_m1)
    return im_m1


def run2():
    visu_file = 'visu/places_NetResults_visu1/Places365_val_00000024--M-nSize-1000-tstamp---visualizations.pickle'
    #visu_file ='visu/floor_NetResults_visu/IMG_20161114_201944--M-nSize-1000-tstamp---visualizations.pickle'
    data_folder = 'data/data_places/val_256/'
    #data_folder = 'data/data_floor/'
    save_place = 'paper_imgs/I2_p_'
    ratio = 0.25
    kernel = np.ones((3, 3), np.uint8)

    with open(visu_file) as f:
        im_name, class_index, tech_s, size_patch_s, outputBlobName, outputLayerName, dilate_iteration_s, heat_map_raw_occ_s, heat_map_raw_grad_s, heat_raw_exci_s = pickle.load(
            f)
    print data_folder + im_name
    img = h2._load_image(data_folder + im_name, 227)

    im_m = get_masked_image(img, heat_map_raw_occ_s[0], ratio)
    cv2.imwrite(save_place + 'occ_10.png', im_m)
    im_m = get_masked_image(img, heat_map_raw_occ_s[1], ratio)
    cv2.imwrite(save_place + 'occ_50.png', im_m)
    im_m = get_masked_image(img, heat_map_raw_occ_s[2], ratio)
    cv2.imwrite(save_place + 'occ_100.png', im_m)

    im_m = get_masked_image(img, heat_map_raw_grad_s[0], ratio)
    cv2.imwrite(save_place + 'grad_0.png', im_m)
    im_m = get_masked_image(img, heat_map_raw_grad_s[1], ratio)
    cv2.imwrite(save_place + 'grad_2.png', im_m)
    im_m = get_masked_image(img, heat_map_raw_grad_s[2], ratio)
    cv2.imwrite(save_place + 'grad_5.png', im_m)


def run3():
    visu_file = 'visu/places_NetResults_visu1/Places365_val_00000112--M-nSize-1000-tstamp---visualizations.pickle'
    data_folder = 'data/data_places/val_256/'
    save_place = 'paper_imgs/I3_p_'

    #visu_file ='visu/floor_NetResults_visu/IMG_20161114_202108--M-nSize-1000-tstamp---visualizations.pickle'
    #data_folder = 'data/data_floor/'
    #save_place = 'paper_imgs/I3_f_'

    ratio = 0.25
    kernel = np.ones((3, 3), np.uint8)

    with open(visu_file) as f:
        im_name, class_index, tech_s, size_patch_s, outputBlobName, outputLayerName, dilate_iteration_s, heat_map_raw_occ_s, heat_map_raw_grad_s, heat_raw_exci_s = pickle.load(
            f)
    print data_folder + im_name
    img = h2._load_image(data_folder + im_name, 227)

    im_m = get_masked_image(img, heat_map_raw_occ_s[0], ratio)
    cv2.imwrite(save_place + 'occ_10.png', im_m)

    im_m = get_masked_image(img, heat_map_raw_grad_s[2], ratio)
    cv2.imwrite(save_place + 'grad_5.png', im_m)

    im_m = get_masked_image(img, heat_raw_exci_s[0], ratio)
    cv2.imwrite(save_place + 'cmwp.png', im_m)

    raw1 = heat_map_raw_grad_s[2]
    raw2 = heat_map_raw_occ_s[0]
    heat_map_com = h2._get_combined_heat_mask(raw1, raw2, ratio, 'inter')
    im_m = get_masked_image_from_mask(img, heat_map_com, ratio)
    cv2.imwrite(save_place + 'og.png', im_m)

    raw1 = heat_map_raw_grad_s[2]
    raw2 = heat_raw_exci_s[0]
    heat_map_com = h2._get_combined_heat_mask(raw1, raw2, ratio, 'inter')
    im_m = get_masked_image_from_mask(img, heat_map_com, ratio)
    cv2.imwrite(save_place + 'ge.png', im_m)

    raw1 = heat_map_raw_occ_s[0]
    raw2 = heat_raw_exci_s[0]
    heat_map_com = h2._get_combined_heat_mask(raw1, raw2, ratio, 'inter')
    im_m = get_masked_image_from_mask(img, heat_map_com, ratio)
    cv2.imwrite(save_place + 'oe.png', im_m)
