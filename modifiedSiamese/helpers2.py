#/usr/bin/python
# --------------------------------------------------------
# floor_recog
# Written by Sai Prabhakar
# CMU-RI Masters
# --------------------------------------------------------

import numpy as np
import cv2
from pythonlayers.helpers import *
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
from ipdb import set_trace as debug
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

from skimage import transform


def _get_prob(p, class_index_n):
    p = p - p.min()
    p = p / p.sum()
    prob1 = p[class_index_n]
    return prob1


def _visu_heat_map(img1, heat_map):
    temp = np.sum(img1, axis=2) / 3.0
    img1[:, :, 2] = temp
    img1[:, :, 1] = temp
    img1[:, :, 0] = temp
    img1[:, :, 2] += heat_map * 100
    img1 = _round_image(img1)
    return img1


def plot_images(img1, img2, img3):
    plt.subplot(131), plt.imshow(img1[:, :, ::-1].astype(np.uint8)), plt.title(
        'Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(img2[:, :, ::-1].astype(np.uint8)), plt.title(
        'Averaging')
    plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(img3[:, :, ::-1].astype(np.uint8)), plt.title(
        'final')
    plt.xticks([]), plt.yticks([])
    plt.show()


def _combine_images(img1, mask, img2):
    """Copy masked regions of img2 to img1
    img2 - original image
    img1 - blur image
    mask - visualized mask
    """
    n_mask = abs(1 - mask)
    return np.stack(
        [mask * img2[:, :, 0] + n_mask * img1[:, :, 0],
         mask * img2[:, :, 1] + n_mask * img1[:, :, 1],
         mask * img2[:, :, 2] + n_mask * img1[:, :, 2]],
        axis=2)


def _analyse_heat_maps(heat_map1, heat_map2):
    ovlap_count = (heat_map1 & heat_map2).sum().astype(float)
    union_count = (heat_map1 | heat_map2).sum()
    iou_percent = 100 * ovlap_count / union_count
    #print ovlap_count, union_count, iou_percent
    return iou_percent


def _load_mean_binaryproto(
        fileName='placesOriginalModel/places205CNN_mean.binaryproto',
        im_target_size=227):
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(fileName, 'rb').read()
    blob.ParseFromString(data)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    # changing order of rows, colmn, channel, batch_size
    arr = np.squeeze(arr.transpose((2, 3, 1, 0)))
    im_scaley = float(im_target_size) / float(256)
    im_scalex = float(im_target_size) / float(256)
    meanarr = cv2.resize(
        arr,
        None,
        None,
        fx=im_scalex,
        fy=im_scaley,
        interpolation=cv2.INTER_LINEAR)
    return meanarr


def _get_mask_from_raw_map(raw_map, ratio):
    threshold = _find_threshold(raw_map, ratio)
    heat_map = np.zeros(raw_map.shape)
    heat_map[raw_map < threshold] = 0
    heat_map[raw_map >= threshold] = 1
    return heat_map


def _get_combined_heat_mask(raw_map1, raw_map2, mask_ratio, tech):
    hm1 = _get_mask_from_raw_map(raw_map1, mask_ratio)
    hm2 = _get_mask_from_raw_map(raw_map2, mask_ratio)
    if tech == 'inter':
        hm = (hm1.astype(bool) & hm2.astype(bool)).astype(float)
    elif tech == 'neg':
        hm = (hm1.astype(bool) & ~hm2.astype(bool)).astype(float)
    return hm


def _find_threshold(h_map, ratio):
    assert ratio <= 1.0
    temp = np.sort(
        h_map.reshape(h_map.shape[0] * h_map.shape[1]), kind='mergesort')
    return temp[int(min(1 - ratio, 0.9) * len(temp))]


def _round_image(img):
    img[img > 255] = 255
    img[img < 0] = 0
    return img


def _get_image_blob(img_name, meanarr, im_target_size):
    im = _load_image(img_name, im_target_size)
    processed_ims = im - meanarr
    blob = im_to_blob(processed_ims)
    return blob


def _get_image_blob_from_image(im, meanarr, im_target_size):
    processed_ims = im - meanarr
    blob = im_to_blob(processed_ims)
    return blob


def _get_exci_final_map(attMap, size):
    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap /= attMap.max()
    #attMap = transform.resize(attMap, (img.shape[:2]), order = 3, mode = 'nearest')
    attMap = transform.resize(attMap, (size, size), order=3, mode='edge')
    #if blur:
    #    attMap = filters.gaussian_filter(attMap, 0.02*max(img.shape[:2]))
    attMap -= attMap.min()
    attMap /= attMap.max()
    return attMap


def _load_image(img_name, im_target_size):
    im = cv2.imread(img_name)
    im = im.astype(np.float32, copy=True)

    min_curr_size = min(im.shape[:2])
    im_scale = float(im_target_size) / float(min_curr_size)

    #im_scaley = float(im_target_size) / float(im_size[0])
    #im_scalex = float(im_target_size) / float(im_size[1])
    im = cv2.resize(
        im[:min_curr_size, :min_curr_size, :],
        None,
        None,
        fx=im_scale,
        fy=im_scale,
        interpolation=cv2.INTER_LINEAR)
    return im.copy()


def _get_occluded_image_blob(im, size_patch, cR, cC, meanarr, im_target_size):
    l_blob = []
    l_occ_map = []
    #import IPython
    #IPython.embed()
    occluded_image, occ_map = _occlude_image(im.copy(), cR, cC, size_patch,
                                             im_target_size)
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.imshow('image',occluded_image.astype(np.uint8))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    processed_ims = occluded_image - meanarr
    l_blob = im_to_blob(processed_ims)
    l_occ_map = occ_map
    return l_blob, l_occ_map


def _get_coordinates(cR, cC, size_patch, maxRow, maxCol):
    r1 = cR
    r2 = cR + size_patch
    c1 = cC
    c2 = cC + size_patch
    if r1 < 0:
        r1 = 0
    if r2 - 1 > maxRow - 1:
        r2 = maxRow
    if c1 < 0:
        c1 = 0
    if c2 - 1 > maxCol - 1:
        c2 = maxCol

    return r1, r2, c1, c2


def _find_percentage_mask(mask):
    return 100 * float(mask.sum()) / (mask.size)


def _occlude_image(im, cR, cC, size_patch, im_target_size):
    """creates gray patches in image."""
    r1, r2, c1, c2 = _get_coordinates(cR, cC, size_patch, im.shape[0],
                                      im.shape[1])
    im[r1:r2, c1:c2, :] = 127.5
    occ_map = np.ones((im_target_size, im_target_size))
    occ_map[r1:r2, c1:c2] = 0
    return im, occ_map
